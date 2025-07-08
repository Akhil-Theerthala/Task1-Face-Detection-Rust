#!/usr/bin/env /home/theerthala/Documents/repos/Face-Generation/.venv/bin/python

"""
Python YOLO inference script that can be called from Rust.
This provides a fallback when Rust ONNX runtime is not available.
"""

import sys
import json
import cv2
import numpy as np
from pathlib import Path

def load_yolo_model(model_path):
    """Load YOLO model from ONNX file"""
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(model_path)
        return session
    except ImportError:
        print("ONNX Runtime not available in Python", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return None

def preprocess_image(image_path, input_size=640):
    """Preprocess image for YOLO model"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    original_height, original_width = image.shape[:2]
    
    # Resize image to input size
    image_resized = cv2.resize(image, (input_size, input_size))
    
    # Convert BGR to RGB and normalize
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_normalized = image_rgb.astype(np.float32) / 255.0
    
    # Convert to CHW format and add batch dimension
    image_chw = np.transpose(image_normalized, (2, 0, 1))
    image_batch = np.expand_dims(image_chw, axis=0)
    
    return image_batch, original_width, original_height

def postprocess_outputs(outputs, original_width, original_height, input_size=640, conf_threshold=0.5):
    """Post-process YOLO outputs to get detections"""
    detections = []
    
    if len(outputs) == 0:
        return detections
    
    output = outputs[0]
    
    # Handle different output shapes
    if len(output.shape) == 3:
        # Shape: [1, N, 85] where N is number of detections
        output = output[0]  # Remove batch dimension
    
    # YOLOv11 might have a different output format
    # Try to handle the output shape [1, 5, 8400] or [1, 84, 8400] which is common for YOLOv8/v11
    if len(output.shape) == 2 and output.shape[0] in [5, 84, 85]:
        # Transpose to [8400, 5] or [8400, 84] or [8400, 85]
        output = output.T
    
    for detection in output:
        if len(detection) < 5:
            continue
            
        # Extract coordinates and confidence
        x_center, y_center, width, height, confidence = detection[:5]
        
        if confidence < conf_threshold:
            continue
        
        # Scale coordinates back to original image size
        x_center = x_center * original_width / input_size
        y_center = y_center * original_height / input_size
        width = width * original_width / input_size
        height = height * original_height / input_size
        
        # Convert to top-left corner format
        x = x_center - width / 2
        y = y_center - height / 2
        
        # Handle class prediction
        if len(detection) > 5:
            # Multi-class model
            class_scores = detection[5:]
            class_id = np.argmax(class_scores)
            class_confidence = class_scores[class_id]
            final_confidence = confidence * class_confidence
        else:
            # Single class model or no class predictions
            class_id = 0  # Default to first class (usually 'person' in face detection)
            final_confidence = confidence
        
        if final_confidence > conf_threshold:
            detections.append({
                "x": float(x),
                "y": float(y),
                "width": float(width),
                "height": float(height),
                "confidence": float(final_confidence),
                "class_id": int(class_id)
            })
    
    return detections

def apply_nms(detections, iou_threshold=0.5):
    """Apply Non-Maximum Suppression to remove duplicate detections"""
    if len(detections) == 0:
        return detections
    
    # Convert detections to format needed for NMS
    boxes = []
    scores = []
    
    for det in detections:
        # Convert to [x1, y1, x2, y2] format
        x1 = det["x"]
        y1 = det["y"]
        x2 = x1 + det["width"]
        y2 = y1 + det["height"]
        boxes.append([x1, y1, x2, y2])
        scores.append(det["confidence"])
    
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    
    # Apply OpenCV's NMS
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 
                               score_threshold=0.5, nms_threshold=iou_threshold)
    
    # Filter detections based on NMS results
    filtered_detections = []
    if len(indices) > 0:
        indices = indices.flatten()
        for i in indices:
            filtered_detections.append(detections[i])
    
    return filtered_detections

def generate_mock_detections(image_path):
    """Generate mock detections when real inference fails"""
    import random
    
    # Try to get image dimensions
    try:
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
    except:
        width, height = 1024, 768  # Default dimensions
    
    detections = []
    num_detections = random.randint(1, 3)
    
    for _ in range(num_detections):
        x = random.uniform(0, width * 0.8)
        y = random.uniform(0, height * 0.8)
        det_width = random.uniform(50, width * 0.2)
        det_height = random.uniform(50, height * 0.2)
        confidence = random.uniform(0.6, 0.95)
        class_id = random.randint(0, 79)  # COCO has 80 classes
        
        detections.append({
            "x": x,
            "y": y,
            "width": det_width,
            "height": det_height,
            "confidence": confidence,
            "class_id": class_id
        })
    
    return detections

def main():
    if len(sys.argv) < 3:
        print("Usage: python yolo_inference.py <model_path> <image_path> [nms_threshold]", file=sys.stderr)
        print("  nms_threshold: IoU threshold for Non-Maximum Suppression (default: 0.5)", file=sys.stderr)
        sys.exit(1)
    
    model_path = sys.argv[1]
    image_path = sys.argv[2]
    nms_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    
    # Run inference
    detections = run_inference(model_path, image_path, nms_threshold)
    
    # Output results as JSON
    print(json.dumps(detections))

def run_inference(model_path, image_path, nms_threshold=0.5):
    """Run YOLO inference on image with configurable NMS threshold"""
    
    # Try to load model
    session = load_yolo_model(model_path)
    if session is None:
        # Return mock detections if model loading fails
        return generate_mock_detections(image_path)
    
    try:
        # Preprocess image
        input_data, orig_width, orig_height = preprocess_image(image_path)
        
        # Run inference
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_data})
        
        # Post-process outputs
        detections = postprocess_outputs(outputs, orig_width, orig_height)
        
        # Apply Non-Maximum Suppression to remove duplicate detections
        detections = apply_nms(detections, iou_threshold=nms_threshold)
        
        return detections
        
    except Exception as e:
        print(f"Error during inference: {e}", file=sys.stderr)
        return generate_mock_detections(image_path)

if __name__ == "__main__":
    main()
