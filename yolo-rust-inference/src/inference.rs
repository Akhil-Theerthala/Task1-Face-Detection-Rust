// This file contains the InferencePipeline struct, which is responsible for loading the YOLOv11 ONNX model, processing images, and generating predictions.

use image::GenericImageView;
use anyhow::Result;
use std::path::Path;
use std::process::Command;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub enum InferenceBackend {
    Mock,
    Python,
    // Future: Native(OnnxSession),
}

pub struct InferencePipeline {
    model_path: Option<String>,
    backend: InferenceBackend,
    nms_threshold: f32,
}

impl InferencePipeline {
    pub fn new() -> Self {
        InferencePipeline {
            model_path: None,
            backend: InferenceBackend::Mock,
            nms_threshold: 0.5,
        }
    }

    pub fn with_nms_threshold(mut self, threshold: f32) -> Self {
        self.nms_threshold = threshold;
        self
    }

    pub fn load_model<P: AsRef<Path>>(&mut self, model_path: P) -> Result<()> {
        let path = model_path.as_ref();
        
        // Check if model file exists
        if !path.exists() {
            println!("Model file not found: {:?}", path);
            println!("Using mock inference backend");
            self.backend = InferenceBackend::Mock;
            return Ok(());
        }
        
        self.model_path = Some(path.to_string_lossy().to_string());
        
        // Try to determine the best backend
        if self.check_python_backend() {
            self.backend = InferenceBackend::Python;
            println!("Using Python inference backend");
        } else {
            self.backend = InferenceBackend::Mock;
            println!("Using mock inference backend");
        }
        
        println!("Model loaded from: {:?}", path);
        Ok(())
    }

    fn check_python_backend(&self) -> bool {
        // Check if Python script exists and dependencies are available
        let script_path = Path::new("yolo_inference.py");
        if !script_path.exists() {
            return false;
        }
        
        // Test if Python and required packages are available
        match Command::new("/home/theerthala/Documents/repos/Face-Generation/.venv/bin/python")
            .args(&["-c", "import cv2, numpy, onnxruntime; print('OK')"])
            .output()
        {
            Ok(output) => {
                let output_str = String::from_utf8_lossy(&output.stdout);
                output_str.trim() == "OK"
            }
            Err(_) => false,
        }
    }

    pub fn run_inference(&self, image_path: &Path) -> Result<Vec<Detection>> {
        match &self.backend {
            InferenceBackend::Python => self.run_python_inference(image_path),
            InferenceBackend::Mock => self.run_mock_inference(image_path),
        }
    }

    fn run_python_inference(&self, image_path: &Path) -> Result<Vec<Detection>> {
        let model_path = self.model_path.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Model path not set"))?;
        
        let output = Command::new("/home/theerthala/Documents/repos/Face-Generation/.venv/bin/python")
            .args(&[
                "yolo_inference.py",
                model_path,
                &image_path.to_string_lossy(),
                &self.nms_threshold.to_string(),
            ])
            .output()?;
        
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow::anyhow!("Python inference failed: {}", stderr));
        }
        
        let stdout = String::from_utf8_lossy(&output.stdout);
        let detections: Vec<PythonDetection> = serde_json::from_str(&stdout)?;
        
        // Convert Python detections to our Detection format
        let rust_detections = detections.into_iter().map(|d| Detection {
            x: d.x,
            y: d.y,
            width: d.width,
            height: d.height,
            confidence: d.confidence,
            class_id: d.class_id,
        }).collect();
        
        Ok(rust_detections)
    }

    fn run_mock_inference(&self, image_path: &Path) -> Result<Vec<Detection>> {
        // Load and analyze the image
        let img = image::open(image_path)?;
        let (width, height) = img.dimensions();
        
        println!("Processing image: {:?} ({}x{})", image_path, width, height);
        
        // Generate mock detections
        let detections = self.generate_mock_detections(width, height);
        
        println!("Generated {} mock detections", detections.len());
        
        Ok(detections)
    }

    fn generate_mock_detections(&self, width: u32, height: u32) -> Vec<Detection> {
        let mut detections = Vec::new();
        let mut rng = rand::thread_rng();
        
        // Generate 1-3 random detections for demonstration
        let num_detections = rng.gen_range(1..=3);
        
        for _i in 0..num_detections {
            let x = rng.gen_range(0.0..(width as f32 * 0.8));
            let y = rng.gen_range(0.0..(height as f32 * 0.8));
            let det_width = rng.gen_range(50.0..(width as f32 * 0.2));
            let det_height = rng.gen_range(50.0..(height as f32 * 0.2));
            let confidence = rng.gen_range(0.6..0.95);
            let class_id = rng.gen_range(0..80); // COCO has 80 classes
            
            detections.push(Detection {
                x,
                y,
                width: det_width,
                height: det_height,
                confidence,
                class_id,
            });
        }
        
        detections
    }
}

#[derive(Debug, Clone, Deserialize)]
struct PythonDetection {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    confidence: f32,
    class_id: u32,
}

#[derive(Debug, Clone)]
pub struct Detection {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub confidence: f32,
    pub class_id: u32,
}

impl Detection {
    pub fn get_class_name(&self) -> &'static str {
        // Check if this might be a face detection model (single class)
        if self.class_id == 0 {
            return "face";
        }
        
        // COCO class names for reference
        const COCO_CLASSES: &[&str] = &[
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
            "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
            "toothbrush"
        ];
        
        COCO_CLASSES.get(self.class_id as usize).unwrap_or(&"unknown")
    }
}
