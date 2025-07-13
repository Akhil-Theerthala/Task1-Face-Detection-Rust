****# YOLO Rust Inference

A face detection inference pipeline using YOLO models with Rust and Python backends. Features automatic Non-Maximum Suppression (NMS) for clean, duplicate-free results.


## Setup Instructions

1. **Build the Rust application:**
   ```bash
   cargo build --release
   ```

2. **Python dependencies** (automatically managed):
   The project uses a pre-configured Python virtual environment with:
   - `onnxruntime` - ONNX model execution
   - `opencv-python` - Image processing and NMS
   - `numpy` - Numerical computations

3. **Model:** The `best.onnx` face detection model is included in the `models/` directory

## LLM Usage
Since I have never had the chance to work with Rust, I could only make use of the agentic-development capabilities of github copilot to work on the majority of this part. I only intervened in the development process to understand the flow of information and to see if there are any severe errors in the inference pipeline. The entire "rust" code was vibe-coded. 

### YOLOv11 Training process
The `WIDERFACE` dataset was used to finetune the YOLOv11n model. Since there is only one class, and the relevant characteristics of faces is fairly common across the training data, the `nano` version of the model is chosen. 

The `WIDERFACE` dataset was filtered in two stages, 
- The first stage involved filtering the dataset in which the face had little to no presence i.e., a face occupies <1% of the total image. This has been done by keeping the downstream tasks in mind, as we needed to generate 128x128 images, for which at least 90.5x90.5 size faces are needed. (*90.5 is 128/sqrt(2), a size below which the feature extraction is going to be hard for generating 128x128 images*). Similar steps were taken to make sure that the final images were in-line with the needed requirements.
- The faces were once again filtered based on the similarity. Though all faces share the same structure, images of the same face can still be found in the dataset. Hence, a simple cosine-similarity based filtering was used to seperate the images. The embeddings were generated from the `ViT models` in the SigLIP2 Image encoding tower (chosen based on previous experience with the same model). 
- Finally, the model was trained using the Ultralytics library, with the image size of 640, and then converted to `ONNX` format.

## 🚀  Usage

### Basic Usage

Process images in a directory with default NMS settings:

```bash
# From the yolo-rust-inference directory
./target/release/yolo-rust-inference <image_directory>

# Example: Process test images
./target/release/yolo-rust-inference ../test_images
```

### Usage with Custom NMS

Control detection sensitivity with custom NMS thresholds:

```bash
# Default NMS threshold (0.5) - balanced filtering
./target/release/yolo-rust-inference ../test_images

# Aggressive filtering (0.3) - fewer, higher-confidence detections
./target/release/yolo-rust-inference ../test_images 0.3

# Conservative filtering (0.7) - more detections, some overlaps
./target/release/yolo-rust-inference ../test_images 0.7
```

### Output

- Results are automatically saved to `../inference_results/` directory
- Each image gets a corresponding `.txt` file with detection results
- Console shows real-time processing progress and summary


## 📋 Output Format

The application creates a `.txt` file for each processed image with clean, NMS-filtered results:

```
# YOLO Detection Results
# Format: class_id confidence x y width height class_name
# Coordinates are in pixels

 0 0.789833   508.37   132.20    94.03   168.93 face
 0 0.868539   926.44   308.05    55.25    58.11 face
 0 0.744021   274.92   177.24    63.11    78.13 face
```

**Field descriptions:**
- `class_id`: Always 0 (face detection)
- `confidence`: Detection confidence (0.5-1.0)
- `x, y`: Top-left corner coordinates in pixels
- `width, height`: Bounding box dimensions in pixels
- `class_name`: Always "face"


