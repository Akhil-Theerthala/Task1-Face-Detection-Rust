# YOLO Rust Inference

A high-performance face detection inference pipeline using YOLO models with Rust and Python backends. Features automatic Non-Maximum Suppression (NMS) for clean, duplicate-free results.

## 🚀 Features

- **Advanced Face Detection:**
  - Single-class face detection optimized for award ceremony images
  - Non-Maximum Suppression (NMS) to eliminate duplicate detections
  - Configurable IoU threshold for detection filtering

- **Multiple Inference Backends:**
  - Python backend (using ONNX Runtime)
  - Mock backend (for testing and demonstration)
  - Extensible architecture for future native Rust backends

- **Automatic Backend Selection:**
  - Automatically detects and uses the best available backend
  - Falls back gracefully if dependencies are missing

- **Batch Processing:**
  - Process entire directories of images
  - Supports common image formats (JPG, PNG, BMP, GIF)
  - Results saved to dedicated output directory

- **High-Quality Results:**
  - Dramatic reduction in false positives (90% fewer duplicate detections)
  - Human-readable detection results with confidence scores
  - Pixel-perfect coordinate output

## 📁 Project Structure

```
yolo-rust-inference/
├── src/
│   ├── main.rs          # Entry point with NMS threshold support
│   ├── inference.rs     # Multi-backend inference pipeline
│   ├── utils.rs         # Utility functions for file handling
│   └── models/
│       └── mod.rs       # Model-related structures and types
├── models/
│   └── best.onnx        # Face detection YOLO model
├── yolo_inference.py    # Python backend with NMS integration
├── Cargo.toml           # Rust project configuration
└── README.md            # This documentation
```

## 🛠️ Installation & Setup

### Prerequisites

1. **Rust:** Install from [rust-lang.org](https://www.rust-lang.org/tools/install)
2. **Python 3.10+** with virtual environment configured

### Setup Instructions

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

## 🚀 Usage

### Basic Usage

Process images in a directory with default NMS settings:

```bash
# From the yolo-rust-inference directory
./target/release/yolo-rust-inference <image_directory>

# Example: Process test images
./target/release/yolo-rust-inference ../test_images
```

### Advanced Usage with Custom NMS

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

## 📊 Performance Results

**Before NMS (with duplicates):**
```
Image 1: 9 detections  → After NMS: 1 detection
Image 2: 41 detections → After NMS: 5 detections  
Image 3: 78 detections → After NMS: 7 detections
Total: 253 detections  → After NMS: 25 detections (90% reduction)
```

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

## 🔧 Backend Selection

The application automatically selects the best available backend:

1. **Python Backend:** Used when:
   - `yolo_inference.py` script is present
   - Python environment has required packages installed
   - ONNX model file exists

2. **Mock Backend:** Used when:
   - Python backend is unavailable
   - For testing and demonstration purposes
   - Generates random but realistic detection results

## 📁 Supported Image Formats

- JPEG/JPG
- PNG  
- BMP
- GIF

## 📺 Output Examples

### Console Output

```
YOLO Rust Inference Pipeline
=============================
Initializing inference pipeline...
NMS threshold: 0.5
Using Python inference backend
Model loaded from: "models/best.onnx"
✓ Model loaded successfully
Found 6 images to process

Results will be saved to: "/path/to/inference_results"

Processing 1/6: "16_Award_Ceremony_Awards_Ceremony_16_22.jpg"
  Found 1 detection(s):
    1: face (confidence: 0.79, bbox: 508,132 94x168)
  ✓ Results saved to: "16_Award_Ceremony_Awards_Ceremony_16_22.txt"

=============================
Inference complete!
Processed 6 images with 25 total detections
```

### File Output (example.txt)

```
# YOLO Detection Results  
# Format: class_id confidence x y width height class_name
# Coordinates are in pixels

 0 0.789833   508.37   132.20    94.03   168.93 face
```

## ⚙️ Non-Maximum Suppression (NMS)

NMS eliminates duplicate detections by removing overlapping bounding boxes:

- **IoU Threshold 0.3:** Aggressive filtering (fewer detections)
- **IoU Threshold 0.5:** Balanced filtering (recommended)  
- **IoU Threshold 0.7:** Conservative filtering (more detections)

**Impact:** Reduces detection count from 253 to 25 high-quality results (90% improvement)

## 🔧 Troubleshooting

### Common Issues

**"No images found in directory"**
- Ensure directory contains supported formats (JPG, PNG, BMP, GIF)
- Check file permissions and directory path

**"Python inference failed"**  
- Verify Python environment has required packages
- Ensure `best.onnx` model exists in `models/` directory

**"Model not found"**
- Check that `best.onnx` is in the `models/` directory
- Verify file permissions

**Detection Quality Issues**
- Too many detections: Use lower NMS threshold (e.g., 0.3)
- Too few detections: Use higher NMS threshold (e.g., 0.7)
- Default NMS threshold: 0.5 (balanced)

### Debug Information

Console output provides:
- Backend selection (Python/Mock)
- NMS threshold setting
- Model loading status
- Processing progress and detection counts

## 🔧 Performance Tips

- Build with `--release` for optimal performance
- Adjust NMS threshold based on precision/recall needs
- Process images in batches for best efficiency

## 📄 License

This project is part of the Face-Generation repository.

## 🤝 Contributing

Contributions welcome! The architecture supports:
- Additional ONNX model formats
- New inference backends  
- Enhanced post-processing algorithms