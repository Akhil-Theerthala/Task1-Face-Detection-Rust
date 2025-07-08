// This is the entry point of the application for the YOLOv11 inference pipeline.

use std::env;
use std::path::PathBuf;
use anyhow::Result;

mod inference;
mod utils;
mod models;

fn main() -> Result<()> {
    println!("YOLO Rust Inference Pipeline");
    println!("=============================");
    
    // Get the image directory from command-line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <image_directory> [nms_threshold]", args[0]);
        eprintln!("Example: {} ../test_images 0.5", args[0]);
        eprintln!("  nms_threshold: IoU threshold for Non-Maximum Suppression (default: 0.5)");
        std::process::exit(1);
    }

    let image_dir = &args[1];
    let nms_threshold: f32 = if args.len() > 2 {
        args[2].parse().unwrap_or_else(|_| {
            eprintln!("Error: Invalid NMS threshold '{}'. Using default 0.5", args[2]);
            0.5
        })
    } else {
        0.5
    };
    let image_path = PathBuf::from(image_dir);

    // Check if the provided path is a directory
    if !image_path.is_dir() {
        eprintln!("Error: {} is not a valid directory.", image_dir);
        std::process::exit(1);
    }

    // Initialize the inference pipeline
    println!("Initializing inference pipeline...");
    println!("NMS threshold: {}", nms_threshold);
    let mut pipeline = inference::InferencePipeline::new().with_nms_threshold(nms_threshold);
    
    // Try to load the model
    let model_path = "models/best.onnx";
    match pipeline.load_model(model_path) {
        Ok(_) => println!("✓ Model loaded successfully"),
        Err(e) => {
            println!("⚠ Warning: Could not load model from {}: {}", model_path, e);
            println!("  Continuing with mock inference for demonstration...");
        }
    }

    // Read images from the directory
    let images = utils::read_images_from_dir(&image_path)?;
    
    if images.is_empty() {
        println!("No images found in directory: {}", image_dir);
        return Ok(());
    }
    
    println!("Found {} images to process", images.len());
    println!();

    // Create inference_results directory
    let results_dir = PathBuf::from("../inference_results");
    std::fs::create_dir_all(&results_dir)?;
    println!("Results will be saved to: {:?}", results_dir.canonicalize().unwrap_or(results_dir.clone()));
    println!();

    // Run inference and save predictions
    let mut total_detections = 0;
    for (i, image_path) in images.iter().enumerate() {
        println!("Processing {}/{}: {:?}", i + 1, images.len(), image_path.file_name().unwrap());
        
        match pipeline.run_inference(image_path) {
            Ok(predictions) => {
                total_detections += predictions.len();
                
                // Print summary to console
                utils::print_detection_summary(&predictions);
                
                // Save to inference_results directory
                let image_name = image_path.file_stem().unwrap().to_string_lossy();
                let output_file = results_dir.join(format!("{}.txt", image_name));
                utils::write_predictions_to_file(&output_file, &predictions)?;
                println!("  ✓ Results saved to: {:?}", output_file.file_name().unwrap());
            }
            Err(e) => {
                eprintln!("  ✗ Failed to process {:?}: {}", image_path.file_name().unwrap(), e);
            }
        }
        println!();
    }
    
    println!("=============================");
    println!("Inference complete!");
    println!("Processed {} images with {} total detections", images.len(), total_detections);
    Ok(())
}