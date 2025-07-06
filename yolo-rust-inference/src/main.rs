// This is the entry point of the application for the YOLOv11 inference pipeline.

use std::env;
use std::fs;
use std::path::PathBuf;

mod inference;
mod utils;

fn main() {
    // Get the image directory from command-line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <image_directory>", args[0]);
        std::process::exit(1);
    }

    let image_dir = &args[1];
    let image_path = PathBuf::from(image_dir);

    // Check if the provided path is a directory
    if !image_path.is_dir() {
        eprintln!("Error: {} is not a valid directory.", image_dir);
        std::process::exit(1);
    }

    // Initialize the inference pipeline
    let mut pipeline = inference::InferencePipeline::new();
    pipeline.load_model("models/yolov11.onnx").expect("Failed to load model");

    // Read images from the directory
    let images = utils::read_images_from_dir(&image_path).expect("Failed to read images");

    // Run inference and save predictions
    for image in images {
        let predictions = pipeline.run_inference(&image).expect("Inference failed");
        let output_file = format!("{}.txt", image.file_stem().unwrap().to_str().unwrap());
        utils::write_predictions_to_file(&output_file, &predictions).expect("Failed to write predictions");
    }
}