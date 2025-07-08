// This file contains utility functions for file handling, such as reading images from the specified directory and writing prediction results to text files.

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use crate::inference::Detection;

pub fn read_images_from_dir(dir: &Path) -> io::Result<Vec<PathBuf>> {
    let mut images = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && is_image(&path) {
            images.push(path);
        }
    }
    images.sort(); // Sort for consistent processing order
    Ok(images)
}

fn is_image(path: &Path) -> bool {
    match path.extension().and_then(|s| s.to_str()) {
        Some("jpg") | Some("jpeg") | Some("png") | Some("bmp") | Some("gif") | Some("JPG") | Some("JPEG") | Some("PNG") => true,
        _ => false,
    }
}

pub fn write_predictions_to_file(output_path: &Path, predictions: &[Detection]) -> io::Result<()> {
    let mut file = fs::File::create(output_path)?;

    if predictions.is_empty() {
        writeln!(file, "# No detections found")?;
    } else {
        for (_i, detection) in predictions.iter().enumerate() {
            writeln!(
                file,
                "{:2} {:.6} {:8.2} {:8.2} {:8.2} {:8.2}",
                detection.class_id,
                detection.confidence,
                detection.x,
                detection.y,
                detection.width,
                detection.height
            )?;
        }
    }
    
    Ok(())
}

pub fn print_detection_summary(detections: &[Detection]) {
    if detections.is_empty() {
        println!("  No detections found");
        return;
    }
    
    println!("  Found {} detection(s):", detections.len());
    for (i, detection) in detections.iter().enumerate() {
        println!(
            "    {}: {} (confidence: {:.2}, bbox: {:.0},{:.0} {}x{})",
            i + 1,
            detection.get_class_name(),
            detection.confidence,
            detection.x,
            detection.y,
            detection.width as u32,
            detection.height as u32
        );
    }
}