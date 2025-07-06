// This file contains utility functions for file handling, such as reading images from the specified directory and writing prediction results to text files.

use std::fs;
use std::io::{self, Write};
use std::path::Path;

pub fn read_images_from_dir(dir: &str) -> io::Result<Vec<String>> {
    let mut images = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && is_image(&path) {
            images.push(path.to_string_lossy().into_owned());
        }
    }
    Ok(images)
}

fn is_image(path: &Path) -> bool {
    match path.extension().and_then(|s| s.to_str()) {
        Some("jpg") | Some("jpeg") | Some("png") | Some("bmp") | Some("gif") => true,
        _ => false,
    }
}

pub fn write_predictions_to_file(image_name: &str, predictions: &str) -> io::Result<()> {
    let file_name = format!("{}.txt", image_name);
    let mut file = fs::File::create(file_name)?;
    file.write_all(predictions.as_bytes())?;
    Ok(())
}