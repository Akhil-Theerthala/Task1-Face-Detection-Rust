// This file contains the InferencePipeline struct, which is responsible for loading the YOLOv11 ONNX model, processing images, and generating predictions.

use onnxruntime::{environment::Environment, session::Session, tensor::Tensor};
use std::fs;
use std::path::{Path, PathBuf};

pub struct InferencePipeline {
    model_path: PathBuf,
    session: Session,
}

impl InferencePipeline {
    pub fn new(model_path: &str) -> Self {
        let environment = Environment::builder()
            .with_name("yolo_inference")
            .build()
            .unwrap();
        
        let session = environment
            .new_session_builder()
            .with_model_from_file(model_path)
            .unwrap();

        InferencePipeline {
            model_path: PathBuf::from(model_path),
            session,
        }
    }

    pub fn run_inference(&self, image: &Tensor<f32>) -> Vec<f32> {
        let inputs = vec![image];
        let outputs = self.session.run(inputs).unwrap();
        outputs[0].to_vec()
    }

    pub fn save_predictions(&self, image_name: &str, predictions: &[f32]) {
        let output_file = format!("{}.txt", image_name);
        let predictions_str = predictions.iter()
            .map(|p| p.to_string())
            .collect::<Vec<String>>()
            .join("\n");
        
        fs::write(output_file, predictions_str).expect("Unable to write predictions to file");
    }
}