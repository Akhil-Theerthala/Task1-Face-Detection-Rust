// This file defines the model-related structures and types for the YOLOv11 inference pipeline. 
// It includes enums or structs that represent the model's input and output formats, 
// as well as any necessary preprocessing steps.

pub struct YoloInput {
    pub image_data: Vec<u8>,
    pub width: usize,
    pub height: usize,
}

pub struct YoloOutput {
    pub boxes: Vec<BoundingBox>,
    pub confidences: Vec<f32>,
    pub class_ids: Vec<u32>,
}

pub struct BoundingBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}