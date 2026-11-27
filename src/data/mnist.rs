use crate::tensor::Tensor;

pub fn load_images(path: &str) -> Tensor {
    let bytes = std::fs::read(path).expect("failed to read MNIST images file");
    let magic_number = u32::from_be_bytes(bytes[0..4].try_into().unwrap());
    assert_eq!(magic_number, 2051);
    let n_images = u32::from_be_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let n_rows = u32::from_be_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let n_cols = u32::from_be_bytes(bytes[12..16].try_into().unwrap()) as usize;
    let mut pixels = Vec::new();
    for &p in bytes[16..].iter() {
        pixels.push(p as f32 / 255.0);
    }
    assert_eq!(pixels.len(), n_images * n_rows * n_cols);
    Tensor::from_vec(pixels, n_images, n_rows * n_cols)
}

pub fn load_labels(path: &str) -> Vec<usize> {
    let bytes = std::fs::read(path).expect("failed to read MNIST labels file");
    let magic_number = u32::from_be_bytes(bytes[0..4].try_into().unwrap());
    assert_eq!(magic_number, 2049);
    let n_labels = u32::from_be_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let mut labels = Vec::new();
    for &p in bytes[8..].iter() {
        labels.push(p as usize);
    }
    assert_eq!(labels.len(), n_labels);
    labels
}
