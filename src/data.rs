use crate::tensor::Tensor;

pub mod mnist;

pub fn one_hot(labels: &[usize], n_classes: usize) -> Tensor {
    let mut t = Tensor::fill(labels.len(), n_classes, 0.0);
    for i in 0..labels.len() {
        assert!(
            labels[i] < n_classes,
            "label {} out of range (n_classes = {})",
            labels[i],
            n_classes
        );
        t.set(i, labels[i], 1.0);
    }
    t
}
