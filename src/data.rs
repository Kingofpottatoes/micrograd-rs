use crate::tensor::Tensor;
use rand::seq::SliceRandom;
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

pub struct DataLoader {
    inputs: Tensor,
    labels: Vec<usize>,
    n_classes: usize,
    batch_size: usize,
}

impl DataLoader {
    pub fn new(
        inputs: Tensor,
        labels: Vec<usize>,
        n_classes: usize,
        batch_size: usize,
    ) -> DataLoader {
        assert_eq!(
            inputs.rows(),
            labels.len(),
            "inputs and labels must have the same number of examples"
        );
        DataLoader {
            inputs,
            labels,
            n_classes,
            batch_size,
        }
    }
    pub fn shuffled_indices(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.inputs.rows()).collect();
        indices.shuffle(&mut rand::rng());

        indices
    }
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn batch(&self, indices: &[usize]) -> (Tensor, Tensor) {
        let x = self.inputs.select_rows(indices);
        let mut batch_labels = Vec::new();
        for &i in indices {
            batch_labels.push(self.labels[i]);
        }
        let y = one_hot(&batch_labels, self.n_classes);

        (x, y)
    }
}
