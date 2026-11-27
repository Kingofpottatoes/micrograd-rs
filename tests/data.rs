use micrograd_rs::{data::DataLoader, tensor::Tensor};

#[test]
fn shuffled_indices_does_not_loose_value() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let inputs: Tensor = Tensor::rand(rows, cols);
    let labels = (0..rows).collect::<Vec<usize>>();
    let n_classes = rand::random_range(1..10);
    let batch_size = rand::random_range(1..10);
    let dataloader = DataLoader::new(inputs, labels, n_classes, batch_size);
    let mut indices = dataloader.shuffled_indices();
    indices.sort();
    assert_eq!(indices, (0..rows).collect::<Vec<usize>>());
}

#[test]
fn batch_size_check() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let inputs: Tensor = Tensor::rand(rows, cols);
    let labels = (0..rows).collect::<Vec<usize>>();
    let n_classes = rows;
    let batch_size = rand::random_range(1..10);
    let dataloader = DataLoader::new(inputs, labels, n_classes, batch_size);
    let indices = dataloader.shuffled_indices();
    let (x, y) = dataloader.batch(&indices);
    assert_eq!(x.rows(), indices.len());
    assert_eq!(y.rows(), indices.len());
    assert_eq!(x.cols(), cols);
    assert_eq!(y.cols(), n_classes);
}

#[test]
fn batch_keep_labels_and_corresponding_data_at_the_same_indices() {
    let rows = rand::random_range(1..10);
    let input_data = (0..rows).map(|i| i as f32).collect();
    let inputs: Tensor = Tensor::from_vec(input_data, rows, 1);
    let labels = (0..rows).collect::<Vec<usize>>();
    let n_classes = rows;
    let batch_size = rand::random_range(1..10);
    let dataloader = DataLoader::new(inputs, labels, n_classes, batch_size);
    let indices = dataloader.shuffled_indices();
    let (x, y) = dataloader.batch(&indices);
    let max_row = y.argmax_rows();
    for i in 0..indices.len() {
        assert_eq!(x.get(i, 0), max_row[i] as f32)
    }
}
