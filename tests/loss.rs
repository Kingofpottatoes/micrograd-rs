mod common;
use common::check_grad;
use micrograd_rs::autograd::Var;
use micrograd_rs::loss::mse_loss;
use micrograd_rs::tensor::Tensor;

#[test]
fn mse_test_size() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let pred = Var::new(Tensor::rand(rows, cols));
    let target = Var::new(Tensor::rand(rows, cols));
    let loss = mse_loss(&pred, &target);
    assert_eq!(loss.data().rows(), 1);
    assert_eq!(loss.data().cols(), 1);
}

#[test]
fn mse_loss_units_test() {
    let pred = Var::new(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2));
    let target = Var::new(Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], 2, 2));
    let loss = mse_loss(&pred, &target);
    assert_eq!(*loss.data(), Tensor::from_vec(vec![3.5], 1, 1));
}

#[test]
fn mse_loss_grad() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols);
    let target = Var::new(Tensor::rand(rows, cols));
    let error = check_grad(|p| mse_loss(p, &target), &t);
    assert!(
        error < 1e-2,
        "max error: {} (rows={}, cols={})",
        error,
        rows,
        cols
    );
}
