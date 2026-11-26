mod common;
use common::check_grad;
use micrograd_rs::autograd::Var;
use micrograd_rs::loss::{bce_loss, mse_loss};
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

#[test]
fn bce_loss_half_prediction_is_ln2() {
    // -log(0.5) = ln(2)
    let pred = Var::new(Tensor::from_vec(vec![0.5], 1, 1));
    let target = Var::new(Tensor::from_vec(vec![1.0], 1, 1));
    let loss = bce_loss(&pred, &target).data().get(0, 0);
    assert!(
        (loss - std::f32::consts::LN_2).abs() < 1e-5,
        "loss: {}",
        loss
    );
}

#[test]
fn bce_loss_perfect_prediction_is_zero() {
    let pred = Var::new(Tensor::from_vec(vec![1.0, 0.0], 1, 2));
    let target = Var::new(Tensor::from_vec(vec![1.0, 0.0], 1, 2));
    let loss = bce_loss(&pred, &target).data().get(0, 0);
    assert!(loss.abs() < 1e-5, "loss: {}", loss);
}

#[test]
fn bce_loss_units_test() {
    // -(ln 0.9 + ln 0.8) / 2
    let pred = Var::new(Tensor::from_vec(vec![0.9, 0.2], 1, 2));
    let target = Var::new(Tensor::from_vec(vec![1.0, 0.0], 1, 2));
    let loss = bce_loss(&pred, &target).data().get(0, 0);
    assert!((loss - 0.164_252).abs() < 1e-5, "loss: {}", loss);

    // 2x2: -(ln 0.3 + ln 0.6 + ln 0.25 + ln 0.9) / 4
    let pred = Var::new(Tensor::from_vec(vec![0.3, 0.4, 0.25, 0.9], 2, 2));
    let target = Var::new(Tensor::from_vec(vec![1.0, 0.0, 1.0, 1.0], 2, 2));
    let loss = bce_loss(&pred, &target).data().get(0, 0);
    assert!((loss - 0.801_613).abs() < 1e-5, "loss: {}", loss);
}
#[test]
fn bce_test_size() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let pred = Var::new(Tensor::rand(rows, cols));
    let target = Var::new(Tensor::rand(rows, cols));
    let loss = bce_loss(&pred, &target);
    assert_eq!(loss.data().rows(), 1);
    assert_eq!(loss.data().cols(), 1);
}
#[test]
fn bce_loss_grad() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols);
    let target = Var::new(Tensor::rand(rows, cols));
    let error = check_grad(|p| bce_loss(&p.sigmoid(), &target), &t);
    assert!(
        error < 1e-2,
        "max error: {} (rows={}, cols={})",
        error,
        rows,
        cols
    );
}
