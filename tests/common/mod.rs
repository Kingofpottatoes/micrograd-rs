#![allow(dead_code)]

use micrograd_rs::autograd::Var;
use micrograd_rs::tensor::Tensor;

pub fn approx_eq(a: &Tensor, b: &Tensor) -> bool {
    let tolerance = 1e-4;
    if a.rows() != b.rows() {
        return false;
    }
    if a.cols() != b.cols() {
        return false;
    }

    a.data()
        .iter()
        .zip(b.data().iter())
        .all(|(a, b)| (*a - *b).abs() <= tolerance)
}

pub fn get_biggest_difference(a: Tensor, b: Tensor) -> f32 {
    assert_eq!(a.rows(), b.rows());
    assert_eq!(a.cols(), b.cols());
    let mut max = (a.get(0, 0) - b.get(0, 0)).abs();
    for i in 0..a.rows() {
        for j in 0..a.cols() {
            let difference = (a.get(i, j) - b.get(i, j)).abs();
            if difference.is_nan() {
                return f32::INFINITY;
            }
            if difference >= max {
                max = difference;
            }
        }
    }
    max
}
pub fn check_grad(f: impl Fn(&Var) -> Var, x: &Tensor) -> f32 {
    let eps = 1e-2;
    let v = Var::new(x.clone());
    f(&v).backward();
    let analytic = v.grad().clone();
    let mut calculed_grad = Tensor::fill(v.data().rows(), v.data().cols(), 0.0);
    for i in 0..x.rows() {
        for j in 0..x.cols() {
            let mut x_plus = x.clone();
            x_plus.set(i, j, x.get(i, j) + eps);
            let mut x_minus = x.clone();
            x_minus.set(i, j, x.get(i, j) - eps);
            let loss_plus = f(&Var::new(x_plus)).data().get(0, 0);
            let loss_minus = f(&Var::new(x_minus)).data().get(0, 0);
            calculed_grad.set(i, j, (loss_plus - loss_minus) / (2.0 * eps));
        }
    }
    get_biggest_difference(analytic, calculed_grad)
}
