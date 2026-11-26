mod common;

use common::check_grad;
use micrograd_rs::autograd::Var;
use micrograd_rs::tensor::Tensor;

#[test]
fn tanh_grad() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols);
    let w = Var::new(Tensor::rand(rows, cols));
    let error = check_grad(|v| v.tanh().mul_elem(&w).sum(), &t);
    assert!(
        error < 1e-2,
        "max error: {} (rows={}, cols={})",
        error,
        rows,
        cols
    );
}

#[test]
fn sigmoid_grad() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols);
    let w = Var::new(Tensor::rand(rows, cols));
    let error = check_grad(|v| v.sigmoid().mul_elem(&w).sum(), &t);
    assert!(
        error < 1e-2,
        "max error: {} (rows={}, cols={})",
        error,
        rows,
        cols
    );
}

#[test]
fn sum_grad() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols);
    let error = check_grad(|v| v.sum(), &t);
    assert!(
        error < 1e-2,
        "max error: {} (rows={}, cols={})",
        error,
        rows,
        cols
    );
}

#[test]
fn matmul_first_entry_grad() {
    let rows = rand::random_range(1..10);
    let common_size = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let a = Tensor::rand(rows, common_size);
    let b = Var::new(Tensor::rand(common_size, cols));
    let w = Var::new(Tensor::rand(rows, cols));
    let error = check_grad(|v| v.matmul(&b).mul_elem(&w).sum(), &a);
    assert!(
        error < 1e-2,
        "max error: {} (rows={}, cols={}, common_size={})",
        error,
        rows,
        cols,
        common_size
    );
}

#[test]
fn matmul_second_entry_grad() {
    let rows = rand::random_range(1..10);
    let common_size = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let b = Tensor::rand(common_size, cols);
    let a = Var::new(Tensor::rand(rows, common_size));
    let w = Var::new(Tensor::rand(rows, cols));
    let error = check_grad(|b| a.matmul(b).mul_elem(&w).sum(), &b);
    assert!(
        error < 1e-2,
        "max error: {} (rows={}, cols={}, common_size={})",
        error,
        rows,
        cols,
        common_size
    );
}

#[test]
fn add_row_grad() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let x = Tensor::rand(rows, cols);
    let bias = Var::new(Tensor::rand(1, cols));
    let w = Var::new(Tensor::rand(rows, cols));
    let error = check_grad(|v| v.add_row(&bias).mul_elem(&w).sum(), &x);
    assert!(
        error < 1e-2,
        "max error: {} (rows={}, cols={})",
        error,
        rows,
        cols
    );
}

#[test]
fn add_row_bias_grad() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let x = Tensor::rand(1, cols);
    let x_cap = Var::new(Tensor::rand(rows, cols));
    let w = Var::new(Tensor::rand(rows, cols));
    let error = check_grad(|b| x_cap.add_row(b).mul_elem(&w).sum(), &x);
    assert!(
        error < 1e-2,
        "max error: {} (rows={}, cols={})",
        error,
        rows,
        cols
    );
}

#[test]
fn sub_first_entry_grad() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols);
    let b = Var::new(Tensor::rand(rows, cols));
    let w = Var::new(Tensor::rand(rows, cols));
    let error = check_grad(|a| a.sub(&b).mul_elem(&w).sum(), &t);
    assert!(
        error < 1e-2,
        "max error: {} (rows={}, cols={})",
        error,
        rows,
        cols
    );
}

#[test]
fn sub_second_entry_grad() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols);
    let a = Var::new(Tensor::rand(rows, cols));
    let w = Var::new(Tensor::rand(rows, cols));
    let error = check_grad(|b| a.sub(b).mul_elem(&w).sum(), &t);
    assert!(
        error < 1e-2,
        "max error: {} (rows={}, cols={})",
        error,
        rows,
        cols
    );
}

#[test]
fn add_first_entry_grad() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols);
    let b = Var::new(Tensor::rand(rows, cols));
    let w = Var::new(Tensor::rand(rows, cols));
    let error = check_grad(|a| a.add(&b).mul_elem(&w).sum(), &t);
    assert!(
        error < 1e-2,
        "max error: {} (rows={}, cols={})",
        error,
        rows,
        cols
    );
}

#[test]
fn add_second_entry_grad() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols);
    let a = Var::new(Tensor::rand(rows, cols));
    let w = Var::new(Tensor::rand(rows, cols));
    let error = check_grad(|b| a.add(b).mul_elem(&w).sum(), &t);
    assert!(
        error < 1e-2,
        "max error: {} (rows={}, cols={})",
        error,
        rows,
        cols
    );
}

#[test]
fn scale_grad() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let k: f32 = rand::random_range(-2.0..2.0);
    let t = Tensor::rand(rows, cols);
    let w = Var::new(Tensor::rand(rows, cols));
    let error = check_grad(|v| v.scale(k).mul_elem(&w).sum(), &t);
    assert!(
        error < 1e-2,
        "max error: {} (rows={}, cols={})",
        error,
        rows,
        cols
    );
}

#[test]
fn mean_grad() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols);
    let error = check_grad(|v| v.mean(), &t);
    assert!(
        error < 1e-2,
        "max error: {} (rows={}, cols={})",
        error,
        rows,
        cols
    );
}

#[test]
fn log_grad() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols).map(|x| x.abs() + 0.5); //only positive value to avoid Nan
    let w = Var::new(Tensor::rand(rows, cols));
    let error = check_grad(|v| v.log().mul_elem(&w).sum(), &t);
    assert!(
        error < 1e-2,
        "max error: {} (rows={}, cols={})",
        error,
        rows,
        cols
    );
}

#[test]
fn relu_grad() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols).map(|x| if x.abs() < 0.1 { 0.5 } else { x }); //move velue close to zero to avoid a +eps who switch signs
    let w = Var::new(Tensor::rand(rows, cols));
    let error = check_grad(|v| v.relu().mul_elem(&w).sum(), &t);
    assert!(
        error < 1e-2,
        "max error: {} (rows={}, cols={})",
        error,
        rows,
        cols
    );
}
#[test]
fn mul_elem_first_entry_grad() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols);
    let b = Var::new(Tensor::rand(rows, cols));
    let w = Var::new(Tensor::rand(rows, cols));
    let error = check_grad(|a| a.mul_elem(&b).mul_elem(&w).sum(), &t);
    assert!(
        error < 1e-2,
        "max error: {} (rows={}, cols={})",
        error,
        rows,
        cols
    );
}

#[test]
fn mul_elem_second_entry() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols);
    let a = Var::new(Tensor::rand(rows, cols));
    let w = Var::new(Tensor::rand(rows, cols));
    let error = check_grad(|b| a.mul_elem(b).mul_elem(&w).sum(), &t);
    assert!(
        error < 1e-2,
        "max error: {} (rows={}, cols={})",
        error,
        rows,
        cols
    );
}

#[test]
fn is_gradient_cumultative() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols);
    let error = check_grad(|v| v.mul_elem(v).sum(), &t);
    assert!(
        error < 1e-2,
        "max error: {} (rows={}, cols={})",
        error,
        rows,
        cols
    );
}

#[test]
fn one_cycle() {
    let n = rand::random_range(1..6); //number of example
    let k = rand::random_range(1..6); //number of inputs
    let h = rand::random_range(1..6); //number of hiden neurons
    let o = rand::random_range(1..6); //number of outputs

    let x = Var::new(Tensor::rand(n, k)); //all exemple inputs
    let b1 = Var::new(Tensor::rand(1, h)); //hidden neurons bias
    let w2 = Var::new(Tensor::rand(h, o)); //weight of output layer
    let b2 = Var::new(Tensor::rand(1, o)); //bias of output layer
    let target = Var::new(Tensor::rand(n, o)); //attended output

    let w1_values = Tensor::rand(k, h);

    let error = check_grad(
        |w1| {
            let hidden = x.matmul(w1).add_row(&b1).tanh();
            let out = hidden.matmul(&w2).add_row(&b2).sigmoid();
            let diff = out.sub(&target);
            diff.mul_elem(&diff).sum() // returned value: the loss
        },
        &w1_values,
    );
    assert!(
        error < 1e-2,
        "max error: {} (n={}, k={}, h={}, o={})",
        error,
        n,
        k,
        h,
        o
    );
}
