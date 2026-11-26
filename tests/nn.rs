use micrograd_rs::loss::mse_loss;
use micrograd_rs::{
    autograd::Var,
    nn::{Linear, Module, Relu, Sequential, Sigmoid, Tanh},
    tensor::Tensor,
};

#[test]
fn linear_output_size_check() {
    let n = rand::random_range(1..10);
    let k = rand::random_range(1..10);
    let m = rand::random_range(1..10);
    let layer = Linear::new(k, m);
    let var = Var::new(Tensor::rand(n, k));
    let out = layer.forward(&var);
    assert_eq!(out.data().rows(), n);
    assert_eq!(out.data().cols(), m);
}
#[test]
fn linear_params_size_check() {
    let k = rand::random_range(1..10);
    let m = rand::random_range(1..10);
    let layer = Linear::new(k, m);
    let params = layer.parameters();
    assert_eq!(params.len(), 2);
    assert_eq!(params[0].data().rows(), k);
    assert_eq!(params[0].data().cols(), m);
    assert_eq!(params[1].data().rows(), 1);
    assert_eq!(params[1].data().cols(), m);
}
#[test]
fn tanh_matches_var_op() {
    let n = rand::random_range(1..10);
    let k = rand::random_range(1..10);
    let layer = Tanh;
    let var = Var::new(Tensor::rand(n, k));
    let out = layer.forward(&var);
    let out2 = var.tanh();
    assert_eq!(*out.data(), *out2.data());
}
#[test]
fn tanh_params_size_check() {
    let layer = Tanh;
    let params = layer.parameters();
    assert!(params.is_empty());
}
#[test]
fn sigmoid_matches_var_op() {
    let n = rand::random_range(1..10);
    let k = rand::random_range(1..10);
    let layer = Sigmoid;
    let var = Var::new(Tensor::rand(n, k));
    let out = layer.forward(&var);
    let out2 = var.sigmoid();
    assert_eq!(*out.data(), *out2.data());
}
#[test]
fn sigmoid_params_size_check() {
    let layer = Sigmoid;
    let params = layer.parameters();
    assert!(params.is_empty());
}
#[test]
fn relu_matches_var_op() {
    let n = rand::random_range(1..10);
    let k = rand::random_range(1..10);
    let layer = Relu;
    let var = Var::new(Tensor::rand(n, k));
    let out = layer.forward(&var);
    let out2 = var.relu();
    assert_eq!(*out.data(), *out2.data());
}
#[test]
fn relu_params_size_check() {
    let layer = Relu;
    let params = layer.parameters();
    assert!(params.is_empty());
}

#[test]
fn sequential_check() {
    let n = rand::random_range(1..10);
    let a = rand::random_range(1..10);
    let b = rand::random_range(1..10);
    let c = rand::random_range(1..10);

    let var = Var::new(Tensor::rand(n, a));
    let model = Sequential::new(vec![
        Box::new(Linear::new(a, b)),
        Box::new(Tanh),
        Box::new(Linear::new(b, c)),
    ]);
    let params = model.parameters();
    let output = model.forward(&var);
    assert_eq!(params.len(), 4);
    assert_eq!(output.data().rows(), n);
    assert_eq!(output.data().cols(), c);
}

#[test]
fn sequential_learns_xor() {
    let n_step = 3000;
    let lr = 2.0;
    //training set
    let xor_vec = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let target_vec = vec![0.0, 1.0, 1.0, 0.0];
    let xor = Var::new(Tensor::from_vec(xor_vec, 4, 2));
    let target = Var::new(Tensor::from_vec(target_vec, 4, 1));
    //2 layers
    let model = Sequential::new(vec![
        Box::new(Linear::new(2, 4)),
        Box::new(Tanh),
        Box::new(Linear::new(4, 1)),
        Box::new(Sigmoid),
    ]);
    let params = model.parameters();
    //training loop
    for i in 0..n_step {
        let output = model.forward(&xor);
        let loss = mse_loss(&output, &target);
        //reset grad
        for p in params.iter() {
            p.set_grad(0.0);
        }
        //backward
        loss.backward();
        for p in params.iter() {
            let delta = p.grad().scale(lr);
            p.apply_delta(&delta);
        }
        if i % 300 == 0 {
            println!("{i}: loss:{}", loss.data().get(0, 0));
        }
    }
    let output = model.forward(&xor);
    let loss = mse_loss(&output, &target);
    let final_loss = loss.data().get(0, 0);
    assert!(final_loss < 0.05, "final loss too high: {}", final_loss);
    for i in 0..4 {
        println!("{}", output.data().get(i, 0))
    }
}
