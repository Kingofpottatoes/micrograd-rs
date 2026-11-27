use micrograd_rs::loss::mse_loss;
use micrograd_rs::optimizer::{Adam, Optimizer, Sgd};
use micrograd_rs::{
    autograd::Var,
    nn::{Linear, Module, Sequential, Sigmoid, Tanh},
    tensor::Tensor,
};

#[test]
fn sgd_zero_grad_check() {
    let n_param = rand::random_range(1..10);
    let mut params = Vec::new();
    for _ in 0..n_param {
        let rows = rand::random_range(1..10);
        let cols = rand::random_range(1..10);
        let var = Var::new(Tensor::rand(rows, cols));
        var.set_grad(5.0);
        params.push(var);
    }
    let params_list = params.clone();
    let sgd = Sgd::new(params, 0.0);
    sgd.zero_grad();
    for i in 0..n_param {
        assert_eq!(
            *params_list[i].grad(),
            Tensor::fill(
                params_list[i].data().rows(),
                params_list[i].data().cols(),
                0.0
            )
        );
    }
}

#[test]
fn adam_zero_grad_check() {
    let n_param = rand::random_range(1..10);
    let mut params = Vec::new();
    for _ in 0..n_param {
        let rows = rand::random_range(1..10);
        let cols = rand::random_range(1..10);
        let var = Var::new(Tensor::rand(rows, cols));
        var.set_grad(5.0);
        params.push(var);
    }
    let params_list = params.clone();
    let adam = Adam::new(params, 0.0);
    adam.zero_grad();
    for i in 0..n_param {
        assert_eq!(
            *params_list[i].grad(),
            Tensor::fill(
                params_list[i].data().rows(),
                params_list[i].data().cols(),
                0.0
            )
        );
    }
}

#[test]
fn sgd_learns_xor() {
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
    let mut sdg = Sgd::new(params, lr);
    //training loop
    for i in 0..n_step {
        let output = model.forward(&xor);
        let loss = mse_loss(&output, &target);
        //reset grad
        sdg.zero_grad();
        //backward
        loss.backward();
        sdg.step();
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

#[test]
fn adam_learns_xor() {
    let n_step = 3000;
    let lr = 0.05;
    //training set
    let xor_vec = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let target_vec = vec![0.0, 1.0, 1.0, 0.0];
    let xor = Var::new(Tensor::from_vec(xor_vec, 4, 2));
    let target = Var::new(Tensor::from_vec(target_vec, 4, 1));
    //2 layers
    let model = Sequential::new(vec![
        Box::new(Linear::new(2, 8)),
        Box::new(Tanh),
        Box::new(Linear::new(8, 1)),
        Box::new(Sigmoid),
    ]);
    let mut adam = Adam::new(model.parameters(), lr);
    //training loop
    for i in 0..n_step {
        let output = model.forward(&xor);
        let loss = mse_loss(&output, &target);
        //reset grad
        adam.zero_grad();
        //backward
        loss.backward();
        adam.step();
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
