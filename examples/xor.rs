use micrograd_rs::autograd::Var;
use micrograd_rs::loss::mse_loss;
use micrograd_rs::nn::{Linear, Module, Sequential, Sigmoid, Tanh};
use micrograd_rs::optimizer::{Optimizer, Sgd};
use micrograd_rs::tensor::Tensor;

fn main() {
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
    let mut sgd = Sgd::new(model.parameters(), lr);
    //training loop
    for i in 0..n_step {
        let output = model.forward(&xor);
        let loss = mse_loss(&output, &target);
        //reset grad
        sgd.zero_grad();
        //backward
        loss.backward();
        sgd.step();
        if i % 300 == 0 {
            println!("{i}: loss:{}", loss.data().get(0, 0));
        }
    }
    let output = model.forward(&xor);
    let loss = mse_loss(&output, &target);
    println!("loss:{}", loss.data().get(0, 0));
    for i in 0..4 {
        println!("{}", output.data().get(i, 0))
    }
}
