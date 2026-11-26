use micrograd_rs::autograd::Var;
use micrograd_rs::tensor::Tensor;
use micrograd_rs::loss::mse_loss;

fn main() {
    let n_step = 3000;
    let lr = 2.0;
    //training set
    let xor_vec = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let target_vec = vec![0.0, 1.0, 1.0, 0.0];
    let xor = Var::new(Tensor::from_vec(xor_vec, 4, 2));
    let target = Var::new(Tensor::from_vec(target_vec, 4, 1));
    //2 layers
    let w1 = Var::new(Tensor::rand(2, 4));
    let b1 = Var::new(Tensor::rand(1, 4));
    let w2 = Var::new(Tensor::rand(4, 1));
    let b2 = Var::new(Tensor::rand(1, 1));

    //training loop
    for i in 0..n_step {
        let hidden = ((xor.matmul(&w1)).add_row(&b1)).tanh();
        let output = ((hidden.matmul(&w2)).add_row(&b2)).sigmoid();
        let loss = mse_loss(&output, &target);
        //reset grad
        w1.set_grad(0.0);
        b1.set_grad(0.0);
        w2.set_grad(0.0);
        b2.set_grad(0.0);
        //backward
        loss.backward();
        w1.update(lr);
        b1.update(lr);
        w2.update(lr);
        b2.update(lr);
        if i % 300 == 0 {
            println!("{i}: loss:{}", loss.data().get(0, 0));
        }
    }
    let hidden = ((xor.matmul(&w1)).add_row(&b1)).tanh();
    let output = ((hidden.matmul(&w2)).add_row(&b2)).sigmoid();
    let loss = mse_loss(&output, &target);
    println!("loss:{}", loss.data().get(0, 0));
    for i in 0..4 {
        println!("{}", output.data().get(i, 0))
    }
}
