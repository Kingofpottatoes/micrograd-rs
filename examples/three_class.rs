use micrograd_rs::autograd::Var;
use micrograd_rs::loss::cross_entropy_loss;
use micrograd_rs::nn::{Linear, Module, Sequential, Tanh};
use micrograd_rs::optimizer::{Adam, Optimizer};
use micrograd_rs::tensor::Tensor;
use micrograd_rs::visualizer::plot_decision_boundary;

fn main() {
    let n_step = 3000;
    let lr = 0.05;
    let n_data_per_class = 30;
    //training set
    let centers = [(-1.0, -1.0), (1.0, -1.0), (0.0, 1.0)];
    let mut input_data_vec = Vec::new();
    let mut target_vec = Vec::new();
    for (c, &(cx, cy)) in centers.iter().enumerate() {
        for _ in 0..n_data_per_class {
            input_data_vec.push(cx + rand::random_range(-0.4..0.4));
            input_data_vec.push(cy + rand::random_range(-0.4..0.4));
            for k in 0..3 {
                if c == k {
                    target_vec.push(1.0)
                } else {
                    target_vec.push(0.0);
                }
            }
        }
    }
    let input = Var::new(Tensor::from_vec(input_data_vec, 3 * n_data_per_class, 2));
    let target = Var::new(Tensor::from_vec(target_vec, 3 * n_data_per_class, 3));
    //2 layers
    let model = Sequential::new(vec![
        Box::new(Linear::new(2, 8)),
        Box::new(Tanh),
        Box::new(Linear::new(8, 3)),
    ]);
    let mut adam = Adam::new(model.parameters(), lr);
    //training loop
    for i in 0..n_step {
        let output = model.forward(&input);
        let loss = cross_entropy_loss(&output, &target);
        //reset grad
        adam.zero_grad();
        //backward
        loss.backward();
        adam.step();
        if i % 300 == 0 {
            println!("{i}: loss:{}", loss.data().get(0, 0));
        }
    }
    let output = model.forward(&input);
    let loss = cross_entropy_loss(&output, &target);
    println!("loss:{}", loss.data().get(0, 0));
    let pred = output.data().argmax_rows();
    let truth = target.data().argmax_rows();
    let corrects = pred
        .iter()
        .zip(truth.iter())
        .filter(|(a, b)| a == b)
        .count();
    let accuracy = corrects as f32 / (3 * n_data_per_class) as f32;
    plot_decision_boundary(
        &model,
        &input.data(),
        &truth,
        "plot_results/boundary_decision_three_class.png",
    )
    .unwrap();
    println!("accuracy: {accuracy}")
}
