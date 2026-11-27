use micrograd_rs::autograd::Var;
use micrograd_rs::visualizer::plot_loss;
use micrograd_rs::{
    data::{
        DataLoader,
        mnist::{load_images, load_labels},
    },
    loss::cross_entropy_loss,
    nn::{Linear, Module, Relu, Sequential},
    optimizer::{Adam, Optimizer},
};

fn main() {
    let start = std::time::Instant::now();
    let train_images = load_images("data/mnist/train-images-idx3-ubyte");
    let train_labels = load_labels("data/mnist/train-labels-idx1-ubyte");
    let test_images = load_images("data/mnist/t10k-images-idx3-ubyte");
    let test_labels = load_labels("data/mnist/t10k-labels-idx1-ubyte");
    let dataloader = DataLoader::new(train_images, train_labels, 10, 64);
    let n_epochs: i32 = 3; //how many time model see all the data
    let mut losses = Vec::new();
    let model = Sequential::new(vec![
        Box::new(Linear::new(784, 128)),
        Box::new(Relu),
        Box::new(Linear::new(128, 10)),
    ]);
    let lr = 0.001;
    let mut adam = Adam::new(model.parameters(), lr);
    for epoch in 0..n_epochs {
        let indices = dataloader.shuffled_indices();
        for (i, chunk) in indices.chunks(dataloader.batch_size()).enumerate() {
            let (x, y) = dataloader.batch(chunk);
            let output = model.forward(&Var::new(x));
            let loss = cross_entropy_loss(&output, &Var::new(y));
            losses.push(loss.data().get(0, 0));
            adam.zero_grad();
            loss.backward();
            adam.step();
            if i % 100 == 0 {
                println!("epoch {epoch}, batch {i}: loss {}", loss.data().get(0, 0));
            }
        }
    }
    println!("{:?}", start.elapsed());
    let output = model.forward(&Var::new(test_images));
    let pred = output.data().argmax_rows();
    let truth = test_labels;
    let corrects = pred
        .iter()
        .zip(truth.iter())
        .filter(|(a, b)| a == b)
        .count();
    let accuracy = corrects as f32 / (truth.len()) as f32;
    println!("accuracy: {accuracy}");
    plot_loss(&losses, "plot_results/losses_mnst.png").unwrap();
}
