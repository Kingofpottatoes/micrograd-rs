mod neuron;
use neuron::Neuron;
mod engine;
use engine::Value;













fn main() {
    let n=Neuron::new(2);
    let out=n.forward(&[Value::new(1.0),Value::new(-2.0)]);
    out.backward();
    println!("out={}",out.data());
    for p in n.parameters() {
        println!("data= {:.3},grad = {:.3}",p.data(),p.grad());
    }
}