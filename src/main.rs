mod neuron;
use neuron::{MLP,Layer, Neuron,Activation};
mod engine;
use engine::Value;
const EPS: f64 = 1e-7;

fn main() {
    let inputs:Vec<Vec<f64>>=vec![vec![0.0,0.0],vec![1.0,0.0],vec![0.0,1.0],vec![1.0,1.0]];
    let target=vec![0.0,1.0,1.0,0.0];
    let mlp=MLP::new(2, &[4,4,1],Activation::Tanh,Activation::Sigmoid);
    let params=mlp.parameters();
    for step in 0..1000{
        let lr = 0.5 * (1.0 - step as f64/ 1000.0) ;
        let mut loss = Value::new(0.0);
        for (i, t) in inputs.iter().zip(target.iter()) {
            let mut value=Vec::new();
            for n in i{
                value.push(Value::new(*n));
            }
            let pred=mlp.forward(&value)[0].clone();
            loss-= (Value::new(*t)*(pred.clone()+Value::new(EPS)).log()+(Value::new(1.0)-Value::new(*t))*((Value::new(1.0)-pred)+Value::new(EPS)).log())
        }
        for p in params.iter(){
            p.set_grad(0.0);
        }
        loss.backward();
        for p in params.iter(){
            p.set_data(p.data() - lr*p.grad());
        }
        if step%100==0{
            println!("{step} : {}",loss.data())
        }
    }
    for x in inputs.iter(){
        let mut value=Vec::new();
        for n in x{
                value.push(Value::new(*n));
            }
        println!("{:?} → {}",x,mlp.forward(&value)[0].data());
    }
}
