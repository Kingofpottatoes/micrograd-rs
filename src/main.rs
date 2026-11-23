mod neuron;
use neuron::{MLP,Layer, Neuron};
mod engine;
use engine::Value;

fn main() {
    let inputs:Vec<Vec<f64>>=vec![vec![0.0,0.0],vec![1.0,0.0],vec![0.0,1.0],vec![1.0,1.0]];
    let target=vec![0.0,1.0,1.0,0.0];
    let mlp=MLP::new(2, &[4,4,1]);
    let params=mlp.parameters();
    let lr=0.025;
    for step in 0..1000{
        let mut loss = Value::new(0.0);
        for (i, t) in inputs.iter().zip(target.iter()) {
            let mut value=Vec::new();
            for n in i{
                value.push(Value::new(*n));
            }
            let pred=mlp.forward(&value)[0].clone();
            let diff=pred-Value::new(*t);
            loss+=diff.clone()*diff
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
    let mut i=0;
    for x in inputs.iter(){
        let mut value=Vec::new();
        for n in x{
                value.push(Value::new(*n));
            }
        println!("{:?} → {}",x,mlp.forward(&value)[0].data());
        i+=1;
    }
}
