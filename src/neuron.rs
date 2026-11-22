
use crate::engine::Value;


#[derive(Debug)]
pub struct Neuron {
    w: Vec<Value>,
    b:Value,
}

impl Neuron {
    pub fn new(n_inputs: usize)->Neuron{
        let mut w=Vec::new();
        for _ in 0..n_inputs {
            w.push(Value::new(rand::random_range(-1.0..1.0)));
        }

        let b=Value::new(rand::random_range(-1.0..1.0));
        Neuron { 
            w: w,
            b: b,
        }
    }
    pub fn forward(&self, x: &[Value]) -> Value{
        assert_eq!(x.len(), self.w.len());
        let mut sum=Value::new(0.0);
        for (input,weight) in x.iter().zip(self.w.iter()){
            sum+= weight.clone() * input.clone();
        }
        sum+=self.b.clone();
        return sum.tanh()
    }
    pub fn parameters(&self)->Vec<Value>{
        let mut param=self.w.clone();
        param.push(self.b.clone());
        return param;
    }
}