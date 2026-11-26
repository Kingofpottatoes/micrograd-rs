use crate::{autograd::Var, tensor::Tensor};

pub trait Module {
    fn forward(&self, x: &Var) -> Var;
    fn parameters(&self) -> Vec<Var>;
}

pub struct Linear {
    w: Var,
    b: Var,
}

impl Linear {
    pub fn new(n_inputs: usize, n_outputs: usize) -> Linear {
        let w = Var::new(Tensor::rand(n_inputs, n_outputs));
        let b = Var::new(Tensor::rand(1, n_outputs));
        Linear { w, b }
    }
}

impl Module for Linear {
    fn forward(&self, x: &Var) -> Var {
        x.matmul(&self.w).add_row(&self.b)
    }
    fn parameters(&self) -> Vec<Var> {
        vec![self.w.clone(), self.b.clone()]
    }
}

pub struct Tanh;
impl Module for Tanh {
    fn forward(&self, x: &Var) -> Var {
        x.tanh()
    }
    fn parameters(&self) -> Vec<Var> {
        vec![]
    }
}
pub struct Sigmoid;
impl Module for Sigmoid {
    fn forward(&self, x: &Var) -> Var {
        x.sigmoid()
    }
    fn parameters(&self) -> Vec<Var> {
        vec![]
    }
}
pub struct Relu;
impl Module for Relu {
    fn forward(&self, x: &Var) -> Var {
        x.relu()
    }
    fn parameters(&self) -> Vec<Var> {
        vec![]
    }
}

pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new(layers: Vec<Box<dyn Module>>) -> Sequential {
        Sequential { layers }
    }
}

impl Module for Sequential {
    fn forward(&self, x: &Var) -> Var {
        let mut out = x.clone();
        for layer in self.layers.iter() {
            out = layer.forward(&out);
        }
        out
    }
    fn parameters(&self) -> Vec<Var> {
        let mut params = Vec::new();
        for layer in self.layers.iter() {
            params.extend(layer.parameters());
        }
        params
    }
}
