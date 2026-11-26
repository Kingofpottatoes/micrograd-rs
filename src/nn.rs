use crate::{autograd::Var, tensor::Tensor};

pub trait Module {
    fn forward(&self,x:&Var)->Var;
    fn parameters(&self)->Vec<Var>;
}

pub struct Linear {
    w:Var,
    b:Var,
}

impl Linear {
    pub fn new(n_inputs: usize, n_outputs: usize)->Linear{
        let w =Var::new(Tensor::rand(n_inputs, n_outputs));
        let b =Var::new(Tensor::rand(1, n_outputs));
        Linear { w, b }
    }
}

impl Module for Linear {
    fn forward(&self,x:&Var)->Var{
        x.matmul(&self.w).add_row(&self.b)
    }
    fn parameters(&self)->Vec<Var>{
        vec![self.w.clone(),self.b.clone()]
    }
}