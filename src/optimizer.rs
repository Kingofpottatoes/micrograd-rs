use crate::autograd::Var;

pub trait Optimizer {
    fn zero_grad(&self);
    fn step(&mut self);
}

pub struct Sgd {
    param_list: Vec<Var>,
    lr: f32,
}
impl Sgd {
    pub fn new(param_list: Vec<Var>, lr: f32) -> Sgd {
        Sgd { param_list, lr }
    }
}
impl Optimizer for Sgd {
    fn zero_grad(&self) {
        for p in self.param_list.iter() {
            p.set_grad(0.0);
        }
    }
    fn step(&mut self) {
        for p in self.param_list.iter() {
            p.update(self.lr);
        }
    }
}
