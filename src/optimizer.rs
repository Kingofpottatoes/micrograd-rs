use crate::{autograd::Var, tensor::Tensor};

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
            let delta=p.grad().scale(self.lr);
            p.apply_delta(&delta);
        }
    }
}

pub struct Adam {
    param_list: Vec<Var>,
    m: Vec<Tensor>,
    v: Vec<Tensor>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    t: usize,
}

impl Adam{
    pub fn new(param_list: Vec<Var>,lr: f32)->Adam{
        let mut m=Vec::new();
        let mut v=Vec::new();
        for p in param_list.iter(){
            m.push(Tensor::fill(p.data().rows(), p.data().cols(), 0.0));
            v.push(Tensor::fill(p.data().rows(), p.data().cols(), 0.0));
        }
        Adam {
            param_list,
            m,
            v,
            lr,
            beta1:0.9,
            beta2: 0.999,
            eps: 1e-8,
            t: 0
        }
    }
    pub fn lr(mut self,lr:f32)->Adam{
        self.lr=lr;
        self
    }
    pub fn beta1(mut self,beta1:f32)->Adam{
        self.beta1=beta1;
        self
    }
    pub fn beta2(mut self,beta2:f32)->Adam{
        self.beta2=beta2;
        self
    }
    pub fn eps(mut self,eps:f32)->Adam{
        self.eps=eps;
        self
    }
    
}

impl Optimizer for Adam{
    fn zero_grad(&self) {
        for p in self.param_list.iter(){
            p.set_grad(0.0);
        }
    }
    fn step(&mut self){
        self.t+=1;
        for i in 0..self.param_list.len(){
            {
                //auto drop of grad, to avoid apply delta borrow conflict
                let grad=self.param_list[i].grad();
                self.m[i]=(self.m[i].scale(self.beta1)).add(&grad.scale(1.0-self.beta1));
                self.v[i]=(self.v[i].scale(self.beta2)).add(&grad.mul_elem(&grad.scale(1.0-self.beta2)));
            }
            let mu=self.m[i].scale(1.0/(1.0-self.beta1.powi(self.t as i32)));
            let vu=self.v[i].scale(1.0/(1.0-self.beta2.powi(self.t as i32)));
            let delta=mu.div_elem(&vu.map(|x| x.sqrt() +self.eps)).scale(self.lr);
            self.param_list[i].apply_delta(&delta);
        }
    }
}