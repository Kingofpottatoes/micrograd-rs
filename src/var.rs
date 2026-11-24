use crate::tensor::Tensor;
use std::cell::{Ref, RefCell};
use std::collections::HashSet;
use std::rc::Rc;

#[derive(Clone, Copy)]
pub enum TOp {
    Leaf,
    Sum,
    Sub,
    MulElem,
    MatMul,
    AddRow,
    Tanh,
    Sigmoid,
}

pub struct VarData {
    data: Tensor,
    grad: Tensor,
    children: Vec<Var>,
    op: TOp,
}

#[derive(Clone)]
pub struct Var(Rc<RefCell<VarData>>);

impl Var {
    pub fn data(&self) -> Ref<'_, Tensor> {
        return Ref::map(self.0.borrow(), |d| &d.data);
    }
    pub fn grad(&self) -> Ref<'_, Tensor> {
        return Ref::map(self.0.borrow(), |d| &d.grad);
    }
    pub fn from_op(data: Tensor, children: Vec<Var>, op: TOp) -> Var {
        let grad = Tensor::fill(data.rows(), data.cols(), 0.0);
        Var(Rc::new(RefCell::new(VarData {
            data: data,
            grad: grad,
            children: children,
            op: op,
        })))
    }
    pub fn new(data: Tensor) -> Var {
        return Var::from_op(data, vec![], TOp::Leaf);
    }
    pub fn matmul(&self, other: &Var) -> Var {
        return Var::from_op(
            self.data().matmul(&other.data()),
            vec![self.clone(), other.clone()],
            TOp::MatMul,
        );
    }
    pub fn tanh(&self) -> Var {
        return Var::from_op(self.data().map(f32::tanh), vec![self.clone()], TOp::Tanh);
    }
    pub fn sum(&self) -> Var {
        let data = Tensor::fill(1, 1, self.data().sum());
        return Var::from_op(data, vec![self.clone()], TOp::Sum);
    }
    pub fn sigmoid(&self) -> Var {
        return Var::from_op(
            self.data().map(|x| 1.0 / (1.0 + ((-x).exp()))),
            vec![self.clone()],
            TOp::Sigmoid,
        );
    }
    pub fn sub(&self,other:&Var)->Var{
        return Var::from_op(self.data().add(&other.data().scale(-1.0)), vec![self.clone(),other.clone()], TOp::Sub);

    }
    pub fn add_row(&self, bias: &Var) -> Var {
        let data = self.data().add_row(&bias.data());
        return Var::from_op(data, vec![self.clone(), bias.clone()], TOp::AddRow);
    }
    pub fn mul_elem(&self, other: &Var) -> Var{
        let data=self.data().mul_elem(&other.data());
        return Var::from_op(data, vec![self.clone(), other.clone()], TOp::MulElem);
    }
    fn accumulate_grad(&self, g: &Tensor) {
        let mut d = self.0.borrow_mut();
        d.grad = d.grad.add(g);
    }
    fn local_backward(&self) {
        let op = self.0.borrow().op;
        let data = self.data().clone();
        let grad = self.grad().clone();
        let children = self.0.borrow().children.clone();
        match op {
            TOp::Leaf => {}
            TOp::Sum => {
                let contribution = Tensor::fill(
                    children[0].data().rows(),
                    children[0].data().cols(),
                    grad.get(0, 0),
                );
                children[0].accumulate_grad(&contribution);
            }
            TOp::Sub => {
                children[0].accumulate_grad(&grad);
                children[1].accumulate_grad(&grad.scale(-1.0));
            }
            TOp::MatMul => {
                let a = children[0].data().clone();
                let b = children[1].data().clone();
                let da = grad.matmul(&b.transpose());
                let db = a.transpose().matmul(&grad);
                children[0].accumulate_grad(&da);
                children[1].accumulate_grad(&db);
            }
            TOp::MulElem => {
                let a = children[0].data().clone();
                let b = children[1].data().clone();
                children[0].accumulate_grad(&grad.mul_elem(&b));
                children[1].accumulate_grad(&grad.mul_elem(&a));
            }
            TOp::AddRow => {
                children[0].accumulate_grad(&grad);
                children[1].accumulate_grad(&grad.sum_rows());
            }
            TOp::Tanh => {
                let contribution = data.map(|y| 1.0 - y * y).mul_elem(&grad);
                children[0].accumulate_grad(&contribution);
            }
            TOp::Sigmoid => {
                let contribution = grad.mul_elem(&data.map(|y| y * (1.0 - y)));
                children[0].accumulate_grad(&contribution);
            }
        }
    }
    pub fn backward(&self) {
        assert_eq!(self.data().rows(), 1);
        assert_eq!(self.data().cols(), 1);
        let mut visited = HashSet::new();
        let mut topo = Vec::new();
        build_topo(self, &mut visited, &mut topo);
        self.0.borrow_mut().grad = Tensor::fill(1, 1, 1.0);
        for var in topo.iter().rev() {
            var.local_backward();
        }
    }
    pub fn set_grad(&self, value: f32) {
        let mut g = self.0.borrow_mut();
        g.grad = Tensor::fill(g.data.rows(), g.data.cols(), value);
    }
    pub fn update(&self, lr: f32) {
        let mut d = self.0.borrow_mut();
        d.data = d.data.add(&d.grad.scale(-lr));
    }
}

fn build_topo(node: &Var, visited: &mut HashSet<*const RefCell<VarData>>, topo: &mut Vec<Var>) {
    let id = Rc::as_ptr(&node.0);
    if !visited.insert(id) {
        return;
    }
    let children = node.0.borrow().children.clone();
    for child in children {
        build_topo(&child, visited, topo);
    }
    topo.push(node.clone());
}
