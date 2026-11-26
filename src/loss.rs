use crate::{autograd::Var, tensor::Tensor};

pub fn mse_loss(pred: &Var, target: &Var) -> Var {
    let diff = pred.sub(target);
    let squared_diff = diff.mul_elem(&diff);

    squared_diff.mean()
}

pub fn bce_loss(pred: &Var, target: &Var) -> Var {
    let ones = Var::new(Tensor::fill(
        target.data().rows(),
        target.data().cols(),
        1.0,
    ));
    let eps = Var::new(Tensor::fill(
        target.data().rows(),
        target.data().cols(),
        1e-7,
    ));
    let one_minus_pred = ones.sub(pred);
    let one_minus_target = ones.sub(target);
    let term1 = target.mul_elem(&pred.add(&eps).log());
    let term2 = one_minus_target.mul_elem(&one_minus_pred.add(&eps).log());
    let loss = (term1.add(&term2)).mean().scale(-1.0);
    loss
}
