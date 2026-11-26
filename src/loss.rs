use crate::autograd::Var;

pub fn mse_loss(pred: &Var, target: &Var) -> Var {
    let diff = pred.sub(target);
    let squared_diff = diff.mul_elem(&diff);

    squared_diff.mean()
}
