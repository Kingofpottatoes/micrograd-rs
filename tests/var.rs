
use micrograd_rs::var::Var;
use micrograd_rs::tensor::Tensor;

fn get_biggest_difference(a:Tensor,b:Tensor)->f32{
    assert_eq!(a.rows(),b.rows());
    assert_eq!(a.cols(),b.cols());
    let mut max=(a.get(0,0)-b.get(0,0)).abs();
    for i in 0..a.rows(){
        for j in 0..a.cols(){
            let difference=(a.get(i,j)-b.get(i,j)).abs();
            if difference>=max{
                max=difference;
            }
        }
    }
    return max;
}
fn check_grad(f: impl Fn(&Var) -> Var, x: &Tensor) -> f32{
    let eps=1e-2;
    let v=Var::new(x.clone());
    let _loss=f(&v).backward();
    let analytic=v.grad().clone();
    let mut calculed_grad=Tensor::fill(v.data().rows(), v.data().cols(), 0.0);
    for i in 0..x.rows(){
        for j in 0..x.cols(){
            let mut x_plus=x.clone();
            x_plus.set(i,j,x.get(i,j)+eps);
            let mut x_minus=x.clone();
            x_minus.set(i,j,x.get(i,j)-eps);
            let loss_plus=f(&Var::new(x_plus)).data().get(0,0);
            let loss_minus=f(&Var::new(x_minus)).data().get(0,0);
            calculed_grad.set(i,j,(loss_plus-loss_minus)/(2.0*eps));
        }
    }
    return get_biggest_difference(analytic, calculed_grad);

}

#[test]
fn tanh_grad(){
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols);
    let w=Var::new(Tensor::rand(rows, cols));
    let error=check_grad(|v| v.tanh().mul_elem(&w).sum(), &t);
    assert!(error < 1e-2, "max error: {}", error);
}