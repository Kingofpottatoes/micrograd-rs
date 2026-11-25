
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

#[test]
fn sigmoid_grad(){
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols);
    let w=Var::new(Tensor::rand(rows, cols));
    let error=check_grad(|v| v.sigmoid().mul_elem(&w).sum(), &t);
    assert!(error < 1e-2, "max error: {}", error);
}

#[test]
fn sum_grad(){
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols);
    let error=check_grad(|v| v.sum(), &t);
    assert!(error < 1e-2, "max error: {}", error);
}

#[test]
fn matmul_first_entry_grad(){
    let rows = rand::random_range(1..10);
    let common_size = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let a = Tensor::rand(rows, common_size);
    let b=Var::new(Tensor::rand(common_size, cols));
    let w=Var::new(Tensor::rand(rows, cols));
    let error=check_grad(|v| v.matmul(&b).mul_elem(&w).sum(), &a);
    assert!(error < 1e-2, "max error: {}", error);
}

#[test]
fn matmul_second_entry_grad(){
    let rows = rand::random_range(1..10);
    let common_size = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let b = Tensor::rand(common_size, cols);
    let a=Var::new(Tensor::rand(rows, common_size));
    let w=Var::new(Tensor::rand(rows, cols));
    let error=check_grad(|b| a.matmul(b).mul_elem(&w).sum(), &b);
    assert!(error < 1e-2, "max error: {}", error);
}

#[test]
fn add_row_grad(){
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let x=Tensor::rand(rows, cols);
    let bias=Var::new(Tensor::rand(1, cols));
    let w=Var::new(Tensor::rand(rows, cols));
    let error=check_grad(|v| v.add_row(&bias).mul_elem(&w).sum(),&x);
    assert!(error < 1e-2, "max error: {}", error);
}

#[test]
fn add_row_bias_grad(){
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let x=Tensor::rand(1, cols);
    let x_cap=Var::new(Tensor::rand(rows, cols));
    let w=Var::new(Tensor::rand(rows, cols));
    let error=check_grad(|b| x_cap.add_row(b).mul_elem(&w).sum(),&x);
    assert!(error < 1e-2, "max error: {}", error);
}

#[test]
fn sub_first_entry_grad(){
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t=Tensor::rand(rows, cols);
    let b=Var::new(Tensor::rand(rows, cols));
    let w=Var::new(Tensor::rand(rows, cols));
    let error=check_grad(|a| a.sub(&b).mul_elem(&w).sum(),&t);
    assert!(error < 1e-2, "max error: {}", error);
}

#[test]
fn sub_second_entry_grad(){
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t=Tensor::rand(rows, cols);
    let a=Var::new(Tensor::rand(rows, cols));
    let w=Var::new(Tensor::rand(rows, cols));
    let error=check_grad(|b| a.sub(b).mul_elem(&w).sum(),&t);
    assert!(error < 1e-2, "max error: {}", error);
}

#[test]
fn mul_elem_first_entry_grad(){
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t=Tensor::rand(rows, cols);
    let b=Var::new(Tensor::rand(rows, cols));
    let w=Var::new(Tensor::rand(rows, cols));
    let error=check_grad(|a| a.mul_elem(&b).mul_elem(&w).sum(),&t);
    assert!(error < 1e-2, "max error: {}", error);
}

#[test]
fn mul_elem_second_entry(){
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t=Tensor::rand(rows, cols);
    let a=Var::new(Tensor::rand(rows, cols));
    let w=Var::new(Tensor::rand(rows, cols));
    let error=check_grad(|b| a.mul_elem(b).mul_elem(&w).sum(),&t);
    assert!(error < 1e-2, "max error: {}", error);
}