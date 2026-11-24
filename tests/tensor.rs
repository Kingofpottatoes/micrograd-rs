
use micrograd_rs::tensor::Tensor;


fn approx_eq(a: &Tensor, b: &Tensor) -> bool{
    let tolerance=1e-4;
    if a.rows()!=b.rows(){return false;}
    if a.cols()!=b.cols(){return false;}
    let all_almost_equals=a.data().iter().zip(b.data().iter()).all(|(a,b)| (*a-*b).abs()<=tolerance);
    return all_almost_equals;
}
//-----------------
//--CONSTRUCTORS---
//-----------------

//FILL
#[test]
fn fill_size_test(){
    let rows=rand::random_range(1..10);
    let cols=rand::random_range(1..10);
    let t=Tensor::fill(rows, cols,0.0);
    assert_eq!(rows,t.rows());
    assert_eq!(cols,t.cols());
}
#[test]
fn fill_value_test(){
    let rows=rand::random_range(1..10);
    let cols=rand::random_range(1..10);
    let random_value=rand::random();
    let t=Tensor::fill(rows, cols,random_value);
    for value in t.data().iter(){
        assert_eq!(random_value,*value);
    }
}


//from-vec
#[test]
fn from_vec_order(){
    let rows=rand::random_range(1..10);
    let cols=rand::random_range(1..10);
    let rdm_vec:Vec<f32>=(0..rows*cols).map(|_| rand::random()).collect();
    let t=Tensor::from_vec(rdm_vec.clone(), rows, cols);
    for i in 0..rows{
        for j in 0..cols{
            assert_eq!(t.get(i, j),rdm_vec[i*cols+j]);
        }
    }
}
#[test]
fn from_vec_size_is_correct(){
    let rows=rand::random_range(1..10);
    let cols=rand::random_range(1..10);
    let rdm_vec:Vec<f32>=(0..rows*cols).map(|_| rand::random()).collect();
    let t=Tensor::from_vec(rdm_vec.clone(), rows, cols);
    assert_eq!(t.rows(),rows);
    assert_eq!(t.cols(),cols);
}
//rand
#[test]
fn rand_size_is_correct(){
    let rows=rand::random_range(1..10);
    let cols=rand::random_range(1..10);
    let t=Tensor::rand(rows, cols);
    assert_eq!(rows,t.rows());
    assert_eq!(cols,t.cols());
}
#[test]
fn rand_value_are_in_range(){
    let rows=rand::random_range(1..10);
    let cols=rand::random_range(1..10);
    let t=Tensor::rand(rows, cols);
    for i in 0..rows{
        for j in 0..cols{
            assert!(t.get(i, j)>= -1.0 && t.get(i, j)< 1.0);
        }
    }
}

//-----------------
//-----ACCESS------
//-----------------

//get-set
#[test]
fn get_get_set_value(){
    let rows=rand::random_range(1..10);
    let cols=rand::random_range(1..10);
    let random_value=rand::random();
    let i=rand::random_range(0..rows);
    let j=rand::random_range(0..cols);
    let mut t=Tensor::rand(rows, cols);
    t.set(i,j,random_value);
    assert_eq!(t.get(i,j),random_value);
}
//set
#[test]
#[should_panic(expected = "row index out of bounds")]
fn set_rows_out_of_bounds(){
    let rows=rand::random_range(1..10);
    let cols=rand::random_range(1..10);
    let mut t=Tensor::rand(rows, cols);
    t.set(rows,cols-1,1.0);
}
#[test]
#[should_panic(expected = "column index out of bounds")]
fn set_cols_out_of_bounds(){
    let rows=rand::random_range(1..10);
    let cols=rand::random_range(1..10);
    let mut t=Tensor::rand(rows, cols);
    t.set(rows-1,cols,1.0);
}
//get
#[test]
#[should_panic(expected = "row index out of bounds")]
fn get_rows_out_of_bounds(){
    let rows=rand::random_range(1..10);
    let cols=rand::random_range(1..10);
    let t=Tensor::rand(rows, cols);
    t.get(rows,cols-1);
}
#[test]
#[should_panic(expected = "column index out of bounds")]
fn get_cols_out_of_bounds(){
    let rows=rand::random_range(1..10);
    let cols=rand::random_range(1..10);
    let t=Tensor::rand(rows, cols);
    t.get(rows-1,cols);
}
#[test]
#[should_panic(expected = "column index out of bounds")]
fn get_cols_overflow_into_next_row(){
    //acces could be valid ([0*(cols+1) +j]) but shouldnt be accepted because there is only cols columns and not cols+1
    let rows=rand::random_range(2..10);
    let cols=rand::random_range(1..10);
    let t=Tensor::rand(rows, cols);
    t.get(0,cols);
}
#[test]
fn set_only_change_one_value(){
    let rows=rand::random_range(1..10);
    let cols=rand::random_range(1..10);
    let mut t0=Tensor::rand(rows, cols);
    let t1=t0.clone();
    let changed_row=rand::random_range(0..rows);
    let changed_cols=rand::random_range(0..cols);
    let mut random_value=rand::random();
    while random_value==t0.get(changed_row,changed_cols) {
        random_value=rand::random();
    }
    t0.set(changed_row,changed_cols,random_value);
    for i in 0..rows{
        for j in 0..cols{
            if i==changed_row && j==changed_cols{
                assert_eq!(t0.get(i,j),random_value);
            }
            else{
                assert_eq!(t0.get(i,j),t1.get(i,j));
            }
            
        }
    }
}
//rows
#[test]
fn rows_should_give_rows(){
    let rows=rand::random_range(1..10);
    let cols=rand::random_range(1..10);
    let t=Tensor::rand(rows, cols);
    assert_eq!(t.rows(),rows);
}
//cols
#[test]
fn cols_should_give_columns(){
    let rows=rand::random_range(1..10);
    let cols=rand::random_range(1..10);
    let t=Tensor::rand(rows, cols);
    assert_eq!(t.cols(),cols);
}
//--------------
//--TRANSPOSE---
//--------------
#[test]
fn transpose_twice(){
    let rows=rand::random_range(1..10);
    let cols=rand::random_range(1..10);
    let t=Tensor::rand(rows, cols);
    assert_eq!(t.transpose().transpose(),t);
}
#[test]
fn transpose_does_swap_cols_and_rows(){
    let rows=rand::random_range(1..10);
    let cols=rand::random_range(1..10);
    let t=Tensor::rand(rows, cols);
    assert_eq!(t.rows(),t.transpose().cols());
    assert_eq!(t.cols(),t.transpose().rows());
}
#[test]
fn transpose_does_swap_data(){
    let rows=rand::random_range(1..10);
    let cols=rand::random_range(1..10);
    let t0=Tensor::rand(rows, cols);
    let t1=t0.transpose();
    for i in 0..rows{
        for j in 0..cols{
            assert_eq!(t0.get(i, j),t1.get(j,i));
        }
    }
}
//--------------
//----MATMUL----
//--------------
#[test]
fn units_tests(){
    let mut a=Tensor::from_vec(vec![1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0], 3, 3);
    let mut b=Tensor::from_vec(vec![9.0,8.0,7.0,6.0,5.0,4.0,3.0,2.0,1.0], 3, 3);
    let mut c=a.matmul(&b);
    let mut r=Tensor::from_vec(vec![30.0,24.0,18.0,84.0,69.0,54.0,138.0,114.0,90.0], 3, 3);
    assert_eq!(c,r);
    // (2x3) * (3x2) -> (2x2)
    a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    b = Tensor::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 3, 2);
    c = a.matmul(&b);
    r = Tensor::from_vec(vec![58.0, 64.0, 139.0, 154.0], 2, 2);
    assert_eq!(c, r);

    // with negative and 0 values (2x4) * (4x3) -> (2x3)
    a = Tensor::from_vec(vec![1.0, -2.0, 0.0, 3.0, 4.0, 5.0, -1.0, 2.0], 2, 4);
    b = Tensor::from_vec(vec![2.0, 0.0, -1.0, 1.0, 3.0, 2.0, 5.0, -2.0, 4.0, 0.0, 1.0, -3.0], 4, 3);
    c = a.matmul(&b);
    r = Tensor::from_vec(vec![0.0, -3.0, -14.0, 8.0, 19.0, -4.0], 2, 3);
    assert_eq!(c, r);

    // (4x2) * (2x1) -> (4x1)
    a = Tensor::from_vec(vec![2.0, 1.0, 0.0, 3.0, -1.0, 4.0, 5.0, -2.0], 4, 2);
    b = Tensor::from_vec(vec![3.0, 2.0], 2, 1);
    c = a.matmul(&b);
    r = Tensor::from_vec(vec![8.0, 6.0, 5.0, 11.0], 4, 1);
    assert_eq!(c, r);

    //(1x3) * (3x4) -> (1x4)
    a = Tensor::from_vec(vec![2.0, -1.0, 3.0], 1, 3);
    b = Tensor::from_vec(vec![1.0, 2.0, 0.0, -1.0, 4.0, 0.0, 3.0, 2.0, -2.0, 5.0, 1.0, 0.0], 3, 4);
    c = a.matmul(&b);
    r = Tensor::from_vec(vec![-8.0, 19.0, 0.0, -4.0], 1, 4);
    assert_eq!(c, r);

    // 2x2
    a = Tensor::from_vec(vec![1.5, -2.0, 0.5, 4.0], 2, 2);
    b = Tensor::from_vec(vec![2.0, 3.0, -1.5, 0.5], 2, 2);
    c = a.matmul(&b);
    r = Tensor::from_vec(vec![6.0, 3.5, -5.0, 3.5], 2, 2);
    assert_eq!(c, r);

    //4x4 and i will not calculate more matrix T-T
    a = Tensor::from_vec(vec![
        1.0, 0.0, 2.0, -1.0,
        3.0, 1.0, 0.0, 2.0,
        -2.0, 4.0, 1.0, 0.0,
        0.0, -1.0, 3.0, 2.0,
    ], 4, 4);
    b = Tensor::from_vec(vec![
        2.0, 1.0, 0.0, -1.0,
        0.0, 3.0, 1.0, 2.0,
        -1.0, 0.0, 2.0, 1.0,
        3.0, -2.0, 1.0, 0.0,
    ], 4, 4);
    c = a.matmul(&b);
    r = Tensor::from_vec(vec![
        -3.0, 3.0, 3.0, 1.0,
        12.0, 2.0, 3.0, -1.0,
        -5.0, 10.0, 6.0, 11.0,
        3.0, -7.0, 7.0, 1.0,
    ], 4, 4);
    assert_eq!(c, r);
}
#[test]
fn matmul_size(){
    //create tensor (n x k) and tensor (k x m) = (n x m)
    let n=rand::random_range(1..10);
    let k=rand::random_range(1..10);
    let m=rand::random_range(1..10);
    let t0=Tensor::rand(n, k);
    let t1=Tensor::rand(k, m);
    let r=t0.matmul(&t1);
    assert_eq!(r.rows(),n);
    assert_eq!(r.cols(),m);
}
#[test]
#[should_panic]
fn matmul_existential_condition(){
    //create tensor (n x k) and tensor (k x m) = (n x m)
    let n=rand::random_range(1..10);
    let m=rand::random_range(1..10);
    let k0=rand::random_range(1..10);
    let mut k1=rand::random_range(1..10);
    while k0==k1 {
        k1=rand::random_range(1..10);
    }
    let t0=Tensor::rand(n, k0);
    let t1=Tensor::rand(k1, m);
    let _r=t0.matmul(&t1); //should panic
}
#[test]
fn matmul_identity(){ // I x A = A
    let size=rand::random_range(1..10);
    let identity=Tensor::identity(size);
    let t0=Tensor::rand(size, size);
    let r=t0.matmul(&identity);
    assert_eq!(r,t0)
}
#[test]
fn matmul_transpose(){
    // (AB)^T=B^T x A^T
    let n=rand::random_range(1..10);
    let k=rand::random_range(1..10);
    let m=rand::random_range(1..10);
    let t0=Tensor::rand(n, k);
    let t1=Tensor::rand(k, m);
    let transpose_ab= t0.matmul(&t1).transpose();
    let transpose_b_transpose_a=t1.transpose().matmul(&t0.transpose());
    assert_eq!(transpose_b_transpose_a,transpose_ab);
}
#[test]
fn matmul_is_not_commutative() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = Tensor::from_vec(vec![0.0, 1.0, 1.0, 0.0], 2, 2);
    let ab = a.matmul(&b);
    let ba = b.matmul(&a);
    assert_eq!(ab, Tensor::from_vec(vec![2.0, 1.0, 4.0, 3.0], 2, 2));
    assert_eq!(ba, Tensor::from_vec(vec![3.0, 4.0, 1.0, 2.0], 2, 2));
    assert_ne!(ab, ba);

    let a = Tensor::from_vec(vec![1.0, 2.0, 0.0, 1.0], 2, 2);
    let b = Tensor::from_vec(vec![1.0, 0.0, 3.0, 1.0], 2, 2);
    let ab = a.matmul(&b);
    let ba = b.matmul(&a);
    assert_eq!(ab, Tensor::from_vec(vec![7.0, 2.0, 3.0, 1.0], 2, 2));
    assert_eq!(ba, Tensor::from_vec(vec![1.0, 2.0, 3.0, 7.0], 2, 2));
    assert_ne!(ab, ba);
}
#[test]
fn matmul_is_associative(){// (A x B) x C = A x (B x C)
    let n=rand::random_range(1..10);
    let k=rand::random_range(1..10);
    let m=rand::random_range(1..10);
    let p=rand::random_range(1..10);
    let a=Tensor::rand(n, k);
    let b=Tensor::rand(k, m);
    let c=Tensor::rand(m, p);
    let axb_x=(a.matmul(&b)).matmul(&c);
    let ax_bxc=a.matmul(&b.matmul(&c));
    assert!(approx_eq(&ax_bxc,&axb_x));
}