mod common;

use common::approx_eq;
use micrograd_rs::tensor::Tensor;

//-----------------
//--CONSTRUCTORS---
//-----------------

//FILL
#[test]
fn fill_size_test() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t = Tensor::fill(rows, cols, 0.0);
    assert_eq!(rows, t.rows());
    assert_eq!(cols, t.cols());
}
#[test]
fn fill_value_test() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let random_value = rand::random();
    let t = Tensor::fill(rows, cols, random_value);
    for value in t.data().iter() {
        assert_eq!(random_value, *value);
    }
}

//from-vec
#[test]
fn from_vec_order() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let rdm_vec: Vec<f32> = (0..rows * cols).map(|_| rand::random()).collect();
    let t = Tensor::from_vec(rdm_vec.clone(), rows, cols);
    for i in 0..rows {
        for j in 0..cols {
            assert_eq!(t.get(i, j), rdm_vec[i * cols + j]);
        }
    }
}
#[test]
fn from_vec_size_is_correct() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let rdm_vec: Vec<f32> = (0..rows * cols).map(|_| rand::random()).collect();
    let t = Tensor::from_vec(rdm_vec.clone(), rows, cols);
    assert_eq!(t.rows(), rows);
    assert_eq!(t.cols(), cols);
}
//rand
#[test]
fn rand_size_is_correct() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols);
    assert_eq!(rows, t.rows());
    assert_eq!(cols, t.cols());
}
#[test]
fn rand_value_are_in_range() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols);
    for i in 0..rows {
        for j in 0..cols {
            assert!(t.get(i, j) >= -1.0 && t.get(i, j) < 1.0);
        }
    }
}

//-----------------
//-----ACCESS------
//-----------------

//get-set
#[test]
fn get_get_set_value() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let random_value = rand::random();
    let i = rand::random_range(0..rows);
    let j = rand::random_range(0..cols);
    let mut t = Tensor::rand(rows, cols);
    t.set(i, j, random_value);
    assert_eq!(t.get(i, j), random_value);
}
//set
#[test]
#[should_panic(expected = "row index out of bounds")]
fn set_rows_out_of_bounds() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let mut t = Tensor::rand(rows, cols);
    t.set(rows, cols - 1, 1.0);
}
#[test]
#[should_panic(expected = "column index out of bounds")]
fn set_cols_out_of_bounds() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let mut t = Tensor::rand(rows, cols);
    t.set(rows - 1, cols, 1.0);
}
//get
#[test]
#[should_panic(expected = "row index out of bounds")]
fn get_rows_out_of_bounds() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols);
    t.get(rows, cols - 1);
}
#[test]
#[should_panic(expected = "column index out of bounds")]
fn get_cols_out_of_bounds() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols);
    t.get(rows - 1, cols);
}
#[test]
#[should_panic(expected = "column index out of bounds")]
fn get_cols_overflow_into_next_row() {
    //acces could be valid ([0*(cols+1) +j]) but shouldnt be accepted because there is only cols columns and not cols+1
    let rows = rand::random_range(2..10);
    let cols = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols);
    t.get(0, cols);
}
#[test]
fn set_only_change_one_value() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let mut t0 = Tensor::rand(rows, cols);
    let t1 = t0.clone();
    let changed_row = rand::random_range(0..rows);
    let changed_cols = rand::random_range(0..cols);
    let mut random_value = rand::random();
    while random_value == t0.get(changed_row, changed_cols) {
        random_value = rand::random();
    }
    t0.set(changed_row, changed_cols, random_value);
    for i in 0..rows {
        for j in 0..cols {
            if i == changed_row && j == changed_cols {
                assert_eq!(t0.get(i, j), random_value);
            } else {
                assert_eq!(t0.get(i, j), t1.get(i, j));
            }
        }
    }
}
//rows
#[test]
fn rows_should_give_rows() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols);
    assert_eq!(t.rows(), rows);
}
//cols
#[test]
fn cols_should_give_columns() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols);
    assert_eq!(t.cols(), cols);
}
//--------------
//--TRANSPOSE---
//--------------
#[test]
fn transpose_twice() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols);
    assert_eq!(t.transpose().transpose(), t);
}
#[test]
fn transpose_does_swap_cols_and_rows() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols);
    assert_eq!(t.rows(), t.transpose().cols());
    assert_eq!(t.cols(), t.transpose().rows());
}
#[test]
fn transpose_does_swap_data() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t0 = Tensor::rand(rows, cols);
    let t1 = t0.transpose();
    for i in 0..rows {
        for j in 0..cols {
            assert_eq!(t0.get(i, j), t1.get(j, i));
        }
    }
}
//--------------
//----MATMUL----
//--------------
#[test]
fn units_tests() {
    let mut a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3);
    let mut b = Tensor::from_vec(vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0], 3, 3);
    let mut c = a.matmul(&b);
    let mut r = Tensor::from_vec(
        vec![30.0, 24.0, 18.0, 84.0, 69.0, 54.0, 138.0, 114.0, 90.0],
        3,
        3,
    );
    assert_eq!(c, r);
    // (2x3) * (3x2) -> (2x2)
    a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    b = Tensor::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 3, 2);
    c = a.matmul(&b);
    r = Tensor::from_vec(vec![58.0, 64.0, 139.0, 154.0], 2, 2);
    assert_eq!(c, r);

    // with negative and 0 values (2x4) * (4x3) -> (2x3)
    a = Tensor::from_vec(vec![1.0, -2.0, 0.0, 3.0, 4.0, 5.0, -1.0, 2.0], 2, 4);
    b = Tensor::from_vec(
        vec![
            2.0, 0.0, -1.0, 1.0, 3.0, 2.0, 5.0, -2.0, 4.0, 0.0, 1.0, -3.0,
        ],
        4,
        3,
    );
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
    b = Tensor::from_vec(
        vec![1.0, 2.0, 0.0, -1.0, 4.0, 0.0, 3.0, 2.0, -2.0, 5.0, 1.0, 0.0],
        3,
        4,
    );
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
    a = Tensor::from_vec(
        vec![
            1.0, 0.0, 2.0, -1.0, 3.0, 1.0, 0.0, 2.0, -2.0, 4.0, 1.0, 0.0, 0.0, -1.0, 3.0, 2.0,
        ],
        4,
        4,
    );
    b = Tensor::from_vec(
        vec![
            2.0, 1.0, 0.0, -1.0, 0.0, 3.0, 1.0, 2.0, -1.0, 0.0, 2.0, 1.0, 3.0, -2.0, 1.0, 0.0,
        ],
        4,
        4,
    );
    c = a.matmul(&b);
    r = Tensor::from_vec(
        vec![
            -3.0, 3.0, 3.0, 1.0, 12.0, 2.0, 3.0, -1.0, -5.0, 10.0, 6.0, 11.0, 3.0, -7.0, 7.0, 1.0,
        ],
        4,
        4,
    );
    assert_eq!(c, r);
}
#[test]
fn matmul_size() {
    //create tensor (n x k) and tensor (k x m) = (n x m)
    let n = rand::random_range(1..10);
    let k = rand::random_range(1..10);
    let m = rand::random_range(1..10);
    let t0 = Tensor::rand(n, k);
    let t1 = Tensor::rand(k, m);
    let r = t0.matmul(&t1);
    assert_eq!(r.rows(), n);
    assert_eq!(r.cols(), m);
}
#[test]
#[should_panic]
fn matmul_existential_condition() {
    //create tensor (n x k) and tensor (k x m) = (n x m)
    let n = rand::random_range(1..10);
    let m = rand::random_range(1..10);
    let k0 = rand::random_range(1..10);
    let mut k1 = rand::random_range(1..10);
    while k0 == k1 {
        k1 = rand::random_range(1..10);
    }
    let t0 = Tensor::rand(n, k0);
    let t1 = Tensor::rand(k1, m);
    let _r = t0.matmul(&t1); //should panic
}
#[test]
fn matmul_identity() {
    // I x A = A
    let size = rand::random_range(1..10);
    let identity = Tensor::identity(size);
    let t0 = Tensor::rand(size, size);
    let r = t0.matmul(&identity);
    assert_eq!(r, t0)
}
#[test]
fn matmul_transpose() {
    // (AB)^T=B^T x A^T
    let n = rand::random_range(1..10);
    let k = rand::random_range(1..10);
    let m = rand::random_range(1..10);
    let t0 = Tensor::rand(n, k);
    let t1 = Tensor::rand(k, m);
    let transpose_ab = t0.matmul(&t1).transpose();
    let transpose_b_transpose_a = t1.transpose().matmul(&t0.transpose());
    assert_eq!(transpose_b_transpose_a, transpose_ab);
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
fn matmul_is_associative() {
    // (A x B) x C = A x (B x C)
    let n = rand::random_range(1..10);
    let k = rand::random_range(1..10);
    let m = rand::random_range(1..10);
    let p = rand::random_range(1..10);
    let a = Tensor::rand(n, k);
    let b = Tensor::rand(k, m);
    let c = Tensor::rand(m, p);
    let axb_x = (a.matmul(&b)).matmul(&c);
    let ax_bxc = a.matmul(&b.matmul(&c));
    assert!(approx_eq(&ax_bxc, &axb_x));
}
//--------------
//--OPERATION---
//--------------

//add
#[test]
fn add_size_check() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t0 = Tensor::rand(rows, cols);
    let t1 = Tensor::rand(rows, cols);
    let r = t0.add(&t1);
    assert_eq!(r.rows(), t0.rows());
    assert_eq!(r.cols(), t0.cols());
}
#[test]
fn add_values_check() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t0 = Tensor::rand(rows, cols);
    let t1 = Tensor::rand(rows, cols);
    let r = t0.add(&t1);
    for i in 0..rows {
        for j in 0..cols {
            assert_eq!(r.get(i, j), t0.get(i, j) + t1.get(i, j));
        }
    }
}
#[test]
#[should_panic]
fn add_different_size() {
    let rows0 = rand::random_range(1..10);
    let cols0 = rand::random_range(1..10);
    let mut rows1 = rand::random_range(1..10);
    let mut cols1 = rand::random_range(1..10);
    while rows0 == rows1 {
        rows1 = rand::random_range(1..10);
    }
    while cols0 == cols1 {
        cols1 = rand::random_range(1..10);
    }
    let t0 = Tensor::rand(rows0, cols0);
    let t1 = Tensor::rand(rows1, cols1);
    let _r = t0.add(&t1);
}
#[test]
fn add_neutral_element() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t0 = Tensor::rand(rows, cols);
    let t1 = Tensor::fill(rows, cols, 0.0);
    let r = t0.add(&t1);
    assert_eq!(t0, r);
}
#[test]
fn add_is_commutative() {
    //A+B=B+A
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t0 = Tensor::rand(rows, cols);
    let t1 = Tensor::rand(rows, cols);
    assert!(approx_eq(&t0.add(&t1), &t1.add(&t0)));
}
#[test]
fn add_is_associative() {
    //(A+B)+C=A+(B+C)
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let a = Tensor::rand(rows, cols);
    let b = Tensor::rand(rows, cols);
    let c = Tensor::rand(rows, cols);
    let ab_c = (a.add(&b)).add(&c);
    let a_bc = a.add(&b.add(&c));
    assert!(approx_eq(&ab_c, &a_bc));
}
#[test]
fn add_himself_is_scale_by_two() {
    let rows0 = rand::random_range(1..10);
    let cols0 = rand::random_range(1..10);
    let t = Tensor::rand(rows0, cols0);
    let r = t.add(&t);
    let r2 = t.scale(2.0);
    assert_eq!(&r, &r2);
}
//mul elem
#[test]
fn mul_elem_size_check() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t0 = Tensor::rand(rows, cols);
    let t1 = Tensor::rand(rows, cols);
    let r = t0.mul_elem(&t1);
    assert_eq!(r.rows(), t0.rows());
    assert_eq!(r.cols(), t0.cols());
}
#[test]
fn mul_elem_values_check() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t0 = Tensor::rand(rows, cols);
    let t1 = Tensor::rand(rows, cols);
    let r = t0.mul_elem(&t1);
    for i in 0..rows {
        for j in 0..cols {
            assert_eq!(r.get(i, j), t0.get(i, j) * t1.get(i, j));
        }
    }
}
#[test]
#[should_panic]
fn mul_elem_different_size() {
    let rows0 = rand::random_range(1..10);
    let cols0 = rand::random_range(1..10);
    let mut rows1 = rand::random_range(1..10);
    let mut cols1 = rand::random_range(1..10);
    while rows0 == rows1 {
        rows1 = rand::random_range(1..10);
    }
    while cols0 == cols1 {
        cols1 = rand::random_range(1..10);
    }
    let t0 = Tensor::rand(rows0, cols0);
    let t1 = Tensor::rand(rows1, cols1);
    let _r = t0.mul_elem(&t1); //should panic
}
#[test]
fn mul_elem_neutral_element() {
    // A * 1 = A
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols);
    let one = Tensor::fill(rows, cols, 1.0);
    assert_eq!(t, t.mul_elem(&one));
}
#[test]
fn mul_elem_absorbing_element() {
    // A * 0 = 0
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols);
    let zero = Tensor::fill(rows, cols, 0.0);
    assert_eq!(zero, t.mul_elem(&zero));
}
#[test]
fn mul_elem_is_commutative() {
    //A*B=B*A
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t0 = Tensor::rand(rows, cols);
    let t1 = Tensor::rand(rows, cols);
    assert_eq!(t0.mul_elem(&t1), t1.mul_elem(&t0));
}
#[test]
fn mul_elem_is_associative() {
    //(A*B)*C=A*(B*C)
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let a = Tensor::rand(rows, cols);
    let b = Tensor::rand(rows, cols);
    let c = Tensor::rand(rows, cols);
    let ab_c = (a.mul_elem(&b)).mul_elem(&c);
    let a_bc = a.mul_elem(&b.mul_elem(&c));
    assert!(approx_eq(&ab_c, &a_bc));
}
#[test]
fn mul_elem_is_distributive() {
    //(A+B)*C=A*C + B*C
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let a = Tensor::rand(rows, cols);
    let b = Tensor::rand(rows, cols);
    let c = Tensor::rand(rows, cols);
    let ab_c = (a.add(&b)).mul_elem(&c);
    let ac_bc = (a.mul_elem(&c)).add(&b.mul_elem(&c));
    assert!(approx_eq(&ab_c, &ac_bc));
}
//div_elem
#[test]
fn div_elem_size_check() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t0 = Tensor::rand(rows, cols);
    let t1 = Tensor::rand(rows, cols);
    let r = t0.div_elem(&t1);
    assert_eq!(r.rows(), t0.rows());
    assert_eq!(r.cols(), t0.cols());
}
#[test]
fn div_elem_values_check() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t0 = Tensor::rand(rows, cols);
    let t1 = Tensor::rand(rows, cols);
    let r = t0.div_elem(&t1);
    for i in 0..rows {
        for j in 0..cols {
            assert_eq!(r.get(i, j), t0.get(i, j) / t1.get(i, j));
        }
    }
}
#[test]
#[should_panic]
fn div_elem_different_size() {
    let rows0 = rand::random_range(1..10);
    let cols0 = rand::random_range(1..10);
    let mut rows1 = rand::random_range(1..10);
    let mut cols1 = rand::random_range(1..10);
    while rows0 == rows1 {
        rows1 = rand::random_range(1..10);
    }
    while cols0 == cols1 {
        cols1 = rand::random_range(1..10);
    }
    let t0 = Tensor::rand(rows0, cols0);
    let t1 = Tensor::rand(rows1, cols1);
    let _r = t0.div_elem(&t1); //should panic
}

#[test]
fn div_elem_neutral_element() {
    // A / 1 = A
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols);
    let one = Tensor::fill(rows, cols, 1.0);
    assert_eq!(t, t.div_elem(&one));
}
#[test]
fn div_elem_absorbing_element() {
    // 0 / A = 0
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols);
    let zero = Tensor::fill(rows, cols, 0.0);
    assert_eq!(zero, zero.div_elem(&t));
}
#[test]
fn div_elem_a_on_a_is_one() {
    // A / A = 1
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols);
    let one = Tensor::fill(rows, cols, 1.0);
    assert_eq!(&one, &t.div_elem(&t));
}
#[test]
fn div_elem_is_distributive_by_right() {
    //(A+B)/C=A/C + B/C
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let a = Tensor::rand(rows, cols);
    let b = Tensor::rand(rows, cols);
    let c = Tensor::rand(rows, cols).map(|x| x + 2.0); //to avoid instability of dividing by near 0 values
    let ab_c = (a.add(&b)).div_elem(&c);
    let ac_bc = (a.div_elem(&c)).add(&b.div_elem(&c));
    assert!(approx_eq(&ab_c, &ac_bc));
}

#[test]
fn div_elem_is_inverse_function_of_mul_elem() {
    // (A/B)*B=A
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let a = Tensor::rand(rows, cols);
    let b = Tensor::rand(rows, cols).map(|x| x + 2.0);
    let r = (a.div_elem(&b)).mul_elem(&b);
    assert!(approx_eq(&a, &r));
}

//scale
#[test]
fn scale_size_check() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let random_value = rand::random();
    let t0 = Tensor::rand(rows, cols);
    let r = t0.scale(random_value);
    assert_eq!(r.rows(), rows);
    assert_eq!(r.cols(), cols);
}
#[test]
fn scale_elem_values_check() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let random_value = rand::random();
    let t0 = Tensor::rand(rows, cols);
    let r = t0.scale(random_value);
    for i in 0..rows {
        for j in 0..cols {
            assert_eq!(r.get(i, j), t0.get(i, j) * random_value);
        }
    }
}
#[test]
fn scale_by_one_is_himself() {
    //A*1=A
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t0 = Tensor::rand(rows, cols);
    let r = t0.scale(1.0);
    for i in 0..rows {
        for j in 0..cols {
            assert_eq!(r.get(i, j), t0.get(i, j));
        }
    }
}
#[test]
fn scale_by_zero_is_zero() {
    //A*1=A
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t0 = Tensor::rand(rows, cols);
    let r = t0.scale(0.0);
    let zero = Tensor::fill(rows, cols, 0.0);
    assert_eq!(r, zero);
}
#[test]
fn scale_by_a_then_b_is_scale_by_a_times_b() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols);
    let a = rand::random();
    let b = rand::random();
    let r0 = t.scale(a).scale(b);
    let r1 = t.scale(a * b);
    assert!(approx_eq(&r0, &r1));
}
#[test]
fn scale_a_by_b_is_a_mul_elem_fill_b() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols);
    let b = rand::random();
    let r = t.scale(b);
    let r1 = t.mul_elem(&Tensor::fill(rows, cols, b));
    assert_eq!(r, r1);
}
#[test]
fn map_size_check() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols);
    let r = t.map(|x| x);
    assert_eq!(r.rows(), rows);
    assert_eq!(r.cols(), cols);
}
#[test]
fn map_chain_rules() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols);
    let f = |x: f32| x.tanh();
    let g = |x: f32| x.sin();
    let r = t.map(g).map(f);
    let r2 = t.map(|x| f(g(x)));
    assert_eq!(r, r2);
}
#[test]
fn map_times_a_is_scale_by_a() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols);
    let random_value = rand::random();
    let r = t.scale(random_value);
    let r1 = t.map(|x| x * random_value);
    assert_eq!(r, r1);
}
//---------------
//------SUM------
//---------------
#[test]
fn sum_units_test() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    assert_eq!(t.sum(), 10.0);
    let t = Tensor::from_vec(vec![0.5, -1.5, 2.0, 1.5, 2.5, -2.0], 2, 3);
    assert_eq!(t.sum(), 3.0);
}
#[test]
fn sum_value_check() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols);
    let r = t.sum();
    let sum: f32 = t.data().iter().sum();
    assert_eq!(sum, r);
}
#[test]
fn sum_row_size_check() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols);
    let row = t.sum_rows();
    assert_eq!(row.rows(), 1);
}
#[test]
fn sum_of_sum_row_is_sum_of_t() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols);
    let row = t.sum_rows();
    assert!((t.sum() - row.sum()).abs() < 1e-4)
}
#[test]
fn sum_rows_hand_computed_values() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
    assert_eq!(t.sum_rows(), Tensor::from_vec(vec![9.0, 12.0], 1, 2));
    let t = Tensor::from_vec(vec![1.0, -2.0, 3.0], 1, 3);
    assert_eq!(t.sum_rows(), Tensor::from_vec(vec![1.0, -2.0, 3.0], 1, 3));
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], 4, 1);
    assert_eq!(t.sum_rows(), Tensor::from_vec(vec![10.0], 1, 1));
    let t = Tensor::from_vec(vec![0.5, -1.5, 2.0, 1.5, 2.5, -2.0], 2, 3);
    assert_eq!(t.sum_rows(), Tensor::from_vec(vec![2.0, 1.0, 0.0], 1, 3));
}
//---------------
//----ADD-ROW----
//---------------
#[test]
fn add_row_size_check() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols);
    let row = Tensor::rand(1, cols);
    let r = t.add_row(&row);
    assert_eq!(t.rows(), r.rows());
}
#[test]
#[should_panic]
fn add_row_to_many_lines() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols);
    let row = Tensor::rand(rand::random_range(2..10), cols);
    let _r = t.add_row(&row); //should panic
}
#[test]
#[should_panic]
fn add_row_to_many_cols() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols);
    let row = Tensor::rand(1, cols + 1);
    let _r = t.add_row(&row); //should panic
}
#[test]
fn add_row_value_check() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols);
    let row = Tensor::rand(1, cols);
    let r = t.add_row(&row);
    for i in 0..rows {
        for j in 0..cols {
            assert_eq!(t.get(i, j) + row.get(0, j), r.get(i, j));
        }
    }
}

#[test]
fn softmax_rows_size_check() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols);
    let r = t.softmax_rows();
    assert_eq!(r.rows(), t.rows());
    assert_eq!(r.cols(), t.cols());
}

#[test]
fn softmax_rows_sum_to_one() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols);
    let r = t.softmax_rows();
    for i in 0..rows {
        let mut sum = 0.0;
        for j in 0..cols {
            sum += r.get(i, j);
        }
        assert!((sum - 1.0).abs() < 1e-5);
    }
}

#[test]
fn softmax_rows_value_are_between_zero_and_ones() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols);
    let r = t.softmax_rows();
    for i in 0..rows {
        for j in 0..cols {
            assert!(r.get(i, j) >= 0.0 && r.get(i, j) <= 1.0);
        }
    }
}

#[test]
fn softmax_rows_adding_bias_does_not_change_result() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols);
    let r = t.softmax_rows();
    let r2 = t.map(|v| v + 5.0).softmax_rows();
    assert!(approx_eq(&r, &r2))
}

#[test]
fn softmax_rows_units_tests() {
    let t = Tensor::from_vec(vec![0.0, std::f32::consts::LN_2, 0.0, 0.0, 0.0, 0.0], 2, 3);
    let expected = Tensor::from_vec(vec![0.25, 0.5, 0.25, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], 2, 3);
    assert!(approx_eq(&t.softmax_rows(), &expected));
}

#[test]
fn softmax_rows_is_stable_with_extreme_values() {
    let t = Tensor::from_vec(vec![1000.0, 1000.0], 1, 2);
    let expected = Tensor::from_vec(vec![0.5, 0.5], 1, 2);
    assert!(approx_eq(&t.softmax_rows(), &expected));

    let t = Tensor::from_vec(vec![-1000.0, -1001.0], 1, 2);
    let expected = Tensor::from_vec(vec![0.731_059, 0.268_941], 1, 2);
    assert!(approx_eq(&t.softmax_rows(), &expected));
}

#[test]
fn log_softmax_rows_exp_is_softmax() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols);
    let r = t.log_softmax_rows().map(|x| x.exp());
    assert!(approx_eq(&r, &t.softmax_rows()));
}

#[test]
fn log_softmax_rows_is_finite_with_extreme_values() {
    // log(softmax) would give log(0) = -inf for the second value
    let t = Tensor::from_vec(vec![0.0, -200.0], 1, 2);
    let r = t.log_softmax_rows();
    assert!(r.data().iter().all(|x| x.is_finite()));
    let expected = Tensor::from_vec(vec![0.0, -200.0], 1, 2);
    assert!(approx_eq(&r, &expected));
}

#[test]
fn select_rows_size_check() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let k = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols);
    let indices: Vec<usize> = (0..k).map(|_| rand::random_range(0..rows)).collect();
    let r = t.select_rows(&indices);
    assert_eq!(r.rows(), k);
    assert_eq!(r.cols(), cols)
}

#[test]
fn select_rows_value_check() {
    let rows = rand::random_range(1..10);
    let cols = rand::random_range(1..10);
    let k = rand::random_range(1..10);
    let t = Tensor::rand(rows, cols);
    let indices: Vec<usize> = (0..k).map(|_| rand::random_range(0..rows)).collect();
    let r = t.select_rows(&indices);
    for i in 0..k {
        for j in 0..cols {
            assert_eq!(r.get(i, j), t.get(indices[i], j));
        }
    }
}
