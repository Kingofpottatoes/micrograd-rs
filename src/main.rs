
use std::rc::Rc;
use std::cell::RefCell;
use std::ops::Add;
use std::ops::Mul;
use std::collections::HashSet;

#[derive(Debug, Clone, Copy)]
enum Op {
    Leaf,
    Add,
    Mul,
    Tanh,
}
#[derive(Debug)]
struct ValueData{
    data: f64,
    grad: f64,
    children: Vec<Value>,
    op: Op,
}

#[derive(Debug, Clone)]
struct Value(Rc<RefCell<ValueData>>);

impl Value{
    fn new(data: f64) -> Value{
        let v=ValueData{
            data:data,
            grad:0.0,
            children:Vec::new(),
            op:Op::Leaf,
        };
        Value(Rc::new(RefCell::new(v)))
    }
    fn tanh(&self) -> Value {
        let data=self.0.borrow().data.tanh();
        let children=vec![self.clone()];
        return Value(Rc::new(RefCell::new(ValueData {
            data:data,
            grad: 0.0,
            children:children,
            op:Op::Tanh,
        })));
    }
    fn local_backward(&self){
        let data=self.0.borrow().data;
        let children=self.0.borrow().children.clone();
        let grad=self.0.borrow().grad;
        let op=self.0.borrow().op;
        match op {
            Op::Leaf =>{}
            Op::Add =>{
                for child in children {
                    child.0.borrow_mut().grad+=grad;
                }
            }
            Op::Mul =>{
                let child_zero_data=children[0].0.borrow().data;
                let child_one_data=children[1].0.borrow().data;
                children[0].0.borrow_mut().grad += child_one_data*grad;
                children[1].0.borrow_mut().grad += child_zero_data*grad;
            }
            Op::Tanh =>{
                children[0].0.borrow_mut().grad += (1.0- (data.powi(2)))*grad;
            }
        };
    }
    fn backward(&self){
        let mut visited=HashSet::new();
        let mut topo=Vec::new();
        build_topo(self,&mut visited, &mut topo);
        self.0.borrow_mut().grad=1.0;
        for node in topo.iter().rev() {
            node.local_backward();
        }


    }
    
}

impl Add for Value {
    type Output = Value;
    fn add(self, other: Value) -> Value {
        let data=self.0.borrow().data + other.0.borrow().data;
        let children=vec![self,other];
        Value(Rc::new(RefCell::new(ValueData {
            data:data,
            grad: 0.0,
            children:children,
            op:Op::Add,
        })))
    }
}
impl Mul for Value {
    type Output = Value;
    fn mul(self, other: Value) -> Value {
        let data=self.0.borrow().data * other.0.borrow().data;
        let children=vec![self,other];
        Value(Rc::new(RefCell::new(ValueData {
            data:data,
            grad: 0.0,
            children:children,
            op:Op::Mul,
        })))
    }
}

fn build_topo(node:&Value,visited:&mut HashSet<*const RefCell<ValueData>>,topo:&mut Vec<Value>){
    let id=Rc::as_ptr(&node.0);
    if !visited.insert(id){return;}
    let children=node.0.borrow().children.clone();
    for child in children {
        build_topo(&child, visited, topo);
    }
    topo.push(node.clone());
}

fn main() {
    let x1=Value::new(2.0);
    let x2=Value::new(0.0);
    let w1=Value::new(-3.0);
    let w2=Value::new(1.0);
    let b=Value::new(6.8813735870195432);

    let n=x1.clone()*w1.clone() +x2.clone() *w2.clone() +b.clone();
    let o= n.tanh();

    o.backward();

    println!("o={}",o.0.borrow().data);
    println!("x1= {}",x1.0.borrow().grad);
    println!("x2={}",x2.0.borrow().grad);
    println!("w1={}",w1.0.borrow().grad);
    println!("w2={}",w2.0.borrow().grad);
    println!("b={}",b.0.borrow().grad);

}