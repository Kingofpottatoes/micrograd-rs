use rand::RngExt;





pub struct Tensor {
    data:Vec<f32>,
    rows:usize,
    columns:usize,
}

impl Tensor{
    pub fn zero(rows:usize,cols:usize)->Tensor{
        Tensor{
            data:vec![0.0;rows*cols],
            rows:rows,
            columns:cols,
        }
    }
    pub fn from_vec(data:Vec<f32>,rows:usize,cols:usize)->Tensor{
        assert_eq!(data.len(),rows * cols);
        Tensor{
            data:data,
            rows:rows,
            columns:cols,
        }
    }
    pub fn rand(rows:usize,cols:usize)->Tensor{
        let mut data=Vec::new();
        let mut rng = rand::rng();
        for _ in 0..(rows*cols){
            data.push(rng.random_range(-1.0..1.0));
        }
        Tensor { 
            data: data, 
            rows:rows, 
            columns: cols 
        }
    }
    pub fn get(&self,i:usize,j:usize)->f32{
        assert!(i*j<self.data.len());
        return self.data[i*self.columns +j]
    }
    pub fn set(&mut self,i:usize,j:usize,value:f32){
        assert!(i*j<self.data.len());
        self.data[i*self.columns +j]=value
    }
    pub fn matmul(&self,other:&Tensor)->Tensor{
        assert_eq!(self.columns,other.rows);
        let mut data=Vec::new();
        for i in 0..self.rows{
            for j in 0..other.columns{
                let mut sum:f32=0.0;
                for k in 0..self.columns{
                    sum+=self.data[k+self.columns*i]*other.data[k*other.columns +j];
                }
                data.push(sum);
            }
        }
        Tensor {
            data:data, 
            rows: self.rows, 
            columns: other.columns,
        }
    }
}