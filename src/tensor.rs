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
        assert!(i<self.rows);
        assert!(j<self.columns);
        return self.data[i*self.columns +j]
    }
    pub fn set(&mut self,i:usize,j:usize,value:f32){
        assert!(i<self.rows);
        assert!(j<self.columns);
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
    pub fn transpose(&self)->Tensor{
        let mut trans=Tensor::zero(self.columns,self.rows);
        for i in 0..self.rows{
            for j in 0..self.columns{
                trans.set(j,i,self.get(i,j));
            }
        }
        return trans;
    }
    pub fn add_row(&self,row:&Tensor)->Tensor{
        assert_eq!(row.rows,1);
        assert_eq!(row.columns,self.columns);
        let mut data=Vec::new();
        for i in 0..self.rows{
            for j in 0..self.columns{
                data.push(self.data[i*self.columns+j]+row.data[j]);
            }
        }
        Tensor {
            data:data, 
            rows: self.rows, 
            columns: self.columns
        }
    }
    pub fn map(&self,f: impl Fn(f32) -> f32)->Tensor{
        let mut data=Vec::new();
        for value in self.data.iter(){
            data.push(f(*value));
        }
        Tensor {
            data:data, 
            rows: self.rows, 
            columns: self.columns 
        }
    }
}