use rand::RngExt;

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    data: Vec<f32>,
    rows: usize,
    columns: usize,
}

impl Tensor {
    pub fn fill(rows: usize, cols: usize, value: f32) -> Tensor {
        Tensor {
            data: vec![value; rows * cols],
            rows,
            columns: cols,
        }
    }
    pub fn from_vec(data: Vec<f32>, rows: usize, cols: usize) -> Tensor {
        assert_eq!(data.len(), rows * cols);
        Tensor {
            data,
            rows,
            columns: cols,
        }
    }
    pub fn identity(size: usize) -> Tensor {
        let mut identity_vec = Vec::new();
        for i in 0..size {
            for j in 0..size {
                if i == j {
                    identity_vec.push(1.0);
                } else {
                    identity_vec.push(0.0);
                }
            }
        }
        Tensor::from_vec(identity_vec, size, size)
    }
    pub fn rand(rows: usize, cols: usize) -> Tensor {
        let mut data = Vec::new();
        let mut rng = rand::rng();
        for _ in 0..(rows * cols) {
            data.push(rng.random_range(-1.0..1.0));
        }
        Tensor {
            data,
            rows,
            columns: cols,
        }
    }
    pub fn get(&self, i: usize, j: usize) -> f32 {
        assert!(i < self.rows, "row index out of bounds");
        assert!(j < self.columns, "column index out of bounds");
        self.data[i * self.columns + j]
    }
    pub fn set(&mut self, i: usize, j: usize, value: f32) {
        assert!(i < self.rows, "row index out of bounds");
        assert!(j < self.columns, "column index out of bounds");
        self.data[i * self.columns + j] = value
    }
    pub fn data(&self) -> &[f32] {
        &self.data
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.columns
    }
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.columns, other.rows);
        let mut data = Vec::new();
        for i in 0..self.rows {
            for j in 0..other.columns {
                let mut sum: f32 = 0.0;
                for k in 0..self.columns {
                    sum += self.data[k + self.columns * i] * other.data[k * other.columns + j];
                }
                data.push(sum);
            }
        }
        Tensor {
            data,
            rows: self.rows,
            columns: other.columns,
        }
    }
    pub fn transpose(&self) -> Tensor {
        let mut trans = Tensor::fill(self.columns, self.rows, 0.0);
        for i in 0..self.rows {
            for j in 0..self.columns {
                trans.set(j, i, self.get(i, j));
            }
        }
        trans
    }
    pub fn add_row(&self, row: &Tensor) -> Tensor {
        assert_eq!(row.rows, 1);
        assert_eq!(row.columns, self.columns);
        let mut data = Vec::new();
        for i in 0..self.rows {
            for j in 0..self.columns {
                data.push(self.data[i * self.columns + j] + row.data[j]);
            }
        }
        Tensor {
            data,
            rows: self.rows,
            columns: self.columns,
        }
    }
    pub fn sum_rows(&self) -> Tensor {
        let mut data = vec![0.0; self.columns];
        for i in 0..self.rows {
            for j in 0..self.columns {
                data[j] += self.data[i * self.columns + j];
            }
        }
        Tensor {
            data,
            rows: 1,
            columns: self.columns,
        }
    }
    pub fn map(&self, f: impl Fn(f32) -> f32) -> Tensor {
        let mut data = Vec::new();
        for value in self.data.iter() {
            data.push(f(*value));
        }
        Tensor {
            data,
            rows: self.rows,
            columns: self.columns,
        }
    }
    pub fn add(&self, other: &Tensor) -> Tensor {
        let mut data = Vec::new();
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.columns, other.columns);
        for i in 0..self.rows {
            for j in 0..self.columns {
                data.push(self.data[i * self.columns + j] + other.data[i * other.columns + j]);
            }
        }
        Tensor {
            data,
            rows: self.rows,
            columns: self.columns,
        }
    }
    pub fn sum(&self) -> f32 {
        let mut sum = 0.0;
        for value in self.data.iter() {
            sum += value;
        }
        sum
    }
    pub fn mul_elem(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.columns, other.columns);
        let mut data = Vec::new();
        for i in 0..self.rows {
            for j in 0..self.columns {
                data.push(self.data[i * self.columns + j] * other.data[i * other.columns + j]);
            }
        }
        Tensor {
            data,
            rows: self.rows,
            columns: self.columns,
        }
    }
    pub fn div_elem(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.columns, other.columns);
        let mut data = Vec::new();
        for i in 0..self.rows {
            for j in 0..self.columns {
                data.push(self.data[i * self.columns + j] / other.data[i * other.columns + j]);
            }
        }
        Tensor {
            data,
            rows: self.rows,
            columns: self.columns,
        }
    }
    pub fn scale(&self, v: f32) -> Tensor {
        let mut data = Vec::new();
        for i in 0..self.rows {
            for j in 0..self.columns {
                data.push(self.data[i * self.columns + j] * v);
            }
        }
        Tensor {
            data,
            rows: self.rows,
            columns: self.columns,
        }
    }
    pub fn softmax_rows(&self) -> Tensor {
        let mut out = self.clone();
        for i in 0..out.rows {
            let mut max = f32::NEG_INFINITY;
            for j in 0..out.columns {
                if out.get(i, j) > max {
                    max = out.get(i, j);
                }
            }
            let mut sum = 0.0;
            for j in 0..out.columns {
                out.data[i * out.columns + j] = (out.data[i * out.columns + j] - max).exp();
                sum += out.data[i * out.columns + j];
            }
            for j in 0..out.columns {
                out.data[i * out.columns + j] = out.data[i * out.columns + j] / sum;
            }
        }

        out
    }
    pub fn log_softmax_rows(&self) -> Tensor {
        let mut out = self.clone();
        for i in 0..out.rows {
            let mut max = f32::NEG_INFINITY;
            for j in 0..out.columns {
                if out.get(i, j) > max {
                    max = out.get(i, j);
                }
            }
            let mut sum = 0.0;
            for j in 0..out.columns {
                out.data[i * out.columns + j] = out.data[i * out.columns + j] - max;
                sum += out.data[i * out.columns + j].exp();
            }
            for j in 0..out.columns {
                out.data[i * out.columns + j] = out.data[i * out.columns + j] - sum.ln();
            }
        }

        out
    }
    pub fn argmax_rows(&self) -> Vec<usize> {
        let mut maximums = Vec::new();
        for i in 0..self.rows {
            let mut j_max = 0;
            for j in 0..self.columns {
                if self.get(i, j) > self.get(i, j_max) {
                    j_max = j;
                }
            }
            maximums.push(j_max);
        }
        maximums
    }
    pub fn select_rows(&self, indices: &[usize]) -> Tensor {
        let rows = indices.len();
        let mut data = Vec::new();
        for &i in indices.iter() {
            assert!(i < self.rows, "Index out of range");
            data.extend_from_slice(&self.data[(i * self.columns)..((i + 1) * self.columns)]);
        }
        Tensor::from_vec(data, rows, self.columns)
    }
}
