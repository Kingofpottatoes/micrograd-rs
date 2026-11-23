use crate::engine::Value;

#[derive(Debug)]
pub struct Neuron {
    w: Vec<Value>,
    b: Value,
}

impl Neuron {
    pub fn new(n_inputs: usize) -> Neuron {
        let mut w = Vec::new();
        for _ in 0..n_inputs {
            w.push(Value::new(rand::random_range(-1.0..1.0)));
        }

        let b = Value::new(rand::random_range(-1.0..1.0));
        Neuron { w: w, b: b }
    }
    pub fn forward(&self, x: &[Value]) -> Value {
        assert_eq!(x.len(), self.w.len());
        let mut sum = Value::new(0.0);
        for (input, weight) in x.iter().zip(self.w.iter()) {
            sum += weight.clone() * input.clone();
        }
        sum += self.b.clone();
        return sum.tanh();
    }
    pub fn parameters(&self) -> Vec<Value> {
        let mut param = self.w.clone();
        param.push(self.b.clone());
        return param;
    }
}

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(n_inputs: usize, n_outputs: usize) -> Layer {
        let mut neurons = Vec::new();
        for _ in 0..n_outputs {
            neurons.push(Neuron::new(n_inputs));
        }
        Layer { neurons }
    }
    pub fn forward(&self, x: &[Value]) -> Vec<Value> {
        let mut output = Vec::new();
        for neuron in self.neurons.iter() {
            output.push(neuron.forward(x));
        }
        return output;
    }
    pub fn parameters(&self) -> Vec<Value> {
        let mut output = Vec::new();
        for neuron in self.neurons.iter() {
            for value in neuron.parameters() {
                output.push(value);
            }
        }
        return output;
    }
}

pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(n_inputs: usize, sizes: &[usize]) -> MLP {
        let mut prev = n_inputs;
        let mut layers = Vec::new();
        for size in sizes {
            layers.push(Layer::new(prev, *size));
            prev = *size;
        }
        return MLP { layers };
    }

    pub fn forward(&self, x: &[Value]) -> Vec<Value> {
        let mut out = x.to_vec();
        for layer in self.layers.iter() {
            out = layer.forward(&out);
        }
        return out;
    }
    pub fn parameters(&self) -> Vec<Value> {
        let mut params = Vec::new();
        for layer in self.layers.iter() {
            params.extend(layer.parameters());
        }
        return params;
    }
}
