use super::engine::Value;

#[derive(Debug, Clone, Copy)]
pub enum Activation {
    Tanh,
    Sigmoid,
    Linear,
}

#[derive(Debug)]
pub struct Neuron {
    w: Vec<Value>,
    b: Value,
    act: Activation,
}

impl Neuron {
    pub fn new(n_inputs: usize, activation: Activation) -> Neuron {
        let mut w = Vec::new();
        for _ in 0..n_inputs {
            w.push(Value::new(rand::random_range(-1.0..1.0)));
        }

        let b = Value::new(rand::random_range(-1.0..1.0));
        Neuron {
            w,
            b,
            act: activation,
        }
    }
    pub fn forward(&self, x: &[Value]) -> Value {
        assert_eq!(x.len(), self.w.len());
        let mut sum = Value::new(0.0);
        for (input, weight) in x.iter().zip(self.w.iter()) {
            sum += weight.clone() * input.clone();
        }
        sum += self.b.clone();
        match self.act {
            Activation::Linear => sum,
            Activation::Sigmoid => sum.sigmoid(),
            Activation::Tanh => sum.tanh(),
        }
    }
    pub fn parameters(&self) -> Vec<Value> {
        let mut param = self.w.clone();
        param.push(self.b.clone());
        param
    }
}

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(n_inputs: usize, n_outputs: usize, activation: Activation) -> Layer {
        let mut neurons = Vec::new();
        for _ in 0..n_outputs {
            neurons.push(Neuron::new(n_inputs, activation));
        }
        Layer { neurons }
    }
    pub fn forward(&self, x: &[Value]) -> Vec<Value> {
        let mut output = Vec::new();
        for neuron in self.neurons.iter() {
            output.push(neuron.forward(x));
        }
        output
    }
    pub fn parameters(&self) -> Vec<Value> {
        let mut output = Vec::new();
        for neuron in self.neurons.iter() {
            for value in neuron.parameters() {
                output.push(value);
            }
        }
        output
    }
}

pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(
        n_inputs: usize,
        sizes: &[usize],
        hidden_act: Activation,
        output_act: Activation,
    ) -> MLP {
        let mut prev = n_inputs;
        let mut layers = Vec::new();
        for (i, size) in sizes.iter().enumerate() {
            if i == sizes.len() - 1 {
                layers.push(Layer::new(prev, *size, output_act));
            } else {
                layers.push(Layer::new(prev, *size, hidden_act));
            }
            prev = *size;
        }
        MLP { layers }
    }

    pub fn forward(&self, x: &[Value]) -> Vec<Value> {
        let mut out = x.to_vec();
        for layer in self.layers.iter() {
            out = layer.forward(&out);
        }
        out
    }
    pub fn parameters(&self) -> Vec<Value> {
        let mut params = Vec::new();
        for layer in self.layers.iter() {
            params.extend(layer.parameters());
        }
        params
    }
}
