use ndarray::{array, Array1, Array2, Ix2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use rand::{RngCore, Rng};

fn main() {
  let inputs = vec![
    array![0.0, 0.0],
    array![0.0, 1.0],
    array![1.0, 0.0],
    array![1.0, 1.0],
  ];

  let targets = vec![
    array![0.0],
    array![1.0],
    array![1.0],
    array![0.0],
  ];

  // let sigmoid = ActivationFunction {
  //   function: |x| 1.0 / (1.0 + (-x).exp()),
  //   derivative: |x| x * (1.0 - x)
  // };

  let relu = ActivationFunction {
    function: |x| x.max(0.0),
    derivative: |x| if x > &0.0 { 1.0 } else { 0.0 },
  };

  let mut rng = rand::thread_rng();
  let mut network = Network::new(vec![2, 3, 1], relu, 0.5, &mut rng);

  let start = std::time::Instant::now();
  network.train(inputs, targets, 10000);
  let end = std::time::Instant::now();

  println!("Time: {:?}s", (end - start).as_secs());

  println!("0 0 -> {}", network.feed_forward(array![0.0, 0.0]));
  println!("0 1 -> {}", network.feed_forward(array![0.0, 1.0]));
  println!("1 0 -> {}", network.feed_forward(array![1.0, 0.0]));
  println!("1 1 -> {}", network.feed_forward(array![1.0, 1.0]));
}

struct Network {
  layers: Vec<usize>, // Amount of neurons per layer
  weights: Vec<Array2<f32>>,
  biases: Vec<Array1<f32>>,
  activation: ActivationFunction,
  learning_rate: f32,
  data: Vec<Array1<f32>>,
}

impl Network {
  fn new(layers: Vec<usize>, activation: ActivationFunction, learning_rate: f32, rng: &mut impl Rng) -> Self {
    let mut weights = Vec::new();
    let mut biases = Vec::new();

    for i in 0..layers.len()-1 {
      let w = Array2::random_using((layers[i+1], layers[i]), Uniform::new(0., 1.), rng);
      let b = Array1::random_using(layers[i+1], Uniform::new(0., 1.), rng);
      weights.push(w);
      biases.push(b);
    }

    Self {
      layers,
      weights,
      biases,
      activation,
      learning_rate,
      data: Vec::new(),
    }
  }

  fn feed_forward(&mut self, inputs: Array1<f32>) -> Array1<f32> {
    assert!(self.layers[0] == inputs.len(), "Invalid Number of Inputs");

    let mut current = inputs;

    self.data = vec![current.clone()];

    for i in 0..self.layers.len() -1 {
      current = self.weights[i].dot(&current) + &self.biases[i].map(self.activation.function);
      self.data.push(current.clone());
    }

    current
  }

  fn back_propagate(&mut self, inputs: Array1<f32>, targets: Array1<f32>) {

    let mut errors: Vec<Array1<f32>> = vec![(targets - &inputs) * inputs.map(self.activation.derivative)];

    for i in (1..self.layers.len() - 1).rev() {
      errors.push(self.weights[i].t().dot(&errors[i-1]) * &self.data[i].map(self.activation.derivative));
    }
    errors.reverse();

    for l in 1..self.layers.len() { 
      self.weights[l-1] = &self.weights[l-1] - self.learning_rate * self.data[l].dot(&errors[l-1]);
      self.biases[l-1] = &self.biases[l-1] - self.learning_rate * &errors[l-1];
    }

  }

  fn train(&mut self, inputs: Vec<Array1<f32>>, targets: Vec<Array1<f32>>, epochs: u32) {
    for i in 1..=epochs {
      if epochs < 100 || i % (epochs / 100) == 0 {
        println!("Epoch {} of {}", i, epochs);
      }
      for j in 0..inputs.len() {
        let outputs = self.feed_forward(inputs[j].clone());

        println!("\nEpoch: {}, Sample: {}", i, j);
        println!("Outputs: {}", outputs);
        println!("Weights: {:?}", &self.weights);
        println!("Biases: {:?}", &self.biases);
        if f32::is_nan(outputs[0]) { panic!(); }

        self.back_propagate(outputs, targets[j].clone());
      }
    }
  }
}

struct ActivationFunction {
  function: fn(&f32) -> f32,
  derivative: fn(&f32) -> f32,
}