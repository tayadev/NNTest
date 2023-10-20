use rand::prelude::*;
use ndarray::array;
use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

fn main() {
  
  let inputs = array![
    [0.0, 0.0, 1.0, 1.0],
    [0.0, 1.0, 0.0, 1.0]
  ];
  
  let expected_outputs = array![[0,1,1,0]];
  
  let input_neuron_count = 2;
  let hidden_neuron_count = 2;
  let output_neuron_count = 1;
  
  let sample_count = inputs.shape()[0];
  
  let learning_rate = 0.1;
  
  // Create seeded random number source
  let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(10);
  
  // Initialize weights and biases
  let w1 = Array2::random_using((hidden_neuron_count, input_neuron_count), Uniform::new(0., 1.), &mut rng);
  let b1: Array1<f32> = (0..hidden_neuron_count).map(|_| rng.gen()).collect();
  let w2 = Array2::random_using((output_neuron_count, hidden_neuron_count), Uniform::new(0., 1.), &mut rng);
  let b2: Array1<f32> = (0..output_neuron_count).map(|_| rng.gen()).collect();

  // let outputs = feed_forward(w1, b1, w2, b2, array![0.0, 1.0]);

  let net = Network {
    weights: vec![w1, w2],
    biases: vec![b1, b2],
    activation_function: Box::new(&sigmoid)
  };

  let input = array![0.0, 1.0];

  println!("Inputs: {i}", i = input.shape()[0]);
  println!("{i}", i = "I ".repeat(input.shape()[0]));
  println!("{l1}", l1 = "O ".repeat(net.weights[0].shape()[0]));
  println!("{l2}", l2 = "O ".repeat(net.weights[1].shape()[0]));

  let outputs = net.feed_forward(input);

  println!("{outputs}");
}

fn sigmoid(x: f32) -> f32 {
  1.0 / (1.0 + (-x).exp())
}

struct Network {
  weights: Vec<Array2<f32>>,
  biases: Vec<Array1<f32>>,
  activation_function: Box<dyn Fn(f32) -> f32>,
}

impl Network {
  fn feed_forward(&self, input: Array1<f32>) -> Array1<f32> {
    let mut state = input;
    for l in 0..self.weights.len() {
      state = (self.weights[l].dot(&state) + &self.biases[l]).map(|z| self.activation_function.as_ref()(*z));
    }
    state
  }
}