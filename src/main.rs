use ndarray::{array, Array, Axis};
use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
use rand_isaac::isaac64::Isaac64Rng;

fn main() -> std::io::Result<()> {
  let seed = 42;
  let mut rng = Isaac64Rng::seed_from_u64(seed);
  
  let x = array![[0., 0.], [1., 0.], [0., 1.], [1., 1.]];
  let y = array![[0., 1., 1., 0.]].reversed_axes();
  
  let topology: Vec<usize> = vec![2, 3, 1];
  
  let mut data = vec![x.clone()];
  for ls in topology.iter().skip(1) {
    data.push(Array::zeros((x.shape()[0], *ls)))
  }

  let mut biases = vec![];
  let mut weights = vec![]; 
  for l in 1..topology.len() {
    weights.push(Array::random_using((topology[l-1], topology[l]), Uniform::new(0., 1.), &mut rng));
    biases.push(Array::random_using(topology[l], Uniform::new(0., 1.), &mut rng));
  }
  
  for i in 0..=5000 {
    // forward propagation
    for l in 0..topology.len()-1 {
      data[l+1] = (-data[l].dot(&weights[l]) + &biases[l]).mapv(logistic)
    }
    
    // gradient calculation
    let mut deltas = vec![(&y - &data[topology.len()-1]) * data[topology.len()-1].mapv(logistic_derivative)];
    for l in (1..topology.len()-1).rev() {
      deltas.push(deltas.last().unwrap().dot(&weights[l].t()) * data[l].mapv(logistic_derivative));
    }
    deltas.reverse();
    
    for l in 0..topology.len()-1 {
      weights[l] += &data[l].t().dot(&deltas[l]);
      biases[l] -= &deltas[l].sum_axis(Axis(0));
    }

    let loss = data[topology.len()-1].mapv(logistic_derivative).sum();
    
    if i % 100 == 0 {
      println!("Epoch: {} \nLoss: {:?}\n", i, loss);
      // println!("{:?}\n", data[topology.len()-1]);
    }
  }
  
  Ok(())
}

fn logistic(x: f64) -> f64 {
  1. / (1. + x.exp())
}

fn logistic_derivative(x: f64) -> f64 {
  x * (1. - x)
}