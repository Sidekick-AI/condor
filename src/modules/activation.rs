use super::NNModule;
use tch::{Tensor, nn};

/// The Rectified Linear Units activation function
#[derive(Debug)]
pub struct ReLU;

impl nn::Module for ReLU {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.relu()
    }
}

impl NNModule for ReLU {
    fn train(&mut self) {}

    fn eval(&mut self) {}

    fn count_parameters(&self) -> u64 {
        0
    }
}

/// The Gausian Linear Units activation function
#[derive(Debug)]
pub struct GeLU;

impl nn::Module for GeLU {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.gelu()
    }
}

impl NNModule for GeLU {
    fn train(&mut self) {}

    fn eval(&mut self) {}

    fn count_parameters(&self) -> u64 {
        0
    }
}

/// The sigmoid activation function
#[derive(Debug)]
pub struct Sigmoid;

impl nn::Module for Sigmoid {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.sigmoid()
    }
}

impl NNModule for Sigmoid {
    fn train(&mut self) {}

    fn eval(&mut self) {}

    fn count_parameters(&self) -> u64 {
        0
    }
}

/// The Parameterized linear units activation function
#[derive(Debug)]
pub struct PReLU {
    weight: Tensor,
}

impl PReLU {
    pub fn new(vs: nn::Path) -> Self {
        PReLU {
            weight: vs.set_group(1).randn("weight", &[1], 0.0, 0.02)
        }
    }
}

impl nn::Module for PReLU {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.prelu(&self.weight)
    }
}

impl NNModule for PReLU {
    fn train(&mut self) {}

    fn eval(&mut self) {}

    fn count_parameters(&self) -> u64 {
        1
    }
}