use super::{ModuleCopy, NNModule};
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
}

impl ModuleCopy for ReLU {
    fn copy(&mut self, _: &Self) {}
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
}

impl ModuleCopy for GeLU {
    fn copy(&mut self, _: &Self) {}
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
}

impl ModuleCopy for PReLU {
    fn copy(&mut self, source: &Self) {
        self.weight.copy_(&source.weight);
    }
}