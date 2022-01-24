use super::{ModuleCopy, NNModule, WeightCopyError};
use tch::{Tensor, nn};

/// The Rectified Linear Units activation function
#[derive(Debug)]
pub struct ReLU;

impl NNModule for ReLU {
    fn train(&mut self) {}

    fn eval(&mut self) {}

    fn forward(&mut self, x: &tch::Tensor) -> tch::Tensor {
        x.relu()
    }
}

impl ModuleCopy for ReLU {
    fn copy(&mut self, _: &Self) -> Result<(), WeightCopyError> {Ok(())}
}

/// The Gausian Linear Units activation function
#[derive(Debug)]
pub struct GeLU;

impl NNModule for GeLU {
    fn train(&mut self) {}

    fn eval(&mut self) {}

    fn forward(&mut self, x: &tch::Tensor) -> tch::Tensor {
        x.gelu()
    }
}

impl ModuleCopy for GeLU {
    fn copy(&mut self, _: &Self) -> Result<(), WeightCopyError> {Ok(())}
}

/// The sigmoid activation function
#[derive(Debug)]
pub struct Sigmoid;

impl NNModule for Sigmoid {
    fn train(&mut self) {}

    fn eval(&mut self) {}

    fn forward(&mut self, x: &tch::Tensor) -> tch::Tensor {
        x.sigmoid()
    }
}

impl ModuleCopy for Sigmoid {
    fn copy(&mut self, _: &Self) -> Result<(), WeightCopyError> {Ok(())}
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

impl NNModule for PReLU {
    fn train(&mut self) {}

    fn eval(&mut self) {}

    fn forward(&mut self, x: &tch::Tensor) -> tch::Tensor {
        x.prelu(&self.weight)
    }
}

impl ModuleCopy for PReLU {
    fn copy(&mut self, source: &Self) -> Result<(), WeightCopyError> {
        if self.weight.size() != source.weight.size() {
            Err(WeightCopyError::SizeMismatch)
        } else {
            tch::no_grad(|| {
                self.weight.copy_(&source.weight);
            });
            Ok(())
        }
    }
}