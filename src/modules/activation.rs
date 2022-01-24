use super::{ModuleCopy, Module, WeightCopyError};
use tch::{Tensor, nn};

/// The Rectified Linear Units activation function
#[derive(Debug)]
pub struct ReLU;

impl Module for ReLU {
    type Input = tch::Tensor;
    type Output = tch::Tensor;

    fn train(&mut self) {}

    fn eval(&mut self) {}

    fn forward(&mut self, input: Self::Input) -> Self::Output {
        input.relu()
    }
}

impl ModuleCopy for ReLU {
    fn copy(&mut self, _: &Self) -> Result<(), WeightCopyError> {Ok(())}
}

/// The Gausian Linear Units activation function
#[derive(Debug)]
pub struct GeLU;

impl Module for GeLU {
    type Input = tch::Tensor;
    type Output = tch::Tensor;

    fn train(&mut self) {}

    fn eval(&mut self) {}

    fn forward(&mut self, input: Self::Input) -> Self::Output {
        input.gelu()
    }
}

impl ModuleCopy for GeLU {
    fn copy(&mut self, _: &Self) -> Result<(), WeightCopyError> {Ok(())}
}

/// The sigmoid activation function
#[derive(Debug)]
pub struct Sigmoid;

impl Module for Sigmoid {
    type Input = tch::Tensor;
    type Output = tch::Tensor;

    fn train(&mut self) {}

    fn eval(&mut self) {}

    fn forward(&mut self, input: Self::Input) -> Self::Output {
        input.sigmoid()
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

impl Module for PReLU {
    type Input = tch::Tensor;
    type Output = tch::Tensor;

    fn train(&mut self) {}

    fn eval(&mut self) {}

    fn forward(&mut self, input: Self::Input) -> Self::Output {
        input.prelu(&self.weight)
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