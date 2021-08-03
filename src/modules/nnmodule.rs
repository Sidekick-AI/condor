use tch::nn::Module;
/// A trait for some basic functions a module should have
pub trait NNModule: std::fmt::Debug + Send {
    fn train(&mut self);
    fn eval(&mut self);
    fn forward(&self, x: &tch::Tensor) -> tch::Tensor;
}

impl Module for dyn NNModule {
    fn forward(&self, xs: &tch::Tensor) -> tch::Tensor {
        self.forward(&xs)
    }
}


/// A trait to allow modules to copy weights
pub trait ModuleCopy {
    fn copy(&mut self, source: &Self) -> Result<(), WeightCopyError>;
}

/// An error type for copying weights
pub enum WeightCopyError {
    SizeMismatch,
    Other(String)
}