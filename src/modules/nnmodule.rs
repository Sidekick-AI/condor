use tch::nn::Module;
/// A trait for some basic functions a module should have
pub trait NNModule: std::fmt::Debug + Send {
    fn train(&mut self);
    fn eval(&mut self);
    fn forward<I, O>(&mut self, x: I) -> O;
}


/// A trait to allow modules to copy weights
pub trait ModuleCopy {
    fn copy(&mut self, source: &Self) -> Result<(), WeightCopyError>;
}

/// An error type for copying weights
#[derive(Debug)]
pub enum WeightCopyError {
    SizeMismatch,
    Other(String)
}