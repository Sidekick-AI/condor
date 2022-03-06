/// A trait for some basic functions a module should have
pub trait Module: std::fmt::Debug + Send {
    type Input;
    type Output;

    fn train(&mut self);
    fn eval(&mut self);
    fn forward(&mut self, input: Self::Input) -> Self::Output;
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