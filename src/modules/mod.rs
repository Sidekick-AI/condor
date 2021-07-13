/// Linear Layers
mod linear;
pub use linear::*;
/// Sequential Layer
mod sequential;
pub use sequential::*;
/// Transformer Layers
mod transformer;
pub use transformer::*;
/// RNN Layers
mod rnn;
pub use rnn::*;
/// Module tests
mod tests;

/// A trait for some basic functions a module should have
pub trait NNModule {
    fn train(&mut self);
    fn eval(&mut self);
    fn count_parameters(&self) -> u64;
}