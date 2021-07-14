/// Linear Layers
mod linear;
pub use linear::*;
/// Sequential Layer
#[macro_use] mod sequential;
pub use sequential::*;
/// Transformer Layers
mod transformer;
use tch::nn::Module;
pub use transformer::*;
/// RNN Layers
mod rnn;
pub use rnn::*;
/// Activation Layers
mod activation;
pub use activation::*;
/// Module tests
mod tests;

/// A trait for some basic functions a module should have
pub trait NNModule: Module {
    fn train(&mut self);
    fn eval(&mut self);
    fn count_parameters(&self) -> u64;
}