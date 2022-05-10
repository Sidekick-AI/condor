/// The NNModule trait
mod nnmodule;
pub use nnmodule::*;
/// Linear Layers
mod linear;
pub use linear::*;
/// Sequential Layer
#[macro_use] mod sequential;
pub use sequential::*;
/// Transformer Layers
mod transformer;
pub use transformer::*;
/// RNN Layers
mod rnn;
pub use rnn::*;
/// Activation Layers
mod activation;
pub use activation::*;
/// Other Layers
mod other;
pub use other::*;
/// Module tests
mod tests;