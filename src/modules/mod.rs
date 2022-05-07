mod activation;
mod linear;
mod sequential;
mod other;
mod rnn;

pub use activation::*;
pub use linear::*;
pub use sequential::*;
pub use other::*;
pub use rnn::*;

#[cfg(test)]
mod tests;

pub trait Module {
    type Input;
    type Output;

    fn train(&mut self) {}
    fn eval(&mut self) {}

    fn forward(&self, input: Self::Input) -> Self::Output;
}