mod activation;
mod linear;
mod sequential;
mod other;

pub use activation::*;
pub use linear::*;
pub use sequential::*;
pub use other::*;

#[cfg(test)]
mod tests;

pub trait Module {
    type Input;
    type Output;

    fn train(&mut self) {}
    fn eval(&mut self) {}

    fn forward(&self, input: Self::Input) -> Self::Output;
}

#[cfg(test)]
mod module_tests {
    use crate::{Module, Tensor2};

    struct Linear<const BATCH: u16, const IN: u16, const OUT: u16> {}

    impl <const BATCH: u16, const IN: u16, const OUT: u16>Module for Linear<BATCH, IN, OUT> {
        type Input = Tensor2<BATCH, IN>;
        type Output = Tensor2<BATCH, OUT>;

        fn forward(&self, input: Self::Input) -> Self::Output {
            todo!()
        }
    }

    #[test]
    fn test_module() {

    }
}