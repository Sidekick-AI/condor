use crate::*;

pub struct Linear<const BATCH: u16, const IN: u16, const OUT: u16> {
    weight: Tensor2<IN, OUT>,
    bias: Tensor1<OUT>,
}

impl <const BATCH: u16, const IN: u16, const OUT: u16>Default for Linear<BATCH, IN, OUT> {
    fn default() -> Self {
        Self {
            weight: Tensor2::rand(),
            bias: Tensor1::rand()
        }
    }
}

impl <const BATCH: u16, const IN: u16, const OUT: u16>Module for Linear<BATCH, IN, OUT> {
    type Input = Tensor2<BATCH, IN>;
    type Output = Tensor2<BATCH, OUT>;

    fn forward(&self, input: Self::Input) -> Self::Output {
        (&input.matmul(&self.weight)) + &self.bias
    }
}