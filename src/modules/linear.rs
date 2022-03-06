use super::{ModuleCopy, Module};
use crate::{Tensor, DimType};

#[derive(Debug)]
pub struct Linear<const IN: DimType, const OUT: DimType> {
    pub ws: Tensor<IN, OUT>,
    pub bs: Tensor<OUT>,
}

const fn static_dim(n: usize) -> DimType {
    DimType::Static(n)
}

impl <const IN: DimType, const OUT: DimType>Clone for Linear<IN, OUT> {
    fn clone(&self) -> Self {
        Linear {
            ws: self.ws.clone(),
            bs: self.bs.clone()
        }
    }
}

impl <const IN: DimType, const OUT: DimType>Linear<IN, OUT> {
    pub fn new(vs: &tch::nn::Path, in_dim: i64, out_dim: i64) -> Self {
        let wd = vs.set_group(1);
        let no_wd = vs.set_group(0);
        Linear {
            ws: Tensor::from_tch(wd.randn("weight", &[out_dim, in_dim], 0.0, 0.02)),
            bs: Tensor::from_tch(no_wd.randn("bias", &[out_dim], 0.0, 0.2)),
        }
    }

    pub fn no_bias(vs: tch::nn::Path, in_dim: i64, out_dim: i64) -> Self {
        let wd = vs.set_group(1);
        let no_wd = vs.set_group(0);
        Linear {
            ws: Tensor::from_tch(wd.randn("weight", &[out_dim, in_dim], 0.0, 0.02)),
            bs: Tensor::from_tch(no_wd.zeros_no_train("bias", &[out_dim])),
        }
    }
}

impl <const IN: DimType, const OUT: DimType>Module for Linear<IN, OUT> {
    type Input = Tensor<{DimType::Dynamic}, IN>;
    type Output = Tensor<{DimType::Dynamic}, OUT>;

    fn train(&mut self) {}

    fn eval(&mut self) {}

    fn forward(&mut self, input: Self::Input) -> Self::Output {
        input.matmul(&self.ws.tr()) + &self.bs
    }
}

impl <const IN: usize, const OUT: usize>ModuleCopy for Linear<IN, OUT> {
    fn copy(&mut self, source: &Self) {
        tch::no_grad(|| {
            self.ws.copy_(&source.ws);
            self.bs.copy_(&source.bs);
        });
    }
}