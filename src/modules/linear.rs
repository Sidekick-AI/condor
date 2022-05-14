use super::{ModuleCopy, Module, WeightCopyError};
use tch::{Tensor, nn};

#[derive(Debug)]
pub struct Linear {
    pub ws: Tensor,
    pub bs: Tensor,
}

impl Clone for Linear {
    fn clone(&self) -> Self {
        Linear {
            ws: self.ws.copy(),
            bs: self.bs.copy()
        }
    }
}

impl Linear {
    pub fn new(vs: &nn::Path, in_dim: i64, out_dim: i64) -> Self {
        let wd = vs.set_group(1);
        let no_wd = vs.set_group(0);
        Linear {
            ws: wd.randn("weight", &[out_dim, in_dim], 0.0, 0.02),
            bs: no_wd.zeros("bias", &[out_dim]),
        }
    }

    pub fn no_bias(vs: nn::Path, in_dim: i64, out_dim: i64) -> Self {
        let wd = vs.set_group(1);
        let no_wd = vs.set_group(0);
        Linear {
            ws: wd.randn("weight", &[out_dim, in_dim], 0.0, 0.02),
            bs: no_wd.zeros_no_train("bias", &[out_dim]),
        }
    }

    // Init weights with "fan-in" variance scaling
    pub fn variance_init(vs: &nn::Path, in_dim: i64, out_dim: i64) -> Self {
        let wd = vs.set_group(1);
        let no_wd = vs.set_group(0);
        Linear {
            ws: wd.randn("weight", &[out_dim, in_dim], 0.0, 1. / (in_dim as f64).sqrt()),
            bs: no_wd.randn("bias", &[out_dim], 0.0, 1. / (out_dim as f64).sqrt()),
        }
    }
}

impl Module for Linear {
    type Input = tch::Tensor;
    type Output = tch::Tensor;

    fn train(&mut self) {}

    fn eval(&mut self) {}

    fn forward(&mut self, input: Self::Input) -> Self::Output {
        input.matmul(&self.ws.tr()) + &self.bs
    }
}

impl ModuleCopy for Linear {
    fn copy(&mut self, source: &Self) -> Result<(), WeightCopyError> {
        if self.ws.size() != source.ws.size() || self.bs.size() != source.bs.size() {
            Err(WeightCopyError::SizeMismatch)
        } else {
            tch::no_grad(|| {
                self.ws.copy_(&source.ws);
                self.bs.copy_(&source.bs);
            });
            Ok(())
        }
    }
}