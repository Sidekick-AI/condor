use super::{ModuleCopy, NNModule, WeightCopyError};
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
            bs: no_wd.randn("bias", &[out_dim], 0.0, 0.2),
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
}

impl NNModule for Linear {
    fn train(&mut self) {}

    fn eval(&mut self) {}

    fn forward(&self, x: &tch::Tensor) -> tch::Tensor {
        x.matmul(&self.ws.tr()) + &self.bs
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