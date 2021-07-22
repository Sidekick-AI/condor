use super::{ModuleCopy, NNModule};
use tch::{Tensor, nn};

#[derive(Debug)]
pub struct Linear {
    pub ws: Tensor,
    pub bs: Tensor,
}

impl nn::Module for Linear {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.matmul(&self.ws.tr()) + &self.bs
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
}

impl ModuleCopy for Linear {
    fn copy(&mut self, source: &Self) {
        tch::no_grad(|| {
            self.ws.copy_(&source.ws);
            self.bs.copy_(&source.bs);
        });
    }
}