use tch::nn::{Embedding, LayerNorm, Module};
/// A trait for some basic functions a module should have
pub trait NNModule: Module {
    fn train(&mut self);
    fn eval(&mut self);
}

/// Implementation of NNModule for LayerNorm
impl NNModule for LayerNorm {
    fn train(&mut self) {}

    fn eval(&mut self) {}
}

/// Implementation of NNModule for Embedding
impl NNModule for Embedding {
    fn train(&mut self) {}

    fn eval(&mut self) {}
}

/// A trait to allow modules to copy weights
pub trait ModuleCopy {
    fn copy(&mut self, source: &Self);
}

impl ModuleCopy for LayerNorm {
    fn copy(&mut self, source: &Self) {
        tch::no_grad(|| {
            if let Some(bs_dest) = &mut self.bs {
                if let Some(bs_source) = &source.bs {
                    bs_dest.copy_(bs_source)
                }
            }
            if let Some(ws_dest) = &mut self.ws {
                if let Some(ws_source) = &source.ws {
                    ws_dest.copy_(ws_source)
                }
            }
        });
    }
}

impl ModuleCopy for Embedding {
    fn copy(&mut self, source: &Self) {
        tch::no_grad(|| {
            self.ws.copy_(&source.ws);
        });
    }
}