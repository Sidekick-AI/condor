use std::borrow::Borrow;
use tch::{Tensor, nn::{self, EmbeddingConfig, LayerNormConfig}};
use super::{NNModule, ModuleCopy};

/// A layer-normalization layer.
#[derive(Debug)]
pub struct LayerNorm {
    config: LayerNormConfig,
    pub ws: Option<Tensor>,
    pub bs: Option<Tensor>,
    pub normalized_shape: Vec<i64>,
}

impl NNModule for LayerNorm {
    fn train(&mut self) {}

    fn eval(&mut self) {}

    fn forward(&self, x: &tch::Tensor) -> tch::Tensor {
        Tensor::layer_norm(
            x,
            self.normalized_shape.as_slice(),
            self.ws.as_ref(),
            self.bs.as_ref(),
            self.config.eps,
            self.config.cudnn_enabled,
        )
    }
}

impl LayerNorm {
    pub fn new<'a, T: Borrow<nn::Path<'a>>>(vs: T, normalized_shape: Vec<i64>) -> Self {
        let vs = vs.borrow();

        let config = LayerNormConfig::default();
        let (ws, bs) = if config.elementwise_affine {
            let ws = vs.var("weight", normalized_shape.as_slice(), config.ws_init);
            let bs = vs.var("bias", normalized_shape.as_slice(), config.bs_init);
            (Some(ws), Some(bs))
        } else {
            (None, None)
        };

        LayerNorm {
            config,
            ws,
            bs,
            normalized_shape,
        }
    }
}

/// An embedding layer.
///
/// An embedding layer acts as a simple lookup table that stores embeddings.
/// This is commonly used to store word embeddings.
#[derive(Debug)]
pub struct Embedding {
    pub ws: Tensor,
    config: EmbeddingConfig,
}

impl NNModule for Embedding {
    fn train(&mut self) {}

    fn eval(&mut self) {}

    fn forward(&self, x: &tch::Tensor) -> tch::Tensor {
        Tensor::embedding(
            &self.ws,
            x,
            self.config.padding_idx,
            self.config.scale_grad_by_freq,
            self.config.sparse,
        )
    }
}

impl Embedding {
    pub fn new<'a, T: Borrow<nn::Path<'a>>>(vs: T, num_embeddings: i64, embedding_dim: i64) -> Self {
        let vs = vs.borrow();
        let config = EmbeddingConfig::default();
        Embedding {
            ws: vs.var("weight", &[num_embeddings, embedding_dim], config.ws_init),
            config,
        }
    }
}

/// A layer defined by a closure
pub struct Func<'a> {
    f: Box<dyn 'a + Fn(&Tensor, bool) -> Tensor + Send>,
    train: bool
}

impl<'a> std::fmt::Debug for Func<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "func")
    }
}

pub fn func<'a, F>(f: F) -> Func<'a>
where
    F: 'a + Fn(&Tensor, bool) -> Tensor + Send,
{
    Func { f: Box::new(f), train: true }
}

impl<'a> NNModule for Func<'a> {
    fn train(&mut self) {
        self.train = true;
    }

    fn eval(&mut self) {
        self.train = false;
    }

    fn forward(&self, x: &tch::Tensor) -> tch::Tensor {
        (*self.f)(x, self.train)
    }
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