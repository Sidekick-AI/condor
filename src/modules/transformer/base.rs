use crate::modules::{LayerNorm, Linear, ModuleCopy, NNModule, WeightCopyError};
use tch::{nn, Device, IndexOp, Kind, Tensor};

/// Different types of positional encoding for Transformers
#[derive(PartialEq, Eq)]
pub enum PositionalEncoding {
    Learned,
    Sinusoidal,
    Rotary,
}

/// An enum for positional encoding which conains a tensor (only used internally)
#[derive(Debug, PartialEq)]
pub(super) enum LocalPositionalEncoding {
    Learned(Tensor),
    Sinusoidal(Tensor),
    Rotary{inv_freq: Tensor, seq_len_cached: usize, cos_cached: Tensor, sin_cached: Tensor},
}

/// The most basic dot-product self attention with an optional causal mask
#[derive(Debug)]
pub(crate) struct SelfAttention {
    n_head: i64,
    n_embd: i64,
    dropout: f64,
    key: Linear,
    query: Linear,
    value: Linear,
    proj: Linear,
    train: bool,
    causal_mask: bool
}

impl SelfAttention {
    pub fn new(p: &nn::Path, n_embd: i64, n_head: i64, dropout: f64, causal_mask: bool) -> Self {
        SelfAttention {
            n_embd,
            n_head,
            dropout,
            key: Linear::new(&(p / "key"), n_embd, n_embd),
            query: Linear::new(&(p / "query"), n_embd, n_embd),
            value: Linear::new(&(p / "value"), n_embd, n_embd),
            proj: Linear::new(&(p / "proj"), n_embd, n_embd),
            train: true,
            causal_mask,
        }
    }

    fn generate_mask(size: i64, device: Device) -> Tensor{
        Tensor::ones(&[size, size], (Kind::Float, device)).tril(0).view([1, 1, size, size])
    }
}

impl Clone for SelfAttention {
    fn clone(&self) -> Self {
        SelfAttention {
            n_head: self.n_head,
            n_embd: self.n_embd,
            dropout: self.dropout,
            key: self.key.clone(),
            query: self.query.clone(),
            value: self.value.clone(),
            proj: self.proj.clone(),
            train: self.train,
            causal_mask: self.causal_mask,
        }
    }
}

impl NNModule for SelfAttention {
    fn train(&mut self) {
        self.train = true;
    }

    fn eval(&mut self) {
        self.train = false;
    }

    fn forward(&mut self, x: &tch::Tensor) -> tch::Tensor {
        let (sz_b, sz_t, sz_c) = x.size3().unwrap();
        let sizes = [sz_b, sz_t, self.n_head, sz_c / self.n_head];
        let k = self.key.forward(x).view(sizes).transpose(1, 2);
        let q = self.query.forward(x).view(sizes).transpose(1, 2);
        let v = self.value.forward(x).view(sizes).transpose(1, 2);
        let mut att = q.matmul(&k.transpose(-2, -1)) * (1.0 / f64::sqrt(sizes[3] as f64));
        if self.causal_mask {
            let mask = SelfAttention::generate_mask(sz_t, x.device());
            att = att.masked_fill(
                &mask.i((.., .., ..sz_t, ..sz_t)).eq(0.),
                std::f64::NEG_INFINITY,
            );
        }
        att = att.softmax(-1, Kind::Float).dropout(self.dropout, self.train);
        let ys = att
            .matmul(&v)
            .transpose(1, 2)
            .contiguous()
            .view([sz_b, sz_t, sz_c]);
        self.proj.forward(&ys).dropout(self.dropout, self.train)
    }
}

impl ModuleCopy for SelfAttention {
    fn copy(&mut self, source: &Self) -> Result<(), WeightCopyError> {
        self.key.copy(&source.key)?;
        self.query.copy(&source.query)?;
        self.value.copy(&source.value)?;
        self.proj.copy(&source.proj)?;
        Ok(())
    }
}

/// A basic transformer encoder block
#[derive(Debug)]
pub struct TransformerBlock {
    norm1: LayerNorm,
    norm2: LayerNorm,
    attn: SelfAttention,
    linear1: Linear,
    linear2: Linear,
    dropout: f64,
    train: bool,
}

impl TransformerBlock {
    pub fn new(p: &nn::Path, n_embd: i64, n_head: i64, dropout: f64, causal_mask: bool) -> Self {
        assert!(n_embd % n_head == 0, "Embedding size ({}) must be divisible by number of heads ({})!", n_embd, n_head);
        TransformerBlock {
            norm1: LayerNorm::new(p / "ln1", vec![n_embd]),
            norm2: LayerNorm::new(p / "ln2", vec![n_embd]),
            attn: SelfAttention::new(&(p / "attn"), n_embd, n_head, dropout, causal_mask),
            linear1: Linear::new(&(p / "lin1"), n_embd, 2 * n_embd),
            linear2: Linear::new(&(p / "lin2"), 2 * n_embd, n_embd),
            dropout,
            train: true
        }
    }
}

impl NNModule for TransformerBlock {
    fn train(&mut self) {
        self.attn.train();
        self.train = true;
    }

    fn eval(&mut self) {
        self.attn.eval();
        self.train = false;
    }

    fn forward(&mut self, x: &tch::Tensor) -> tch::Tensor {
        let x = x + self.attn.forward(&self.norm1.forward(x));
        let ys = self.linear2.forward(
                &self.linear1.forward(
                    &self.norm2.forward(&x)
                ).gelu()
            ).dropout(self.dropout, self.train);
        x + ys
    }
}

impl ModuleCopy for TransformerBlock {
    fn copy(&mut self, source: &Self) -> Result<(), WeightCopyError> {
        self.attn.copy(&source.attn)?;
        self.norm1.copy(&source.norm1)?;
        self.norm2.copy(&source.norm2)?;
        self.linear1.copy(&source.linear1)?;
        self.linear2.copy(&source.linear2)
    }
}