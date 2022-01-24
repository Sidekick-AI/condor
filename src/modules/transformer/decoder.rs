use crate::modules::{Embedding, LayerNorm, Linear, ModuleCopy, NNModule, WeightCopyError, TransformerBlock, PositionalEncoding};
use tch::{nn, IndexOp, Kind, Tensor};
use super::LocalPositionalEncoding;

/// A simple autoregressive transformer decoder
#[derive(Debug)]
pub struct TransformerDecoder {
    token_embedding: Embedding,
    position_embedding: LocalPositionalEncoding,
    layernorm: LayerNorm,
    blocks: Vec<TransformerBlock>,
    dropout: f64,
    pub(super) n_embed: i64,
    train: bool,
}