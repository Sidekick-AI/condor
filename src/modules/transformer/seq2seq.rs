use crate::modules::{Embedding, LayerNorm, Linear, ModuleCopy, Module, WeightCopyError, TransformerBlock, PositionalEncoding};
use tch::{nn, IndexOp, Kind, Tensor, Device};
use super::{LocalPositionalEncoding, SelfAttention};

