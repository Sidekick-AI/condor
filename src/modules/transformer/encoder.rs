use crate::modules::{Embedding, LayerNorm, Linear, ModuleCopy, Module, WeightCopyError, TransformerBlock, PositionalEncoding};
use tch::{nn, IndexOp, Kind, Tensor};
use super::LocalPositionalEncoding;

/// A basic transformer encoder stack using learned embeddings
#[derive(Debug)]
pub struct TransformerEncoder {
    token_embedding: Embedding,
    position_embedding: LocalPositionalEncoding,
    layernorm: LayerNorm,
    blocks: Vec<TransformerBlock>,
    dropout: f64,
    pub(super) n_embed: i64,
    train: bool,
}

pub struct TransformerEncoderProps<'a> {
    pub p: &'a nn::Path<'a>, 
    pub n_embd: i64, 
    pub n_head: i64, 
    pub n_layers: i64, 
    pub vocab_size: i64, 
    pub positional_encoding: PositionalEncoding, 
    pub max_len: i64, 
    pub dropout: f64, 
    pub causal_mask: bool
}

impl TransformerEncoder {
    #[allow(clippy::too_many_arguments)]
    pub fn new(props: TransformerEncoderProps) -> Self {
        TransformerEncoder {
            token_embedding: Embedding::new(
                props.p / "tok_emb",
                props.vocab_size,
                props.n_embd,
            ),
            position_embedding: match props.positional_encoding {
                PositionalEncoding::Learned => LocalPositionalEncoding::Learned(props.p.randn("pos_emb", &[1, props.max_len, props.n_embd], 0., 0.5)),
                PositionalEncoding::Sinusoidal => {
                    // Build the sinusoidal vector (This is based on an online implementation here: https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec#d554
                    let mut pe = vec![vec![0.; props.n_embd as usize]; props.max_len as usize];
                    #[allow(clippy::needless_range_loop)]
                    for pos in 0..props.max_len as usize {
                        for i in 0..props.n_embd as usize {
                            pe[pos][i] = (pos as f64 / f64::powf(10000., (2.*i as f64) / props.n_embd as f64)).sin();
                        }
                    }
                    LocalPositionalEncoding::Sinusoidal(Tensor::of_slice2(&pe).to_kind(Kind::Float).to(props.p.device())) // Doesn't need to be a variable, we aren't tracking it's gradients
                },
                PositionalEncoding::Rotary => {
                    // Build rotary vector
                    todo!()
                }
            },
            layernorm: LayerNorm::new(props.p / "ln_f", vec![props.n_embd]),
            blocks: {
                //let p = &p.set_group(0);
                let mut blocks = Vec::new();
                for block_idx in 0..props.n_layers {
                    blocks.push(TransformerBlock::new(&(props.p / block_idx), props.n_embd, props.n_head, props.dropout, props.causal_mask));
                }
                blocks
            },
            dropout: props.dropout,
            n_embed: props.n_embd,
            train: true,
        }
    }

    pub fn forward_no_embed(&mut self, xs: &Tensor) -> Tensor {
        // xs shape: (batch size, seq len, n_embd)
        let (batch_size, sz_t, _) = xs.size3().unwrap();
        let pos_emb = match &mut self.position_embedding {
            LocalPositionalEncoding::Learned(l) => l.i((.., ..sz_t, ..)).repeat(&[batch_size, 1, 1]),
            LocalPositionalEncoding::Sinusoidal(pe) => {
                pe.i(..sz_t).repeat(&[batch_size, 1, 1])
            },
            LocalPositionalEncoding::Rotary { inv_freq, seq_len_cached, cos_cached, sin_cached } => {
                todo!()
            }
        };
        let mut x = (xs + pos_emb)
            .dropout(self.dropout, self.train);
        // Run through transformer blocks
        x = self.blocks[0].forward(x);
        x = self.blocks
            .iter_mut()
            .skip(1)
            .fold(x, |x, layer| layer.forward(x));
        // Return first token
        self.layernorm.forward(x)
        // output shape: (batch size, n_embd)
    }
}

impl Module for TransformerEncoder {
    type Input = tch::Tensor;
    type Output = tch::Tensor;

    fn train(&mut self) {
        for block in &mut self.blocks {
            block.train();
        }
        self.train = true;
    }

    fn eval(&mut self) {
        for block in &mut self.blocks {
            block.eval();
        }
        self.train = false;
    }

    fn forward(&mut self, input: Self::Input) -> Self::Output {
        // x shape: (batch size, seq len)
        let (batch_size, sz_t) = input.size2().unwrap();
        // Run through embeddings
        let tok_emb = self.token_embedding.forward(input);
        let pos_emb = match &mut self.position_embedding {
            LocalPositionalEncoding::Learned(l) => l.i((.., ..sz_t, ..)).repeat(&[batch_size, 1, 1]),
            LocalPositionalEncoding::Sinusoidal(pe) => {
                pe.i(..sz_t).repeat(&[batch_size, 1, 1])
            },
            LocalPositionalEncoding::Rotary{inv_freq, seq_len_cached, cos_cached, sin_cached} => {
                todo!()
            }
        };
        let x = (tok_emb + pos_emb)
            .dropout(self.dropout, self.train);
        // Run through transformer blocks
        let x = self.blocks[0].forward(x);
        let x = self.blocks
            .iter_mut()
            .skip(1)
            .fold(x, |x, layer| layer.forward(x));
        // Return first token
        self.layernorm.forward(x)
        // output shape: (batch size, n_embd)
    }
}

impl ModuleCopy for TransformerEncoder {
    fn copy(&mut self, source: &Self) -> Result<(), WeightCopyError> {
        assert_eq!(self.blocks.len(), source.blocks.len());
        self.token_embedding.copy(&source.token_embedding)?;

        // Copy position embedding
        match &source.position_embedding {
            LocalPositionalEncoding::Learned(s) => {
                if let LocalPositionalEncoding::Learned(t) = &mut self.position_embedding {
                    if s.size() != t.size() {
                        return Err(WeightCopyError::SizeMismatch);
                    }
                    tch::no_grad(|| {
                        t.copy_(s);
                    });
                } else {return Err(WeightCopyError::Other("Positional Encodings are of wrong type!".to_string()));}
            },
            LocalPositionalEncoding::Sinusoidal(s) => {
                if let LocalPositionalEncoding::Sinusoidal(t) = &mut self.position_embedding {
                    if s.size() != t.size() {
                        return Err(WeightCopyError::SizeMismatch);
                    }
                    tch::no_grad(|| {
                        t.copy_(s);
                    });
                } else {return Err(WeightCopyError::Other("Positional Encodings are of wrong type!".to_string()));}
            },
            LocalPositionalEncoding::Rotary{inv_freq, seq_len_cached, cos_cached, sin_cached} => {
                if let LocalPositionalEncoding::Rotary{inv_freq: t_inv_freq, seq_len_cached: t_seq_len_cached, cos_cached: t_cos_cached, sin_cached: t_sin_cached} = &mut self.position_embedding {
                    if t_inv_freq.size() != inv_freq.size() || t_cos_cached.size() != cos_cached.size() || t_sin_cached.size() != sin_cached.size() {
                        return Err(WeightCopyError::SizeMismatch);
                    }
                    *t_seq_len_cached = *seq_len_cached;
                    tch::no_grad(|| {
                        t_inv_freq.copy_(inv_freq);
                        t_cos_cached.copy_(cos_cached);
                        t_sin_cached.copy_(sin_cached);
                    });
                }
            }
        }

        self.layernorm.copy(&source.layernorm)?;
        for i in 0..self.blocks.len() {
            self.blocks[i].copy(&source.blocks[i])?;
        }
        Ok(())
    }
}


/// A transformer encoder that aggregates a sequence into a single vector
#[derive(Debug)]
pub struct TransformerAggregator {
    pub encoder: TransformerEncoder,
    pub aggregation_embedding: Tensor,
    pub head: Linear
}

pub struct TransformerAggregatorProps<'a> {
    pub p: &'a nn::Path<'a>, 
    pub n_embd: i64, 
    pub n_head: i64,
    pub n_layers: i64, 
    pub aggregation_size: i64,
    pub vocab_size: i64, 
    pub positional_encoding: PositionalEncoding, 
    pub max_len: i64, 
    pub dropout: f64
}

impl TransformerAggregator {
    #[allow(clippy::too_many_arguments)]
    pub fn new(props: TransformerAggregatorProps) -> Self {
        TransformerAggregator {
            encoder: TransformerEncoder::new(TransformerEncoderProps{
                p: &(props.p / "encoder"), 
                n_embd: props.n_embd, 
                n_head: props.n_head, 
                n_layers: props.n_layers, 
                vocab_size: props.vocab_size, 
                positional_encoding: props.positional_encoding, 
                max_len: props.max_len, 
                dropout: props.dropout, 
                causal_mask: false
            }),
            head: Linear::new(&(props.p / "aggregation_head"), props.n_embd, props.aggregation_size),
            aggregation_embedding: props.p.randn("aggregation_vector", &[props.n_embd], 0.0, 0.2),
        }
    }

    pub fn from_encoder(p: &nn::Path, encoder: TransformerEncoder, aggregation_size: i64) -> Self {
        TransformerAggregator {
            aggregation_embedding: p.randn("aggregation_vector", &[encoder.n_embed], 0.0, 0.2),
            head: Linear::new(&(p / "aggregation_head"), encoder.n_embed, aggregation_size),
            encoder
        }
    }
}

impl Module for TransformerAggregator {
    type Input = tch::Tensor;
    type Output = tch::Tensor;

    fn train(&mut self) {
        self.encoder.train();
    }

    fn eval(&mut self) {
        self.encoder.eval();
    }

    fn forward(&mut self, x: Self::Input) -> Self::Output {
        // xs shape: (batch size, seq len)
        let batch_size = x.size()[0];
        // Embed and append aggregation embedding to beginning
        let mut xs = tch::Tensor::cat(&[
            &self.aggregation_embedding.unsqueeze(0).unsqueeze(0).repeat(&[batch_size, 1, 1]), 
            &self.encoder.token_embedding.forward(x)
        ], 1);
        // Run through encoder
        xs = self.encoder.forward_no_embed(&xs);
        // Return first token
        self.head.forward(xs.i((.., 0, ..)).squeeze_dim(1))
        // output shape: (batch size, n_embd)
    }
}

impl ModuleCopy for TransformerAggregator {
    fn copy(&mut self, source: &Self) -> Result<(), WeightCopyError> {
        self.encoder.copy(&source.encoder)?;
        if self.aggregation_embedding.size() != source.aggregation_embedding.size() {
            return Err(WeightCopyError::SizeMismatch);
        }
        tch::no_grad(|| {
            self.aggregation_embedding.copy_(&source.aggregation_embedding);
        });
        self.head.copy(&source.head)
    }
}

unsafe impl Send for TransformerAggregator {}
unsafe impl Sync for TransformerAggregator {}