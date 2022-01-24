use super::{Embedding, LayerNorm, Linear, ModuleCopy, NNModule, WeightCopyError};
use tch::{nn, Device, IndexOp, Kind, Tensor};

/// Different types of positional encoding for Transformers
#[derive(PartialEq, Eq)]
pub enum PositionalEncoding {
    Learned,
    Sinusoidal,
    Rotary,
}

/// The most basic dot-product self attention with an optional causal mask
#[derive(Debug)]
struct SelfAttention {
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
    fn new(p: &nn::Path, n_embd: i64, n_head: i64, dropout: f64, causal_mask: bool) -> Self {
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
    fn new(p: &nn::Path, n_embd: i64, n_head: i64, dropout: f64, causal_mask: bool) -> Self {
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

/// A basic transformer encoder stack using learned embeddings
#[derive(Debug)]
pub struct TransformerEncoder {
    token_embedding: Embedding,
    position_embedding: LocalPositionalEncoding,
    layernorm: LayerNorm,
    blocks: Vec<TransformerBlock>,
    dropout: f64,
    vocab_size: i64,
    n_embed: i64,
    train: bool,
}

/// An enum for positional encoding which conains a tensor (only used internally)
#[derive(Debug, PartialEq)]
enum LocalPositionalEncoding {
    Learned(Tensor),
    Sinusoidal(Tensor),
    Rotary{inv_freq: Tensor, seq_len_cached: usize, cos_cached: Tensor, sin_cached: Tensor},
}

impl TransformerEncoder {
    #[allow(clippy::too_many_arguments)]
    pub fn new(p: &nn::Path, n_embd: i64, n_head: i64, n_layers: i64, vocab_size: i64, positional_encoding: PositionalEncoding, max_len: i64, dropout: f64, causal_mask: bool) -> Self {
        TransformerEncoder {
            token_embedding: Embedding::new(
                p / "tok_emb",
                vocab_size,
                n_embd,
            ),
            position_embedding: match positional_encoding {
                PositionalEncoding::Learned => LocalPositionalEncoding::Learned(p.randn("pos_emb", &[1, max_len, n_embd], 0., 0.5)),
                PositionalEncoding::Sinusoidal => {
                    // Build the sinusoidal vector (This is based on an online implementation here: https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec#d554
                    let mut pe = vec![vec![0.; n_embd as usize]; max_len as usize];
                    #[allow(clippy::needless_range_loop)]
                    for pos in 0..max_len as usize {
                        for i in 0..n_embd as usize {
                            pe[pos][i] = (pos as f64 / f64::powf(10000., (2.*i as f64) / n_embd as f64)).sin();
                        }
                    }
                    LocalPositionalEncoding::Sinusoidal(Tensor::of_slice2(&pe).to_kind(Kind::Float).to(p.device())) // Doesn't need to be a variable, we aren't tracking it's gradients
                },
                PositionalEncoding::Rotary => {
                    // Build rotary vector
                    let inv_freq = 1.0 / (tch::Tensor::pow_scalar(10_000, &(tch::Tensor::arange_start_step(0, 2, 2, (Kind::Float, p.device())) / 2)));
                    LocalPositionalEncoding::Rotary{
                        inv_freq,
                        seq_len_cached: 0,
                        cos_cached: Tensor::empty(&[1], (Kind::Float, p.device())),
                        sin_cached: Tensor::empty(&[1], (Kind::Float, p.device())),
                    }
                }
            },
            layernorm: LayerNorm::new(p / "ln_f", vec![n_embd]),
            blocks: {
                //let p = &p.set_group(0);
                let mut blocks = Vec::new();
                for block_idx in 0..n_layers {
                    blocks.push(TransformerBlock::new(&(p / block_idx), n_embd, n_head, dropout, causal_mask));
                }
                blocks
            },
            dropout,
            vocab_size,
            n_embed: n_embd,
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
        x = self.blocks[0].forward(&x);
        x = self.blocks
            .iter_mut()
            .skip(1)
            .fold(x, |x, layer| layer.forward(&x));
        // Return first token
        self.layernorm.forward(&x)
        // output shape: (batch size, n_embd)
    }
}

impl NNModule for TransformerEncoder {
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

    fn forward(&mut self, x: &tch::Tensor) -> tch::Tensor {
        // x shape: (batch size, seq len)
        let (batch_size, sz_t) = x.size2().unwrap();
        // Run through embeddings
        let tok_emb = self.token_embedding.forward(x);
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
        let x = self.blocks[0].forward(&x);
        let x = self.blocks
            .iter_mut()
            .skip(1)
            .fold(x, |x, layer| layer.forward(&x));
        // Return first token
        self.layernorm.forward(&x)
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

impl TransformerAggregator {
    #[allow(clippy::too_many_arguments)]
    pub fn new(p: &nn::Path, n_embd: i64, n_head: i64, n_layers: i64, aggregation_size: i64, vocab_size: i64, positional_encoding: PositionalEncoding, max_len: i64, dropout: f64) -> Self {
        TransformerAggregator {
            encoder: TransformerEncoder::new(&(p / "encoder"), n_embd, n_head, n_layers, vocab_size, positional_encoding, max_len, dropout, false),
            head: Linear::new(&(p / "aggregation_head"), n_embd, aggregation_size),
            aggregation_embedding: p.randn("aggregation_vector", &[n_embd], 0.0, 0.2),
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

impl NNModule for TransformerAggregator {
    fn train(&mut self) {
        self.encoder.train();
    }

    fn eval(&mut self) {
        self.encoder.eval();
    }

    fn forward(&mut self, x: &tch::Tensor) -> tch::Tensor {
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
        self.head.forward(&xs.i((.., 0, ..)).squeeze_dim(1))
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

/// A simple language model, using a causally masked transformer encoder and a head
#[derive(Debug)]
pub struct LanguageModel {
    pub transformer: TransformerEncoder,
    pub head: Linear
}

unsafe impl Send for LanguageModel {}
unsafe impl Sync for LanguageModel {}

impl LanguageModel {
    pub fn new(p: &nn::Path, n_embd: i64, n_head: i64, n_layers: i64, vocab_size: i64, positional_encoding: PositionalEncoding, max_len: i64, dropout: f64) -> Self {
        LanguageModel {
            transformer: TransformerEncoder::new(&(p / "transformer"), n_embd, n_head, n_layers, vocab_size, positional_encoding, max_len, dropout, true),
            head: Linear::new(&(p / "lm_head"), n_embd, vocab_size)
        }
    }

    pub fn from_encoder(p: &nn::Path, encoder: TransformerEncoder, vocab_size: i64) -> Self {
        LanguageModel {
            head: Linear::new(&(p / "lm_head"), encoder.n_embed, vocab_size),
            transformer: encoder,
        }
    }
}

impl NNModule for LanguageModel {
    fn train(&mut self) {
        self.transformer.train();
        self.head.train();
    }

    fn eval(&mut self) {
        self.transformer.eval();
        self.head.eval();
    }

    fn forward(&mut self, x: &tch::Tensor) -> tch::Tensor {
        self.head.forward(
            &self.transformer.forward(x)
        )
    }
}

impl ModuleCopy for LanguageModel {
    fn copy(&mut self, source: &Self) -> Result<(), WeightCopyError> {
        self.transformer.copy(&source.transformer)?;
        self.head.copy(&source.head)
    }
}