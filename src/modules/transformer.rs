use super::{Linear, NNModule, Sequential};
use tch::nn::{Embedding, LayerNorm, Module};
use tch::{nn, Device, IndexOp, Kind, Tensor};

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

impl NNModule for SelfAttention {
    fn train(&mut self) {
        self.train = true;
    }

    fn eval(&mut self) {
        self.train = false;
    }

    fn count_parameters(&self) -> u64 {
        self.key.count_parameters() + self.query.count_parameters() + self.value.count_parameters() + self.proj.count_parameters()
    }
}

impl nn::Module for SelfAttention {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let (sz_b, sz_t, sz_c) = xs.size3().unwrap();
        let sizes = [sz_b, sz_t, self.n_head, sz_c / self.n_head];
        let k = xs.apply(&self.key).view(sizes).transpose(1, 2);
        let q = xs.apply(&self.query).view(sizes).transpose(1, 2);
        let v = xs.apply(&self.value).view(sizes).transpose(1, 2);
        let mut att = q.matmul(&k.transpose(-2, -1)) * (1.0 / f64::sqrt(sizes[3] as f64));
        if self.causal_mask {
            let mask = SelfAttention::generate_mask(sz_t, xs.device());
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
        ys.apply(&self.proj).dropout(self.dropout, self.train)
    }
}

/// A basic transformer encoder block
#[derive(Debug)]
struct TransformerBlock {
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
        TransformerBlock {
            norm1: nn::layer_norm(p / "ln1", vec![n_embd], Default::default()),
            norm2: nn::layer_norm(p / "ln2", vec![n_embd], Default::default()),
            attn: SelfAttention::new(&(p / "attn"), n_embd, n_head, dropout, causal_mask),
            linear1: Linear::new(&(p / "lin1"), n_embd, 2 * n_embd),
            linear2: Linear::new(&(p / "lin2"), 2 * n_embd, n_embd),
            dropout,
            train: true
        }
    }
}

impl nn::Module for TransformerBlock {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let xs = xs + xs.apply(&self.norm1).apply_t(&self.attn, self.train);
        let ys = xs
            .apply(&self.norm2)
            .apply(&self.linear1)
            .gelu()
            .apply(&self.linear2)
            .dropout(self.dropout, self.train);
        xs + ys
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

    fn count_parameters(&self) -> u64 {
        self.attn.count_parameters() + self.linear1.count_parameters() + self.linear2.count_parameters()
    }
}

/// A basic transformer encoder stack using learned embeddings
#[derive(Debug)]
pub struct TransformerEncoder {
    token_embedding: Embedding,
    position_embedding: Tensor,
    layernorm: LayerNorm,
    blocks: Sequential,
    dropout: f64,
    vocab_size: i64,
    n_embed: i64,
    train: bool,
}

impl TransformerEncoder {
    pub fn new(p: &nn::Path, n_embd: i64, n_head: i64, n_layers: i64, vocab_size: i64, max_len: i64, dropout: f64, causal_mask: bool) -> Self {
        TransformerEncoder {
            token_embedding: nn::embedding(
                p / "tok_emb",
                vocab_size,
                n_embd,
                Default::default(),
            ),
            position_embedding: p.zeros("pos_emb", &[1, max_len, n_embd]),
            layernorm: nn::layer_norm(p / "ln_f", vec![n_embd], Default::default()),
            blocks: {
                //let p = &p.set_group(0);
                let mut blocks = Sequential::new();
                for block_idx in 0..n_layers {
                    blocks.add(TransformerBlock::new(&(p / block_idx), n_embd, n_head, dropout, causal_mask));
                }
                blocks
            },
            dropout,
            vocab_size,
            n_embed: n_embd,
            train: true,
        }
    }

    pub fn forward_no_embed(&self, xs: &Tensor) -> Tensor {
        // xs shape: (batch size, seq len, n_embd)
        let (_, sz_t, _) = xs.size3().unwrap();
        let pos_emb = self.position_embedding.i((.., ..sz_t, ..));
        let mut x = (xs + pos_emb)
            .dropout(self.dropout, self.train);
        // Run through transformer blocks
        x = self.blocks.forward(&x);
        // Return first token
        x.apply(&self.layernorm)
        // output shape: (batch size, n_embd)
    }
}

impl nn::Module for TransformerEncoder {
    fn forward(&self, xs: &Tensor) -> Tensor {
        // xs shape: (batch size, seq len)
        let (_, sz_t) = xs.size2().unwrap();
        // Run through embeddings
        let tok_emb = xs.apply(&self.token_embedding);
        let pos_emb = self.position_embedding.i((.., ..sz_t, ..));
        let mut x = (tok_emb + pos_emb)
            .dropout(self.dropout, self.train);
        // Run through transformer blocks
        x = self.blocks.forward(&x);
        // Return first token
        x.apply(&self.layernorm)
        // output shape: (batch size, n_embd)
    }
}

impl NNModule for TransformerEncoder {
    fn train(&mut self) {
        for block in &mut self.blocks.layers {
            block.train();
        }
        self.train = true;
    }

    fn eval(&mut self) {
        for block in &mut self.blocks.layers {
            block.eval();
        }
        self.train = false;
    }

    fn count_parameters(&self) -> u64 {
        self.token_embedding.ws.size().iter().map(|t| {*t as u64}).fold(1u64, |total, val| {total * val})
        + self.position_embedding.size().iter().map(|t| {*t as u64}).fold(1u64, |total, val| {total * val})
        + self.blocks.layers.iter().map(|block| {block.count_parameters()}).sum::<u64>()
    }
}


/// A transformer encoder that aggregates a sequence into a single vector
#[derive(Debug)]
pub struct TransformerAggregator {
    encoder: TransformerEncoder,
    aggregation_embedding: Tensor,
    head: Linear
}

impl TransformerAggregator {
    pub fn new(p: &nn::Path, n_embd: i64, n_head: i64, n_layers: i64, aggregation_size: i64, vocab_size: i64, max_len: i64, dropout: f64) -> Self {
        TransformerAggregator {
            encoder: TransformerEncoder::new(&(p / "encoder"), n_embd, n_head, n_layers, vocab_size, max_len, dropout, false),
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

impl nn::Module for TransformerAggregator {
    fn forward(&self, xs: &Tensor) -> Tensor {
        // xs shape: (batch size, seq len)
        let batch_size = xs.size()[0];
        // Embed and append aggregation embedding to beginning
        let mut xs = tch::Tensor::cat(&[
            &self.aggregation_embedding.unsqueeze(0).unsqueeze(0).repeat(&[batch_size, 1, 1]), 
            &self.encoder.token_embedding.forward(&xs)
        ], 1);
        // Run through encoder
        xs = self.encoder.forward_no_embed(&xs);
        // Return first token
        xs.i((.., 0, ..)).squeeze_dim(1).apply(&self.head)
        // output shape: (batch size, n_embd)
    }
}

impl NNModule for TransformerAggregator {
    fn train(&mut self) {
        self.encoder.train();
    }

    fn eval(&mut self) {
        self.encoder.eval();
    }

    fn count_parameters(&self) -> u64 {
        self.encoder.count_parameters() + self.head.count_parameters()
    }
}

/// A simple language model, using a causally masked transformer encoder and a head
#[derive(Debug)]
pub struct LanguageModel {
    transformer: TransformerEncoder,
    head: Linear
}

impl LanguageModel {
    pub fn new(p: &nn::Path, n_embd: i64, n_head: i64, n_layers: i64, vocab_size: i64, max_len: i64, dropout: f64) -> Self {
        LanguageModel {
            transformer: TransformerEncoder::new(&(p / "transformer"), n_embd, n_head, n_layers, vocab_size, max_len, dropout, true),
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

impl nn::Module for LanguageModel {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.transformer).apply(&self.head)
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

    fn count_parameters(&self) -> u64 {
        self.transformer.count_parameters() + self.head.count_parameters()
    }
}