use super::{NNModule, Linear};
use tch::nn::{Embedding, LayerNorm};
use tch::{nn, Device, IndexOp, Kind, Tensor};

#[derive(Debug)]
struct CausalSelfAttention {
    n_head: i64,
    n_embd: i64,
    dropout: f64,
    key: Linear,
    query: Linear,
    value: Linear,
    proj: Linear,
    train: bool,
}

impl CausalSelfAttention {
    fn new(p: &nn::Path, n_embd: i64, n_head: i64, dropout: f64) -> Self {
        CausalSelfAttention {
            n_embd,
            n_head,
            dropout,
            key: Linear::new(p / "key", n_embd, n_embd),
            query: Linear::new(p / "query", n_embd, n_embd),
            value: Linear::new(p / "value", n_embd, n_embd),
            proj: Linear::new(p / "proj", n_embd, n_embd),
            train: true,
        }
    }

    fn generate_mask(size: i64, device: Device) -> Tensor{
        Tensor::ones(&[size, size], (Kind::Float, device)).tril(0).view([1, 1, size, size])
    }
}

impl NNModule for CausalSelfAttention {
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

impl nn::Module for CausalSelfAttention {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let (sz_b, sz_t, sz_c) = xs.size3().unwrap();
        let sizes = [sz_b, sz_t, self.n_head, sz_c / self.n_head];
        let k = xs.apply(&self.key).view(sizes).transpose(1, 2);
        let q = xs.apply(&self.query).view(sizes).transpose(1, 2);
        let v = xs.apply(&self.value).view(sizes).transpose(1, 2);
        let att = q.matmul(&k.transpose(-2, -1)) * (1.0 / f64::sqrt(sizes[3] as f64));
        let mask = CausalSelfAttention::generate_mask(sz_t, xs.device());
        let att = att.masked_fill(
            &mask.i((.., .., ..sz_t, ..sz_t)).eq(0.),
            std::f64::NEG_INFINITY,
        );
        let att = att.softmax(-1, Kind::Float).dropout(self.dropout, self.train);
        let ys = att
            .matmul(&v)
            .transpose(1, 2)
            .contiguous()
            .view([sz_b, sz_t, sz_c]);
        ys.apply(&self.proj).dropout(self.dropout, self.train)
    }
}

#[derive(Debug)]
struct TransformerBlock {
    norm1: LayerNorm,
    norm2: LayerNorm,
    attn: CausalSelfAttention,
    linear1: Linear,
    linear2: Linear,
    dropout: f64,
    train: bool,
}

impl TransformerBlock {
    fn new(p: &nn::Path, n_embd: i64, n_head: i64, dropout: f64) -> Self {
        TransformerBlock {
            norm1: nn::layer_norm(p / "ln1", vec![n_embd], Default::default()),
            norm2: nn::layer_norm(p / "ln2", vec![n_embd], Default::default()),
            attn: CausalSelfAttention::new(p, n_embd, n_head, dropout),
            linear1: Linear::new(p / "lin1", n_embd, 4 * n_embd),
            linear2: Linear::new(p / "lin2", 4 * n_embd, n_embd),
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

#[derive(Debug)]
pub struct TransformerEncoder {
    token_embedding: Embedding,
    position_embedding: Tensor,
    layernorm: LayerNorm,
    head: Linear,
    blocks: Vec<TransformerBlock>,
    dropout: f64,
    vocab_size: i64,
    n_embed: i64,
    train: bool,
}

impl TransformerEncoder {
    pub fn new(p: &nn::Path, n_embd: i64, n_head: i64, n_layers: i64, vocab_size: i64, max_len: i64, dropout: f64) -> Self {
        TransformerEncoder {
            token_embedding: nn::embedding(
                p / "tok_emb",
                vocab_size,
                n_embd,
                Default::default(),
            ),
            position_embedding: p.zeros("pos_emb", &[1, max_len, n_embd]),
            layernorm: nn::layer_norm(p / "ln_f", vec![n_embd], Default::default()),
            head: Linear::no_bias(p / "head", n_embd, n_embd),
            blocks: {
                let p = &p.set_group(0);
                let mut blocks = vec![];
                for block_idx in 0..n_layers {
                    blocks.push(TransformerBlock::new(&(p / block_idx), n_embd, n_head, dropout));
                }
                blocks
            },
            dropout,
            vocab_size,
            n_embed: n_embd,
            train: true,
        }
    }
}

impl nn::Module for TransformerEncoder {
    fn forward(&self, xs: &Tensor) -> Tensor {
        // xs shape: (batch size, seq len)
        // Append aggregation token to beginning
        let (sz_b, sz_t) = xs.size2().unwrap();
        let new = Tensor::full(&[sz_b, 1], self.vocab_size - 1, (xs.kind(), xs.device()));
        let xs = tch::Tensor::cat(&[xs, &new], 1);
        // Run through embeddings
        let tok_emb = xs.apply(&self.token_embedding);
        let pos_emb = self.position_embedding.i((.., ..sz_t + 1, ..));
        let mut x = (tok_emb + pos_emb)
            .dropout(self.dropout, self.train);
        // Run through transformer blocks
        for block in &self.blocks {
            x = x.apply_t(block, self.train);
        }
        // Return first token
        x.i((.., 0, ..)).squeeze_dim(1)
            .apply(&self.layernorm).apply(&self.head)
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

    fn count_parameters(&self) -> u64 {
        self.token_embedding.ws.size().iter().map(|t| {*t as u64}).fold(1u64, |total, val| {total * val})
        + self.position_embedding.size().iter().map(|t| {*t as u64}).fold(1u64, |total, val| {total * val})
        + self.head.count_parameters()
        + self.blocks.iter().map(|block| {block.count_parameters()}).sum::<u64>()
    }
}