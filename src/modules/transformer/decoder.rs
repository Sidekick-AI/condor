use crate::modules::{Embedding, LayerNorm, Linear, ModuleCopy, Module, WeightCopyError, PositionalEncoding};
use tch::{nn, IndexOp, Kind, Tensor, Device};
use super::{LocalPositionalEncoding, SelfAttention};

/// The most basic dot-product self attention with an optional causal mask
#[derive(Debug)]
pub(crate) struct DecoderSelfAttention {
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

impl DecoderSelfAttention {
    pub fn new(p: &nn::Path, n_embd: i64, n_head: i64, dropout: f64, causal_mask: bool) -> Self {
        DecoderSelfAttention {
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

impl Clone for DecoderSelfAttention {
    fn clone(&self) -> Self {
        DecoderSelfAttention {
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

impl Module for DecoderSelfAttention {
    type Input = (tch::Tensor, tch::Tensor);
    type Output = tch::Tensor;
    fn train(&mut self) {
        self.train = true;
    }

    fn eval(&mut self) {
        self.train = false;
    }

    fn forward(&mut self, input: Self::Input) -> Self::Output {
        let (input, encoder_output) = input;
        let (sz_b, sz_t, sz_c) = input.size3().unwrap();
        let (enc_sz_b, enc_sz_t, enc_sz_c) = encoder_output.size3().unwrap();
        let sizes = [sz_b, sz_t, self.n_head, sz_c / self.n_head];
        let enc_sizes = [enc_sz_b, enc_sz_t, self.n_head, enc_sz_c / self.n_head];
        let device = input.device();
        let k = self.key.forward(encoder_output.shallow_clone()).view(enc_sizes).transpose(1, 2);
        let q = self.query.forward(input.shallow_clone()).view(sizes).transpose(1, 2);
        let v = self.value.forward(encoder_output.shallow_clone()).view(enc_sizes).transpose(1, 2);
        let mut att = q.matmul(&k.transpose(-2, -1)) * (1.0 / f64::sqrt(sizes[3] as f64));
        if self.causal_mask {
            let mask = DecoderSelfAttention::generate_mask(sz_t, device);
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
        self.proj.forward(ys).dropout(self.dropout, self.train)
    }
}

impl ModuleCopy for DecoderSelfAttention {
    fn copy(&mut self, source: &Self) -> Result<(), WeightCopyError> {
        self.key.copy(&source.key)?;
        self.query.copy(&source.query)?;
        self.value.copy(&source.value)?;
        self.proj.copy(&source.proj)?;
        Ok(())
    }
}

/// A basic transformer decoder block
#[derive(Debug)]
pub struct TransformerDecoderBlock {
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
    attn: SelfAttention,
    attn2: DecoderSelfAttention,
    linear1: Linear,
    linear2: Linear,
    dropout: f64,
    train: bool,
}

impl TransformerDecoderBlock {
    pub fn new(p: &nn::Path, n_embd: i64, n_head: i64, dropout: f64, causal_mask: bool) -> Self {
        assert!(n_embd % n_head == 0, "Embedding size ({}) must be divisible by number of heads ({})!", n_embd, n_head);
        TransformerDecoderBlock {
            norm1: LayerNorm::new(p / "ln1", vec![n_embd]),
            norm2: LayerNorm::new(p / "ln2", vec![n_embd]),
            norm3: LayerNorm::new(p / "ln3", vec![n_embd]),
            attn: SelfAttention::new(&(p / "attn"), n_embd, n_head, dropout, causal_mask),
            attn2: DecoderSelfAttention::new(&(p / "attn2"), n_embd, n_head, dropout, false),
            linear1: Linear::new(&(p / "lin1"), n_embd, 2 * n_embd),
            linear2: Linear::new(&(p / "lin2"), 2 * n_embd, n_embd),
            dropout,
            train: true
        }
    }
}

impl Module for TransformerDecoderBlock {
    type Input = (tch::Tensor, tch::Tensor);
    type Output = tch::Tensor;

    fn train(&mut self) {
        self.attn.train();
        self.train = true;
    }

    fn eval(&mut self) {
        self.attn.eval();
        self.train = false;
    }

    fn forward(&mut self, input: Self::Input) -> Self::Output {
        let (input, encoder_output) = input;
        let x = input.shallow_clone() + self.norm1.forward(self.attn.forward(input));

        let x = self.norm2.forward(x.shallow_clone() + self.attn2.forward((x, encoder_output)));

        self.norm3.forward(
            x.shallow_clone() + 
            self.linear2.forward(
                self.linear1.forward(x).gelu()
            )
        ).dropout(self.dropout, self.train)
    }
}

impl ModuleCopy for TransformerDecoderBlock {
    fn copy(&mut self, source: &Self) -> Result<(), WeightCopyError> {
        self.attn.copy(&source.attn)?;
        self.attn2.copy(&source.attn2)?;
        self.norm1.copy(&source.norm1)?;
        self.norm2.copy(&source.norm2)?;
        self.norm3.copy(&source.norm3)?;
        self.linear1.copy(&source.linear1)?;
        self.linear2.copy(&source.linear2)
    }
}

/// A simple autoregressive transformer decoder
#[derive(Debug)]
pub struct TransformerDecoder {
    token_embedding: Embedding,
    position_embedding: LocalPositionalEncoding,
    layernorm: LayerNorm,
    blocks: Vec<TransformerDecoderBlock>,
    dropout: f64,
    train: bool,
}

pub struct TransformerDecoderProps<'a> {
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

impl TransformerDecoder {
    #[allow(clippy::too_many_arguments)]
    pub fn new(props: TransformerDecoderProps) -> Self {
        TransformerDecoder {
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
                }
            },
            layernorm: LayerNorm::new(props.p / "ln_f", vec![props.n_embd]),
            blocks: {
                //let p = &p.set_group(0);
                let mut blocks = Vec::new();
                for block_idx in 0..props.n_layers {
                    blocks.push(TransformerDecoderBlock::new(&(props.p / block_idx), props.n_embd, props.n_head, props.dropout, props.causal_mask));
                }
                blocks
            },
            dropout: props.dropout,
            train: true,
        }
    }
}

impl Module for TransformerDecoder {
    type Input = (tch::Tensor, tch::Tensor);
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
        let (input, encoder_output) = input;
        // x shape: (batch size, seq len)
        let (batch_size, sz_t) = input.size2().unwrap();
        // Run through embeddings
        let tok_emb = self.token_embedding.forward(input);
        let pos_emb = match &mut self.position_embedding {
            LocalPositionalEncoding::Learned(l) => l.i((.., ..sz_t, ..)).repeat(&[batch_size, 1, 1]),
            LocalPositionalEncoding::Sinusoidal(pe) => {
                pe.i(..sz_t).repeat(&[batch_size, 1, 1])
            }
        };
        let x = (tok_emb + pos_emb)
            .dropout(self.dropout, self.train);
        // Run through transformer blocks
        let x = self.blocks[0].forward((x, encoder_output.shallow_clone()));
        let x = self.blocks
            .iter_mut()
            .skip(1)
            .fold(x, |x, layer| layer.forward((x, encoder_output.shallow_clone())));
        // Return first token
        self.layernorm.forward(x)
        // output shape: (batch size, n_embd)
    }
}

impl ModuleCopy for TransformerDecoder {
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
            }
        }

        self.layernorm.copy(&source.layernorm)?;
        for i in 0..self.blocks.len() {
            self.blocks[i].copy(&source.blocks[i])?;
        }
        Ok(())
    }
}