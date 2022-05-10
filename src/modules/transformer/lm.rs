use crate::modules::{Linear, ModuleCopy, Module, WeightCopyError, TransformerEncoder, TransformerEncoderProps, PositionalEncoding};
use tch::nn;

/// A simple language model, using a causally masked transformer encoder and a head
#[derive(Debug)]
pub struct LanguageModel {
    pub transformer: TransformerEncoder,
    pub head: Linear
}

pub struct LanguageModelProps<'a> {
    pub p: &'a nn::Path<'a>, 
    pub n_embd: i64, 
    pub n_head: i64, 
    pub n_layers: i64, 
    pub vocab_size: i64, 
    pub positional_encoding: PositionalEncoding, 
    pub max_len: i64, 
    pub dropout: f64
}

unsafe impl Send for LanguageModel {}
unsafe impl Sync for LanguageModel {}

impl LanguageModel {
    pub fn new(props: LanguageModelProps) -> Self {
        LanguageModel {
            transformer: TransformerEncoder::new(TransformerEncoderProps{
                p: &(props.p / "transformer"), 
                n_embd: props.n_embd, 
                n_head: props.n_head, 
                n_layers: props.n_layers, 
                vocab_size: props.vocab_size, 
                positional_encoding: props.positional_encoding, 
                max_len: props.max_len, 
                dropout: props.dropout, 
                causal_mask: true
            }),
            head: Linear::new(&(props.p / "lm_head"), props.n_embd, props.vocab_size)
        }
    }

    pub fn from_encoder(p: &nn::Path, encoder: TransformerEncoder, vocab_size: i64) -> Self {
        LanguageModel {
            head: Linear::new(&(p / "lm_head"), encoder.n_embed, vocab_size),
            transformer: encoder,
        }
    }
}

impl Module for LanguageModel {
    type Input = tch::Tensor;
    type Output = tch::Tensor;

    fn train(&mut self) {
        self.transformer.train();
        self.head.train();
    }

    fn eval(&mut self) {
        self.transformer.eval();
        self.head.eval();
    }

    fn forward(&mut self, input: Self::Input) -> Self::Output {
        self.head.forward(
            self.transformer.forward(input)
        )
    }
}

impl ModuleCopy for LanguageModel {
    fn copy(&mut self, source: &Self) -> Result<(), WeightCopyError> {
        self.transformer.copy(&source.transformer)?;
        self.head.copy(&source.head)
    }
}