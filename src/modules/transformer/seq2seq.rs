use crate::modules::{Linear, ModuleCopy, Module, WeightCopyError, PositionalEncoding};
use tch::{nn, Tensor, Kind, IndexOp};
use super::{TransformerEncoder, TransformerDecoder, TransformerEncoderProps, TransformerDecoderProps};

pub enum InferenceMode {
    NTokens(u32),
    EndToken(i64),
}

/// A basic transformer encoder stack using learned embeddings
#[derive(Debug)]
pub struct Seq2SeqTransformer {
    encoder: TransformerEncoder,
    decoder: TransformerDecoder,
    head: Linear,
}

pub struct Seq2SeqTransformerProps<'a> {
    pub p: &'a nn::Path<'a>, 
    pub n_embd: i64, 
    pub n_encoder_heads: i64,
    pub n_decoder_heads: i64, 
    pub n_encoder_layers: i64,
    pub n_decoder_layers: i64, 
    pub vocab_size: i64, 
    pub positional_encoding: PositionalEncoding, 
    pub max_len: i64, 
    pub dropout: f64, 
}

impl Seq2SeqTransformer {
    pub fn new(props: Seq2SeqTransformerProps<'_>) -> Self {
        Seq2SeqTransformer {
            encoder: TransformerEncoder::new(TransformerEncoderProps {
                p: &(props.p / "encoder"),
                n_embd: props.n_embd,
                n_head: props.n_encoder_heads,
                vocab_size: props.vocab_size,
                n_layers: props.n_encoder_layers,
                positional_encoding: props.positional_encoding.clone(),
                max_len: props.max_len,
                dropout: props.dropout,
                causal_mask: false,
            }),
            decoder: TransformerDecoder::new(TransformerDecoderProps {
                p: &(props.p / "decoder"),
                n_embd: props.n_embd,
                n_head: props.n_decoder_heads,
                n_layers: props.n_decoder_layers,
                vocab_size: props.vocab_size,
                positional_encoding: props.positional_encoding,
                max_len: props.max_len,
                dropout: props.dropout,
                causal_mask: true
            }),
            head: Linear::new(&(props.p / "head"), props.n_embd, props.vocab_size),
        }
    }

    pub fn forward_generate(&mut self, input: tch::Tensor, sos_index: i64, mode: InferenceMode) -> tch::Tensor {
        // Input shape: (1, seq_len)
        assert_eq!(input.size()[0], 1, "During inference, can only use batch size of 1");
        // Encode input
        let encoded_input = self.encoder.forward(input);

        // Iteratively decode output until stop condition based on mode
        let mut output = Tensor::full(&[1, 1], sos_index, (Kind::Int, encoded_input.device()));

        let mut counter = 0;
        loop {
            // Feed through decoder
            output = self.head.forward(self.decoder.forward((output, encoded_input.shallow_clone())));
            // output: (1, current seq len, vocab size)

            // Exit condition
            counter += 1;
            match mode {
                InferenceMode::NTokens(n) => if counter >= n {break},
                InferenceMode::EndToken(token) => {
                    // Argmax last token
                    let output_token = output.i((0, 0)).argmax(Some(0), true).int64_value(&[0]);
                    if output_token == token || counter > 5000 {break} // Provide a built in stop at 5000 tokens
                },
            }

            // Sample outputs
            output = output.squeeze_dim(0).softmax(-1, Kind::Float).multinomial(1, true).squeeze_dim(-1).unsqueeze(0);
            // Cat new SOS token
            output = Tensor::cat(&[Tensor::full(&[1, 1], sos_index, (Kind::Int, encoded_input.device())), output], 1);
        }

        output
    }
}

impl Module for Seq2SeqTransformer {
    type Input = (tch::Tensor, tch::Tensor);
    type Output = tch::Tensor;

    fn train(&mut self) {
        self.encoder.train();
        self.decoder.train();
    }

    fn eval(&mut self) {
        self.encoder.eval();
        self.decoder.eval();
    }

    fn forward(&mut self, input: Self::Input) -> Self::Output {
        let (input, target) = input;
        // Encode inputs
        let encoded_inputs = self.encoder.forward(input);

        // Decode outputs
        let output_vecs = self.decoder.forward((target, encoded_inputs));

        // Convert into logits over tokens
        self.head.forward(output_vecs)
    }
}

impl ModuleCopy for Seq2SeqTransformer {
    fn copy(&mut self, source: &Self) -> Result<(), WeightCopyError> {
        self.encoder.copy(&source.encoder)?;
        self.decoder.copy(&source.decoder)
    }
}