use crate::modules::Module;
use super::{TransformerEncoder, TransformerDecoder, TransformerEncoderProps, TransformerDecoderProps};

/// A basic transformer encoder stack using learned embeddings
#[derive(Debug)]
pub struct TransformerSeq2Seq {
    pub encoder: TransformerEncoder,
    pub decoder: TransformerDecoder,
}

pub enum Seq2SeqInferenceMode {
    NTokens(usize), // Generate N tokens then stop
    UntilToken(usize), // Generate until a certain token is produced
}

impl TransformerSeq2Seq {
    pub fn new(encoder_props: TransformerEncoderProps, decoder_props: TransformerDecoderProps) -> TransformerSeq2Seq {
        TransformerSeq2Seq {
            encoder: TransformerEncoder::new(encoder_props),
            decoder: TransformerDecoder::new(decoder_props),
        }
    }

    pub fn forward_inference(&mut self, input: tch::Tensor, mode: Seq2SeqInferenceMode) -> tch::Tensor {
        // encode input
        let encoded_input = self.encoder.forward(input);
        // Depending on the inference mode produce output
        match mode {
            Seq2SeqInferenceMode::NTokens(n) => {
                let mut output = tch::Tensor::full(&[encoded_input.size()[0], 0], 0, (tch::Kind::Int, encoded_input.device()));
                for _ in 0..n {
                    output = tch::Tensor::cat(&[output, tch::Tensor::full(&[encoded_input.size()[0], 1], 0, (tch::Kind::Int, encoded_input.device()))], 1);
                    output = self.decoder.forward((encoded_input.shallow_clone(), output));
                }
                output
            },
            Seq2SeqInferenceMode::UntilToken(_token) => {
                todo!()
            }
        }
    }
}

impl Module for TransformerSeq2Seq {
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
        // x shape: (batch size, seq len)
        let encoded = self.encoder.forward(input);
        // encoded shape: (batch size, n embed)
        self.decoder.forward((encoded, target))
    }
}