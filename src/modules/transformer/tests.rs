use tch::{Device, Kind, Tensor, nn};

use crate::{modules::{Module, Seq2SeqTransformer, Seq2SeqTransformerProps}, utils::count_parameters};

use super::super::{TransformerAggregator, TransformerAggregatorProps, TransformerEncoder, TransformerEncoderProps};

#[test]
fn test_transformer_encoder() {
    let vs = nn::VarStore::new(Device::cuda_if_available());
    let mut transformer_encoder = TransformerEncoder::new(
        TransformerEncoderProps {
            p: &(&vs.root() / "transformer"),
            n_embd: 100,
            n_head: 10,
            n_layers: 3,
            vocab_size: 120,
            positional_encoding: crate::modules::PositionalEncoding::Learned,
            max_len: 110,
            dropout: 0.1,
            causal_mask: true,
        });
    let input = Tensor::randint(119, &[15, 50], (Kind::Int, Device::cuda_if_available()));
    let output = transformer_encoder.forward(input);
    assert_eq!(output.size(), &[15, 50, 100]);
    assert_eq!(count_parameters(&vs), 266_500);
}

#[test]
fn test_transformer_aggregator() {
    let vs = nn::VarStore::new(Device::cuda_if_available());
    let mut transformer_aggregator = TransformerAggregator::new(TransformerAggregatorProps {
        p: &(&vs.root() / "transformer"),
        n_embd: 100,
        n_head: 10,
        n_layers: 3,
        aggregation_size: 150,
        vocab_size: 120,
        positional_encoding: crate::modules::PositionalEncoding::Learned,
        max_len: 110,
        dropout: 0.1,
    });
    let input = Tensor::randint(119, &[15, 50], (Kind::Int, Device::cuda_if_available()));
    let output = transformer_aggregator.forward(input);
    assert_eq!(output.size(), &[15, 150]);
    assert_eq!(count_parameters(&vs), 281_750);
}

#[test]
fn test_transformer_aggregator_from_encoder() {
    let vs = nn::VarStore::new(Device::cuda_if_available());
    let transformer_encoder = TransformerEncoder::new(TransformerEncoderProps {
        p: &(&vs.root() / "transformer"),
        n_embd: 100,
        n_head: 10,
        n_layers: 3,
        vocab_size: 120,
        positional_encoding: crate::modules::PositionalEncoding::Learned,
        max_len: 110,
        dropout: 0.1,
        causal_mask: true,
    });
    let mut transformer_aggregator = TransformerAggregator::from_encoder(&(&vs.root() / "transformer"), transformer_encoder, 150);
    let input = Tensor::randint(119, &[15, 50], (Kind::Int, Device::cuda_if_available()));
    let output = transformer_aggregator.forward(input);
    assert_eq!(output.size(), &[15, 150]);
    assert_eq!(count_parameters(&vs), 281_750);
}

#[test]
fn test_transformer_seq2seq() {
    let vs = nn::VarStore::new(Device::cuda_if_available());
    let mut transformer_seq2seq = Seq2SeqTransformer::new(
        Seq2SeqTransformerProps {
            p: &(&vs.root() / "transformer"),
            n_embd: 100,
            n_encoder_heads: 10,
            n_encoder_layers: 3,
            n_decoder_heads: 5,
            n_decoder_layers: 3,
            vocab_size: 120,
            positional_encoding: crate::modules::PositionalEncoding::Learned,
            max_len: 110,
            dropout: 0.1,
        });

    // batch size : 15, seq len: 50
    let input = Tensor::randint(119, &[15, 50], (Kind::Int, Device::cuda_if_available()));
    let output = transformer_seq2seq.forward((input.shallow_clone(), input));
    assert_eq!(output.size(), &[15, 50, 120]);
    assert_eq!(count_parameters(&vs), 666920);
}