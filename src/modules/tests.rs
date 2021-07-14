#[cfg(test)]
mod linear_tests {
    use tch::{Device, Kind, Tensor, nn::{self, Module}};
    use crate::modules::NNModule;

    use super::super::Linear;

    #[test]
    fn test_linear() {
        let vs = nn::VarStore::new(Device::cuda_if_available());
        let layer = Linear::new(&(&vs.root() / "linear"), 100, 20);
        let input = Tensor::rand(&[64, 100], (Kind::Float, Device::cuda_if_available()));
        let output = layer.forward(&input);
        assert_eq!(output.size(), &[64, 20]);
        assert_eq!(layer.count_parameters(), 2020);
    }
}

#[cfg(test)]
mod rnn_test {

}

#[cfg(test)]
mod transformer_test {
    use tch::{Device, Kind, Tensor, nn::{self, Module}};
    use super::super::{NNModule, TransformerAggregator, TransformerEncoder};

    #[test]
    fn test_transformer_encoder() {
        let vs = nn::VarStore::new(Device::cuda_if_available());
        let transformer_encoder = TransformerEncoder::new(&(&vs.root() / "transformer"), 
        100, 
        10, 
        3, 
        100, 
        100, 
        0.1, 
        true);
        let input = Tensor::randint(99, &[15, 50], (Kind::Int, Device::cuda_if_available()));
        let output = transformer_encoder.forward(&input);
        assert_eq!(output.size(), &[15, 50, 100]);
        assert_eq!(transformer_encoder.count_parameters(), 382700);
    }

    #[test]
    fn test_transformer_aggregator() {
        let vs = nn::VarStore::new(Device::cuda_if_available());
        let transformer_aggregator = TransformerAggregator::new(&(&vs.root() / "transformer"), 
        100, 
        10, 
        3, 
        150,
        100,
        100, 
        0.1);
        let input = Tensor::randint(99, &[15, 50], (Kind::Int, Device::cuda_if_available()));
        let output = transformer_aggregator.forward(&input);
        assert_eq!(output.size(), &[15, 150]);
        assert_eq!(transformer_aggregator.count_parameters(), 397850);
    }

    #[test]
    fn test_transformer_aggregator_from_encoder() {
        let vs = nn::VarStore::new(Device::cuda_if_available());
        let transformer_encoder = TransformerEncoder::new(&(&vs.root() / "transformer"), 
        100, 
        10, 
        3, 
        100, 
        100, 
        0.1, 
        true);
        let transformer_aggregator = TransformerAggregator::from_encoder(&(&vs.root() / "transformer"), transformer_encoder, 150);
        let input = Tensor::randint(99, &[15, 50], (Kind::Int, Device::cuda_if_available()));
        let output = transformer_aggregator.forward(&input);
        assert_eq!(output.size(), &[15, 150]);
        assert_eq!(transformer_aggregator.count_parameters(), 397850);
    }
}

#[cfg(test)]
mod sequential_tests {
    use tch::{Device, Kind, Tensor, nn::{self, Module}};
    use crate::{modules::NNModule, sequential};
    use super::super::{Linear, Sequential, PReLU};

    #[test]
    fn test_sequential() {
        let vs = nn::VarStore::new(Device::cuda_if_available());
        let linear_layer1 = Linear::new(&(&vs.root() / "linear"), 100, 20);
        let linear_layer2 = Linear::new(&(&vs.root() / "linear"), 20, 150);
        let seq = sequential!(linear_layer1, PReLU::new(&vs.root() / "linear"), linear_layer2);
        
        let input = Tensor::rand(&[64, 100], (Kind::Float, Device::cuda_if_available()));
        let output = seq.forward(&input);
        assert_eq!(output.size(), &[64, 150]);
        assert_eq!(seq.count_parameters(), 5171);
    }
}