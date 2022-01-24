#[cfg(test)]
mod linear_tests {
    use tch::{Device, Kind, Tensor, nn};
    use crate::{modules::NNModule, utils::count_parameters};

    use super::super::Linear;

    #[test]
    fn test_linear() {
        let vs = nn::VarStore::new(Device::cuda_if_available());
        let mut layer = Linear::new(&(&vs.root() / "linear"), 100, 20);
        let input = Tensor::rand(&[64, 100], (Kind::Float, Device::cuda_if_available()));
        let output = layer.forward(&input);
        assert_eq!(output.size(), &[64, 20]);
        assert_eq!(count_parameters(&vs), 2020);
    }
}

#[cfg(test)]
mod rnn_test {

}

#[cfg(test)]
mod sequential_tests {
    use tch::{Device, Kind, Tensor, nn};
    use crate::{modules::NNModule, sequential, utils::count_parameters};
    use super::super::{Linear, Sequential, PReLU};

    #[test]
    fn test_sequential() {
        let vs = nn::VarStore::new(Device::cuda_if_available());
        let linear_layer1 = Linear::new(&(&vs.root() / "linear"), 100, 20);
        let linear_layer2 = Linear::new(&(&vs.root() / "linear"), 20, 150);
        let mut seq = sequential!(linear_layer1, PReLU::new(&vs.root() / "linear"), linear_layer2);
        
        let input = Tensor::rand(&[64, 100], (Kind::Float, Device::cuda_if_available()));
        let output = seq.forward(&input);
        assert_eq!(output.size(), &[64, 150]);
        assert_eq!(count_parameters(&vs), 5171);
    }
}