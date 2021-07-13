#[cfg(test)]
mod linear_tests {
    use tch::{Device, Kind, Tensor, nn::{self, Module}};
    use super::super::linear::Linear;

    #[test]
    fn test_linear() {
        let vs = nn::VarStore::new(Device::cuda_if_available());
        let layer = Linear::new(&vs.root() / "linear", 100, 20);
        let input = Tensor::rand(&[64, 100], (Kind::Float, Device::cuda_if_available()));
        let output = layer.forward(&input);
        assert_eq!(output.size(), &[64, 20]);
    }
}