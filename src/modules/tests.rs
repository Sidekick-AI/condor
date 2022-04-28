use crate::{Module, Tensor2, sequential, Tensor3};
use crate::modules::*;

#[test]
fn test_linear() {
    let linear: Linear<10, 100, 50> = Linear::default();

    let inp: Tensor2<10, 100> = Tensor2::rand();
    let _output = linear.forward(inp);
}

#[test]
fn test_activations() {
    let tensor: Tensor3<10, 64, 1000> = Tensor3::rand();

    let tensor = tensor.relu();
    let tensor = tensor.gelu();
    let tensor = tensor.sigmoid();

    let prelu = PReLU3::default();
    let _tensor = prelu.forward(tensor);
}

#[test]
fn test_sequential() {
    let sequential = sequential!(
        Linear::<100, 50, 100>::default(),
        Linear::<100, 100, 25>::default()
    );

    let input: Tensor2<100, 50> = Tensor2::rand();
    let _output = sequential.forward(input);
}