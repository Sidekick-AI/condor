use crate::tensor::*;

pub trait MatMul<T> {
    type Output;

    fn matmul(&self, other: &T) -> Self::Output;
}

// 1 x 2
impl <const D1: u16, const D2: u16>MatMul<Tensor2<D1, D2>> for Tensor1<D1> {
    default type Output = Tensor2<D1, D2>;

    default fn matmul(&self, other: &Tensor2<D1, D2>) -> Self::Output {
        todo!()
    }
}

// 1 x 1
impl <const D1: u16>MatMul<Tensor1<D1>> for Tensor1<D1> {
    type Output = Tensor1<D1>;

    fn matmul(&self, other: &Tensor1<D1>) -> Self::Output {
        todo!()
    }
}

// 2 x 1
impl <const D1: u16, const D2: u16>MatMul<Tensor1<D1>> for Tensor2<D1, D2> {
    type Output = Tensor2<D1, D2>;

    fn matmul(&self, other: &Tensor1<D1>) -> Self::Output {
        todo!()
    }
}

// 2 x 2
impl <const D1: u16, const D2: u16, const D3: u16>MatMul<Tensor2<D2, D3>> for Tensor2<D1, D2> {
    type Output = Tensor2<D1, D3>;

    fn matmul(&self, other: &Tensor2<D2, D3>) -> Self::Output {
        todo!()
    }
}