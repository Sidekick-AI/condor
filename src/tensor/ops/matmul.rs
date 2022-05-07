use crate::tensor::*;

pub trait MatMul<T> {
    type Output;

    fn matmul(&self, other: &T) -> Self::Output;
}

// 1 x 2
impl <const D1: u16, const D2: u16>MatMul<Tensor2<D1, D2>> for Tensor1<D1> {
    type Output = Tensor2<D1, D2>;

    fn matmul(&self, other: &Tensor2<D1, D2>) -> Self::Output {
        Tensor2::new(self.data.matmul(&other.data))
    }
}

// 1 x 1
impl <const D1: u16>MatMul<Tensor1<D1>> for Tensor1<D1> {
    type Output = Tensor1<D1>;

    fn matmul(&self, other: &Tensor1<D1>) -> Self::Output {
        Tensor1::new(self.data.matmul(&other.data))
    }
}

// 2 x 1
impl <const D1: u16, const D2: u16>MatMul<Tensor1<D1>> for Tensor2<D1, D2> {
    type Output = Tensor2<D1, D2>;

    fn matmul(&self, other: &Tensor1<D1>) -> Self::Output {
        Tensor2::new(self.data.matmul(&other.data))
    }
}

// 2 x 2
impl <const D1: u16, const D2: u16, const D3: u16>MatMul<Tensor2<D2, D3>> for Tensor2<D1, D2> {
    type Output = Tensor2<D1, D3>;

    fn matmul(&self, other: &Tensor2<D2, D3>) -> Self::Output {
        Tensor2::new(self.data.matmul(&other.data))
    }
}

// 2 x 3 (batch matrix multiply across first dim)
// impl <const D1: u16, const D2: u16, const D3: u16>MatMul<Tensor3<D1, D2, D3>> for Tensor2<D1, D2> {
//     type Output = Tensor2<D1, D3>;

//     fn matmul(&self, other: &Tensor3<D1, D2, D3>) -> Self::Output {
//         Tensor2::new(self.data.matmul(&other.data))
//     }
// }