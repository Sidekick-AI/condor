use std::ops::{Add, Sub, Mul, Div};
use crate::tensor::*;

// Scalar addition
impl <const D1: u16>Add<f32> for &Tensor1<D1> {
    type Output = Tensor1<D1>;

    fn add(self, rhs: f32) -> Self::Output {
        todo!()
    }
}

impl <const D1: u16, const D2: u16>Add<f32> for &Tensor2<D1, D2> {
    type Output = Tensor2<D1, D2>;

    fn add(self, rhs: f32) -> Self::Output {
        todo!()
    }
}

impl <const D1: u16, const D2: u16, const D3: u16>Add<f32> for &Tensor3<D1, D2, D3> {
    type Output = Tensor3<D1, D2, D3>;

    fn add(self, rhs: f32) -> Self::Output {
        todo!()
    }
}

impl <const D1: u16>Add<f32> for Tensor1<D1> {
    type Output = Tensor1<D1>;

    fn add(self, rhs: f32) -> Self::Output {
        todo!()
    }
}

impl <const D1: u16, const D2: u16>Add<f32> for Tensor2<D1, D2> {
    type Output = Tensor2<D1, D2>;

    fn add(self, rhs: f32) -> Self::Output {
        todo!()
    }
}

impl <const D1: u16, const D2: u16, const D3: u16>Add<f32> for Tensor3<D1, D2, D3> {
    type Output = Tensor3<D1, D2, D3>;

    fn add(self, rhs: f32) -> Self::Output {
        todo!()
    }
}

// Scalar subtraction
impl <const D1: u16>Sub<f32> for &Tensor1<D1> {
    type Output = Tensor1<D1>;

    fn sub(self, rhs: f32) -> Self::Output {
        todo!()
    }
}

impl <const D1: u16, const D2: u16>Sub<f32> for &Tensor2<D1, D2> {
    type Output = Tensor2<D1, D2>;

    fn sub(self, rhs: f32) -> Self::Output {
        todo!()
    }
}

impl <const D1: u16, const D2: u16, const D3: u16>Sub<f32> for &Tensor3<D1, D2, D3> {
    type Output = Tensor3<D1, D2, D3>;

    fn sub(self, rhs: f32) -> Self::Output {
        todo!()
    }
}

impl <const D1: u16>Sub<f32> for Tensor1<D1> {
    type Output = Tensor1<D1>;

    fn sub(self, rhs: f32) -> Self::Output {
        todo!()
    }
}

impl <const D1: u16, const D2: u16>Sub<f32> for Tensor2<D1, D2> {
    type Output = Tensor2<D1, D2>;

    fn sub(self, rhs: f32) -> Self::Output {
        todo!()
    }
}

impl <const D1: u16, const D2: u16, const D3: u16>Sub<f32> for Tensor3<D1, D2, D3> {
    type Output = Tensor3<D1, D2, D3>;

    fn sub(self, rhs: f32) -> Self::Output {
        todo!()
    }
}

// Scalar multiplication
impl <const D1: u16>Mul<f32> for &Tensor1<D1> {
    type Output = Tensor1<D1>;

    fn mul(self, rhs: f32) -> Self::Output {
        todo!()
    }
}

impl <const D1: u16, const D2: u16>Mul<f32> for &Tensor2<D1, D2> {
    type Output = Tensor2<D1, D2>;

    fn mul(self, rhs: f32) -> Self::Output {
        todo!()
    }
}

impl <const D1: u16, const D2: u16, const D3: u16>Mul<f32> for &Tensor3<D1, D2, D3> {
    type Output = Tensor3<D1, D2, D3>;

    fn mul(self, rhs: f32) -> Self::Output {
        todo!()
    }
}

impl <const D1: u16>Mul<f32> for Tensor1<D1> {
    type Output = Tensor1<D1>;

    fn mul(self, rhs: f32) -> Self::Output {
        todo!()
    }
}

impl <const D1: u16, const D2: u16>Mul<f32> for Tensor2<D1, D2> {
    type Output = Tensor2<D1, D2>;

    fn mul(self, rhs: f32) -> Self::Output {
        todo!()
    }
}

impl <const D1: u16, const D2: u16, const D3: u16>Mul<f32> for Tensor3<D1, D2, D3> {
    type Output = Tensor3<D1, D2, D3>;

    fn mul(self, rhs: f32) -> Self::Output {
        todo!()
    }
}

// Scalar division
impl <const D1: u16>Div<f32> for &Tensor1<D1> {
    type Output = Tensor1<D1>;

    fn div(self, rhs: f32) -> Self::Output {
        todo!()
    }
}

impl <const D1: u16, const D2: u16>Div<f32> for &Tensor2<D1, D2> {
    type Output = Tensor2<D1, D2>;

    fn div(self, rhs: f32) -> Self::Output {
        todo!()
    }
}

impl <const D1: u16, const D2: u16, const D3: u16>Div<f32> for &Tensor3<D1, D2, D3> {
    type Output = Tensor3<D1, D2, D3>;

    fn div(self, rhs: f32) -> Self::Output {
        todo!()
    }
}

impl <const D1: u16>Div<f32> for Tensor1<D1> {
    type Output = Tensor1<D1>;

    fn div(self, rhs: f32) -> Self::Output {
        todo!()
    }
}

impl <const D1: u16, const D2: u16>Div<f32> for Tensor2<D1, D2> {
    type Output = Tensor2<D1, D2>;

    fn div(self, rhs: f32) -> Self::Output {
        todo!()
    }
}

impl <const D1: u16, const D2: u16, const D3: u16>Div<f32> for Tensor3<D1, D2, D3> {
    type Output = Tensor3<D1, D2, D3>;

    fn div(self, rhs: f32) -> Self::Output {
        todo!()
    }
}