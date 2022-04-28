use std::ops::{Add, Sub, Mul, Div};
use crate::tensor::*;

// Element-wise addition
impl <const D1: u16>Add<&Tensor1<D1>> for &Tensor1<D1> {
    type Output = Tensor1<D1>;

    fn add(self, rhs: &Tensor1<D1>) -> Self::Output {
        let mut new = self.clone();
        new.data += &rhs.data;
        new
    }
}

impl <const D1: u16, const D2: u16>Add<&Tensor2<D1, D2>> for &Tensor2<D1, D2> {
    type Output = Tensor2<D1, D2>;

    fn add(self, rhs: &Tensor2<D1, D2>) -> Self::Output {
        let mut new = self.clone();
        new.data += &rhs.data;
        new
    }
}

// Broadcast across first dim
impl <const D1: u16, const D2: u16>Add<&Tensor1<D2>> for &Tensor2<D1, D2> {
    type Output = Tensor2<D1, D2>;

    fn add(self, rhs: &Tensor1<D2>) -> Self::Output {
        let mut new = self.clone();
        new.data += &rhs.data;
        new
    }
}

impl <const D1: u16, const D2: u16, const D3: u16>Add<&Tensor3<D1, D2, D3>> for &Tensor3<D1, D2, D3> {
    type Output = Tensor3<D1, D2, D3>;

    fn add(self, rhs: &Tensor3<D1, D2, D3>) -> Self::Output {
        let mut new = self.clone();
        new.data += &rhs.data;
        new
    }
}

// Element-wise subtraction
impl <const D1: u16>Sub<&Tensor1<D1>> for &Tensor1<D1> {
    type Output = Tensor1<D1>;

    fn sub(self, rhs: &Tensor1<D1>) -> Self::Output {
        let mut new = self.clone();
        new.data -= &rhs.data;
        new
    }
}

impl <const D1: u16, const D2: u16>Sub<&Tensor2<D1, D2>> for &Tensor2<D1, D2> {
    type Output = Tensor2<D1, D2>;

    fn sub(self, rhs: &Tensor2<D1, D2>) -> Self::Output {
        let mut new = self.clone();
        new.data -= &rhs.data;
        new
    }
}

impl <const D1: u16, const D2: u16, const D3: u16>Sub<&Tensor3<D1, D2, D3>> for &Tensor3<D1, D2, D3> {
    type Output = Tensor3<D1, D2, D3>;

    fn sub(self, rhs: &Tensor3<D1, D2, D3>) -> Self::Output {
        let mut new = self.clone();
        new.data -= &rhs.data;
        new
    }
}

// Element-wise multiplication
impl <const D1: u16>Mul<&Tensor1<D1>> for &Tensor1<D1> {
    type Output = Tensor1<D1>;

    fn mul(self, rhs: &Tensor1<D1>) -> Self::Output {
        let mut new = self.clone();
        new.data *= &rhs.data;
        new
    }
}

impl <const D1: u16, const D2: u16>Mul<&Tensor2<D1, D2>> for &Tensor2<D1, D2> {
    type Output = Tensor2<D1, D2>;

    fn mul(self, rhs: &Tensor2<D1, D2>) -> Self::Output {
        let mut new = self.clone();
        new.data *= &rhs.data;
        new
    }
}

impl <const D1: u16, const D2: u16, const D3: u16>Mul<&Tensor3<D1, D2, D3>> for &Tensor3<D1, D2, D3> {
    type Output = Tensor3<D1, D2, D3>;

    fn mul(self, rhs: &Tensor3<D1, D2, D3>) -> Self::Output {
        let mut new = self.clone();
        new.data *= &rhs.data;
        new
    }
}

// Element-wise division
impl <const D1: u16>Div<&Tensor1<D1>> for &Tensor1<D1> {
    type Output = Tensor1<D1>;

    fn div(self, rhs: &Tensor1<D1>) -> Self::Output {
        let mut new = self.clone();
        new.data /= &rhs.data;
        new
    }
}

impl <const D1: u16, const D2: u16>Div<&Tensor2<D1, D2>> for &Tensor2<D1, D2> {
    type Output = Tensor2<D1, D2>;

    fn div(self, rhs: &Tensor2<D1, D2>) -> Self::Output {
        let mut new = self.clone();
        new.data /= &rhs.data;
        new
    }
}

impl <const D1: u16, const D2: u16, const D3: u16>Div<&Tensor3<D1, D2, D3>> for &Tensor3<D1, D2, D3> {
    type Output = Tensor3<D1, D2, D3>;

    fn div(self, rhs: &Tensor3<D1, D2, D3>) -> Self::Output {
        let mut new = self.clone();
        new.data /= &rhs.data;
        new
    }
}