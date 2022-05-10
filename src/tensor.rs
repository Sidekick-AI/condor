use std::ops::{Add, Sub, Mul, Div};

// #[derive(Eq)]
// pub enum DimType {
//     Static(usize),
//     Dynamic
// }

// impl PartialEq for DimType {
//     fn eq(&self, other: &Self) -> bool {
//         match self {
//             DimType::Static(s) => match other {
//                 DimType::Static(t) => s == t,
//                 DimType::Dynamic => true,
//             },
//             DimType::Dynamic => true,
//         }
//     }
// }

#[derive(Debug)]
pub struct Tensor<const D1: u16, const D2: u16 = { 1 }, const D3: u16 = { 1 }, const D4: u16 = { 1 }, const D5: u16 = { 1 }> {
    pub data: tch::Tensor,    
}

impl<const D1: u16, const D2: u16, const D3: u16, const D4: u16, const D5: u16> Default for Tensor<D1, D2, D3, D4, D5> {
    fn default() -> Self {
        Tensor {
            data: tch::Tensor::new()
        }
    }
}

impl<const D1: u16, const D2: u16, const D3: u16, const D4: u16, const D5: u16> Clone for Tensor<D1, D2, D3, D4, D5> {
    fn clone(&self) -> Self {
        Self { data: self.data.shallow_clone() }
    }
}

impl <const D1: u16, const D2: u16, const D3: u16, const D4: u16, const D5: u16>Tensor<D1, D2, D3, D4, D5> {
    pub fn from_arr<T: tch::kind::Element>(data: &[T]) -> Self {
        Tensor {
            data: tch::Tensor::of_slice(data)
        }
    }

    pub fn from_arr_2d<T: tch::kind::Element, U: AsRef<[T]>>(data: &[U]) -> Self {
        Tensor {
            data: tch::Tensor::of_slice2(data)
        }
    }

    pub fn from_tch(data: tch::Tensor) -> Self {
        Tensor {
            data
        }
    }

    /// Matrix multiply
    pub fn matmul<const N: u16>(&self, rhs: &Tensor<N, D2, D3, D4, D5>) -> Tensor<D1, D2, D3, D4, D5> {
        Tensor::from_tch(self.data.matmul(&rhs.data))
    }
}

// Operators
impl <const D1: u16, const D2: u16, const D3: u16, const D4: u16, const D5: u16>Add<&Tensor<D1, D2, D3, D4, D5>> for &Tensor<D1, D2, D3, D4, D5> {
    type Output = Tensor<D1, D2, D3, D4, D5>;

    fn add(self, rhs: &Tensor<D1, D2, D3, D4, D5>) -> Self::Output {
        Tensor::from_tch(self.data.shallow_clone() + rhs.data.shallow_clone())
    }
}

impl <const D1: u16, const D2: u16, const D3: u16, const D4: u16, const D5: u16>Sub<&Tensor<D1, D2, D3, D4, D5>> for &Tensor<D1, D2, D3, D4, D5> {
    type Output = Tensor<D1, D2, D3, D4, D5>;

    fn sub(self, rhs: &Tensor<D1, D2, D3, D4, D5>) -> Self::Output {
        Tensor::from_tch(self.data.shallow_clone() - rhs.data.shallow_clone())
    }
}

impl <const D1: u16, const D2: u16, const D3: u16, const D4: u16, const D5: u16>Mul<&Tensor<D1, D2, D3, D4, D5>> for &Tensor<D1, D2, D3, D4, D5> {
    type Output = Tensor<D1, D2, D3, D4, D5>;

    fn mul(self, rhs: &Tensor<D1, D2, D3, D4, D5>) -> Self::Output {
        Tensor::from_tch(self.data.shallow_clone() * rhs.data.shallow_clone())
    }
}

impl <const D1: u16, const D2: u16, const D3: u16, const D4: u16, const D5: u16>Div<&Tensor<D1, D2, D3, D4, D5>> for &Tensor<D1, D2, D3, D4, D5> {
    type Output = Tensor<D1, D2, D3, D4, D5>;

    fn div(self, rhs: &Tensor<D1, D2, D3, D4, D5>) -> Self::Output {
        Tensor::from_tch(self.data.shallow_clone() / rhs.data.shallow_clone())
    }
}