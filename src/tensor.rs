use std::ops::{Add, Sub, Mul, Div};

#[derive(Eq)]
pub enum DimType {
    Static(usize),
    Dynamic
}

impl PartialEq for DimType {
    fn eq(&self, other: &Self) -> bool {
        match self {
            DimType::Static(s) => match other {
                DimType::Static(t) => s == t,
                DimType::Dynamic => true,
            },
            DimType::Dynamic => true,
        }
    }
}

#[derive(Debug)]
pub struct Tensor<const DIM1: DimType, const DIM2: DimType = { DimType::Static(1) }, const DIM3: DimType = { DimType::Static(1) }, const DIM4: DimType = { DimType::Static(1) }, const DIM5: DimType = { DimType::Static(1) }> {
    pub data: tch::Tensor,    
}

impl<const DIM1: DimType, const DIM2: DimType, const DIM3: DimType, const DIM4: DimType, const DIM5: DimType> Default for Tensor<DIM1, DIM2, DIM3, DIM4, DIM5> {
    fn default() -> Self {
        Tensor {
            data: tch::Tensor::new()
        }
    }
}

impl<const DIM1: DimType, const DIM2: DimType, const DIM3: DimType, const DIM4: DimType, const DIM5: DimType> Clone for Tensor<DIM1, DIM2, DIM3, DIM4, DIM5> {
    fn clone(&self) -> Self {
        Self { data: self.data.shallow_clone() }
    }
}

impl <const DIM1: DimType, const DIM2: DimType, const DIM3: DimType, const DIM4: DimType, const DIM5: DimType>Tensor<DIM1, DIM2, DIM3, DIM4, DIM5> {
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
    pub fn matmul<const N: DimType>(&self, rhs: &Tensor<N, DIM2, DIM3, DIM4, DIM5>) -> Tensor<DIM1, DIM2, DIM3, DIM4, DIM5> {
        Tensor::from_tch(self.data.matmul(&rhs.data))
    }
}

// Operators
impl <const DIM1: DimType, const DIM2: DimType, const DIM3: DimType, const DIM4: DimType, const DIM5: DimType>Add<&Tensor<DIM1, DIM2, DIM3, DIM4, DIM5>> for &Tensor<DIM1, DIM2, DIM3, DIM4, DIM5> {
    type Output = Tensor<DIM1, DIM2, DIM3, DIM4, DIM5>;

    fn add(self, rhs: &Tensor<DIM1, DIM2, DIM3, DIM4, DIM5>) -> Self::Output {
        Tensor::from_tch(self.data.shallow_clone() + rhs.data.shallow_clone())
    }
}

impl <const DIM1: DimType, const DIM2: DimType, const DIM3: DimType, const DIM4: DimType, const DIM5: DimType>Sub<&Tensor<DIM1, DIM2, DIM3, DIM4, DIM5>> for &Tensor<DIM1, DIM2, DIM3, DIM4, DIM5> {
    type Output = Tensor<DIM1, DIM2, DIM3, DIM4, DIM5>;

    fn sub(self, rhs: &Tensor<DIM1, DIM2, DIM3, DIM4, DIM5>) -> Self::Output {
        Tensor::from_tch(self.data.shallow_clone() - rhs.data.shallow_clone())
    }
}

impl <const DIM1: DimType, const DIM2: DimType, const DIM3: DimType, const DIM4: DimType, const DIM5: DimType>Mul<&Tensor<DIM1, DIM2, DIM3, DIM4, DIM5>> for &Tensor<DIM1, DIM2, DIM3, DIM4, DIM5> {
    type Output = Tensor<DIM1, DIM2, DIM3, DIM4, DIM5>;

    fn mul(self, rhs: &Tensor<DIM1, DIM2, DIM3, DIM4, DIM5>) -> Self::Output {
        Tensor::from_tch(self.data.shallow_clone() * rhs.data.shallow_clone())
    }
}

impl <const DIM1: DimType, const DIM2: DimType, const DIM3: DimType, const DIM4: DimType, const DIM5: DimType>Div<&Tensor<DIM1, DIM2, DIM3, DIM4, DIM5>> for &Tensor<DIM1, DIM2, DIM3, DIM4, DIM5> {
    type Output = Tensor<DIM1, DIM2, DIM3, DIM4, DIM5>;

    fn div(self, rhs: &Tensor<DIM1, DIM2, DIM3, DIM4, DIM5>) -> Self::Output {
        Tensor::from_tch(self.data.shallow_clone() / rhs.data.shallow_clone())
    }
}