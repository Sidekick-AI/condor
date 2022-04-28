mod ops;
use derive_new::new;
pub use ops::*;
use tch::{Kind, Device, Tensor as TchTensor};

#[derive(Debug, new)]
pub struct Tensor1<const D1: u16> {
    pub data: TchTensor
}

impl <const D1: u16>Tensor1<D1> {
    pub fn from_arr(arr: &[f32; D1 as usize]) -> Self {
        Self {
            data: TchTensor::of_slice(arr)
        }
    }

    pub fn rand() -> Self {
        Self { data: TchTensor::rand(&[D1 as i64], (Kind::Float, Device::Cpu)) }
    }
}

impl <const D1: u16>Clone for Tensor1<D1> {
    fn clone(&self) -> Self {
        Self { data: self.data.shallow_clone() }
    }
}

#[derive(Debug, new)]
pub struct Tensor2<const D1: u16, const D2: u16> {
    pub data: TchTensor
}

impl <const D1: u16, const D2: u16>Tensor2<D1, D2> {
    pub fn from_arr(arr: &[[f32; D2 as usize]; D1 as usize]) -> Self {
        Self {
            data: TchTensor::of_slice2(arr)
        }
    }

    pub fn rand() -> Self {
        Self { data: TchTensor::rand(&[D1 as i64, D2 as i64], (Kind::Float, Device::Cpu)) }
    }
}

impl <const D1: u16, const D2: u16>Clone for Tensor2<D1, D2> {
    fn clone(&self) -> Self {
        Self { data: self.data.shallow_clone() }
    }
}

#[derive(Debug, new)]
pub struct Tensor3<const D1: u16, const D2: u16, const D3: u16> {
    pub data: TchTensor
}

impl <const D1: u16, const D2: u16, const D3: u16>Tensor3<D1, D2, D3> {
    pub fn rand() -> Self {
        Self { data: TchTensor::rand(&[D1 as i64, D2 as i64, D3 as i64], (Kind::Float, Device::Cpu)) }
    }
}

impl <const D1: u16, const D2: u16, const D3: u16>Clone for Tensor3<D1, D2, D3> {
    fn clone(&self) -> Self {
        Self { data: self.data.shallow_clone() }
    }
}

// #[derive(Debug, Default, Clone)]
// pub struct Tensor4<const D1: u16, const D2: u16, const D3: u16, const D4: u16> {}

// #[derive(Debug, Default, Clone)]
// pub struct Tensor5<const D1: u16, const D2: u16, const D3: u16, const D4: u16, const D5: u16> {}
