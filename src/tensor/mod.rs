mod ops;

#[cfg(test)]
mod tests;

use derive_new::new;
pub use ops::*;
use tch::{Kind, Device, Tensor as TchTensor, IndexOp};

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

    pub fn from_tch(tensor: TchTensor) -> Self {
        Self {
            data: tensor
        }
    }

    pub fn rand() -> Self {
        Self { data: TchTensor::rand(&[D1 as i64], (Kind::Float, Device::Cpu)) }
    }

    pub fn zeros() -> Self {
        Self { data: TchTensor::zeros(&[D1 as i64], (Kind::Float, Device::Cpu)) }
    }

    pub fn get(&self, index: u16) -> Tensor1<1> {
        Tensor1::from_tch(self.data.get(index as i64))
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

    pub fn from_tch(tensor: TchTensor) -> Self {
        Self {
            data: tensor
        }
    }

    pub fn rand() -> Self {
        Self { data: TchTensor::rand(&[D1 as i64, D2 as i64], (Kind::Float, Device::Cpu)) }
    }

    pub fn zeros() -> Self {
        Self { data: TchTensor::zeros(&[D1 as i64, D2 as i64], (Kind::Float, Device::Cpu)) }
    }

    pub fn get(&self, index: u16) -> Tensor1<D2> {
        Tensor1::from_tch(self.data.get(index as i64))
    }

    pub fn set(&mut self, index: u16, value: &Tensor1<D2>) {
        self.data.i(index as i64).copy_(&value.data);
    }

    pub fn cat_1<const D3: u16>(&mut self, value: &Tensor2<D3, D2>) -> Tensor2<{D1 + D3}, D2> {
        Tensor2::from_tch(tch::Tensor::cat(&[self.data, value.data], 0))
    }

    pub fn cat_2<const D3: u16>(&mut self, value: &Tensor2<D1, D3>) -> Tensor2<D1, {D2 + D3}> {
        Tensor2::from_tch(tch::Tensor::cat(&[self.data, value.data], 1))
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

    pub fn zeros() -> Self {
        Self { data: TchTensor::zeros(&[D1 as i64, D2 as i64, D3 as i64], (Kind::Float, Device::Cpu)) }
    }

    pub fn from_tch(tensor: TchTensor) -> Self {
        Self {
            data: tensor
        }
    }

    pub fn get(&self, index: u16) -> Tensor2<D2, D3> {
        Tensor2::from_tch(self.data.get(index as i64))
    }

    pub fn set(&mut self, index: u16, value: &Tensor2<D2, D3>) {
        self.data.i(index as i64).copy_(&value.data);
    }

    pub fn cat_1<const D4: u16>(&mut self, value: &Tensor3<D4, D2, D3>) -> Tensor3<{D1 + D4}, D2, D3> {
        Tensor3::from_tch(tch::Tensor::cat(&[self.data, value.data], 0))
    }

    pub fn cat_2<const D4: u16>(&mut self, value: &Tensor3<D1, D4, D3>) -> Tensor3<D1, {D2 + D4}, D3> {
        Tensor3::from_tch(tch::Tensor::cat(&[self.data, value.data], 1))
    }

    pub fn cat_3<const D4: u16>(&mut self, value: &Tensor3<D1, D2, D4>) -> Tensor3<D1, D2, {D3 + D4}> {
        Tensor3::from_tch(tch::Tensor::cat(&[self.data, value.data], 1))
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
