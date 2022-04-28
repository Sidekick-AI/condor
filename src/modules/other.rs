use derive_new::new;

use crate::{Module, Tensor1, Tensor2, Tensor3};

// #[derive(new)]
// pub struct Dropout5<const D1: u16, const D2: u16, const D3: u16, const D4: u16, const D5: u16> {
//     p: f64
// }

// impl <const D1: u16, const D2: u16, const D3: u16, const D4: u16, const D5: u16>Module for Dropout5<D1, D2, D3, D4, D5> {
//     type Input = Tensor5<D1, D2, D3, D4, D5>;
//     type Output = Tensor5<D1, D2, D3, D4, D5>;

//     fn forward(&self, input: Self::Input) -> Self::Output {
//         input * self.p
//     }
// }

// #[derive(new)]
// pub struct Dropout4<const D1: u16, const D2: u16, const D3: u16, const D4: u16> {
//     p: f64
// }

// impl <const D1: u16, const D2: u16, const D3: u16, const D4: u16>Module for Dropout4<D1, D2, D3, D4> {
//     type Input = Tensor4<D1, D2, D3, D4>;
//     type Output = Tensor4<D1, D2, D3, D4>;

//     fn forward(&self, input: Self::Input) -> Self::Output {
//         input * self.p
//     }
// }

#[derive(new)]
pub struct Dropout3<const D1: u16, const D2: u16, const D3: u16> {
    p: f32
}

impl <const D1: u16, const D2: u16, const D3: u16>Module for Dropout3<D1, D2, D3> {
    type Input = Tensor3<D1, D2, D3>;
    type Output = Tensor3<D1, D2, D3>;

    fn forward(&self, input: Self::Input) -> Self::Output {
        input * self.p
    }
}

#[derive(new)]
pub struct Dropout2<const D1: u16, const D2: u16> {
    p: f32
}

impl <const D1: u16, const D2: u16>Module for Dropout2<D1, D2> {
    type Input = Tensor2<D1, D2>;
    type Output = Tensor2<D1, D2>;

    fn forward(&self, input: Self::Input) -> Self::Output {
        input * self.p
    }
}

#[derive(new)]
pub struct Dropout1<const D1: u16> {
    p: f32
}

impl <const D1: u16>Module for Dropout1<D1> {
    type Input = Tensor1<D1>;
    type Output = Tensor1<D1>;

    fn forward(&self, input: Self::Input) -> Self::Output {
        input * self.p
    }
}