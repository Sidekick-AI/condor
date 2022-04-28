use crate::{Tensor1, Tensor2, Tensor3, Module};

// Tensor impls
// impl <const D1: u16, const D2: u16, const D3: u16, const D4: u16, const D5: u16>Tensor5<D1, D2, D3, D4, D5> {
//     pub fn relu(&self) -> Self {
//         todo!()
//     }

//     pub fn gelu(&self) -> Self {
//         todo!()
//     }

//     pub fn sigmoid(&self) -> Self {
//         todo!()
//     }

//     pub fn prelu(&self, weight: &Tensor1<1>) -> Self {
//         todo!()
//     }
// }

// impl <const D1: u16, const D2: u16, const D3: u16, const D4: u16>Tensor4<D1, D2, D3, D4> {
//     pub fn relu(&self) -> Self {
//         todo!()
//     }

//     pub fn gelu(&self) -> Self {
//         todo!()
//     }

//     pub fn sigmoid(&self) -> Self {
//         todo!()
//     }

//     pub fn prelu(&self, weight: &Tensor1<1>) -> Self {
//         todo!()
//     }
// }

impl <const D1: u16, const D2: u16, const D3: u16>Tensor3<D1, D2, D3> {
    pub fn relu(&self) -> Self {
        Self::new(self.data.relu())
    }

    pub fn gelu(&self) -> Self {
        Self::new(self.data.gelu())
    }

    pub fn sigmoid(&self) -> Self {
        Self::new(self.data.sigmoid())
    }

    pub fn prelu(&self, weight: &Tensor1<1>) -> Self {
        Self::new(self.data.prelu(&weight.data))
    }
}

impl <const D1: u16, const D2: u16>Tensor2<D1, D2> {
    pub fn relu(&self) -> Self {
        Self::new(self.data.relu())
    }

    pub fn gelu(&self) -> Self {
        Self::new(self.data.gelu())
    }

    pub fn sigmoid(&self) -> Self {
        Self::new(self.data.sigmoid())
    }

    pub fn prelu(&self, weight: &Tensor1<1>) -> Self {
        Self::new(self.data.prelu(&weight.data))
    }
}

impl <const D1: u16>Tensor1<D1> {
    pub fn relu(&self) -> Self {
        Self::new(self.data.relu())
    }

    pub fn gelu(&self) -> Self {
        Self::new(self.data.gelu())
    }

    pub fn sigmoid(&self) -> Self {
        Self::new(self.data.sigmoid())
    }

    pub fn prelu(&self, weight: &Tensor1<1>) -> Self {
        Self::new(self.data.prelu(&weight.data))
    }
}

// Activation Modules
#[derive(Default)]
pub struct ReLU1<const D1: u16> {}

impl <const D1: u16>Module for ReLU1<D1> {
    type Input = Tensor1<D1>;
    type Output = Tensor1<D1>;

    fn forward(&self, input: Self::Input) -> Self::Output {
        input.relu()
    }
}

#[derive(Default)]
pub struct ReLU2<const D1: u16, const D2: u16> {}

impl <const D1: u16, const D2: u16>Module for ReLU2<D1, D2> {
    type Input = Tensor2<D1, D2>;
    type Output = Tensor2<D1, D2>;

    fn forward(&self, input: Self::Input) -> Self::Output {
        input.relu()
    }
}

#[derive(Default)]
pub struct ReLU3<const D1: u16, const D2: u16, const D3: u16> {}

impl <const D1: u16, const D2: u16, const D3: u16>Module for ReLU3<D1, D2, D3> {
    type Input = Tensor3<D1, D2, D3>;
    type Output = Tensor3<D1, D2, D3>;

    fn forward(&self, input: Self::Input) -> Self::Output {
        input.relu()
    }
}

// #[derive(Default)]
// pub struct ReLU4<const D1: u16, const D2: u16, const D3: u16, const D4: u16> {}

// impl <const D1: u16, const D2: u16, const D3: u16, const D4: u16>Module for ReLU4<D1, D2, D3, D4> {
//     type Input = Tensor4<D1, D2, D3, D4>;
//     type Output = Tensor4<D1, D2, D3, D4>;

//     fn forward(&self, input: Self::Input) -> Self::Output {
//         input.relu()
//     }
// }

// #[derive(Default)]
// pub struct ReLU5<const D1: u16, const D2: u16, const D3: u16, const D4: u16, const D5: u16> {}

// impl <const D1: u16, const D2: u16, const D3: u16, const D4: u16, const D5: u16>Module for ReLU5<D1, D2, D3, D4, D5> {
//     type Input = Tensor5<D1, D2, D3, D4, D5>;
//     type Output = Tensor5<D1, D2, D3, D4, D5>;

//     fn forward(&self, input: Self::Input) -> Self::Output {
//         input.relu()
//     }
// }

#[derive(Default)]
pub struct GeLU1<const D1: u16> {}

impl <const D1: u16>Module for GeLU1<D1> {
    type Input = Tensor1<D1>;
    type Output = Tensor1<D1>;

    fn forward(&self, input: Self::Input) -> Self::Output {
        input.gelu()
    }
}

#[derive(Default)]
pub struct GeLU2<const D1: u16, const D2: u16> {}

impl <const D1: u16, const D2: u16>Module for GeLU2<D1, D2> {
    type Input = Tensor2<D1, D2>;
    type Output = Tensor2<D1, D2>;

    fn forward(&self, input: Self::Input) -> Self::Output {
        input.gelu()
    }
}

#[derive(Default)]
pub struct GeLU3<const D1: u16, const D2: u16, const D3: u16> {}

impl <const D1: u16, const D2: u16, const D3: u16>Module for GeLU3<D1, D2, D3> {
    type Input = Tensor3<D1, D2, D3>;
    type Output = Tensor3<D1, D2, D3>;

    fn forward(&self, input: Self::Input) -> Self::Output {
        input.gelu()
    }
}

// #[derive(Default)]
// pub struct GeLU4<const D1: u16, const D2: u16, const D3: u16, const D4: u16> {}

// impl <const D1: u16, const D2: u16, const D3: u16, const D4: u16>Module for GeLU4<D1, D2, D3, D4> {
//     type Input = Tensor4<D1, D2, D3, D4>;
//     type Output = Tensor4<D1, D2, D3, D4>;

//     fn forward(&self, input: Self::Input) -> Self::Output {
//         input.gelu()
//     }
// }

// #[derive(Default)]
// pub struct GeLU5<const D1: u16, const D2: u16, const D3: u16, const D4: u16, const D5: u16> {}

// impl <const D1: u16, const D2: u16, const D3: u16, const D4: u16, const D5: u16>Module for GeLU5<D1, D2, D3, D4, D5> {
//     type Input = Tensor5<D1, D2, D3, D4, D5>;
//     type Output = Tensor5<D1, D2, D3, D4, D5>;

//     fn forward(&self, input: Self::Input) -> Self::Output {
//         input.gelu()
//     }
// }

#[derive(Default)]
pub struct Sigmoid1<const D1: u16> {}

impl <const D1: u16>Module for Sigmoid1<D1> {
    type Input = Tensor1<D1>;
    type Output = Tensor1<D1>;

    fn forward(&self, input: Self::Input) -> Self::Output {
        input.sigmoid()
    }
}

#[derive(Default)]
pub struct Sigmoid2<const D1: u16, const D2: u16> {}

impl <const D1: u16, const D2: u16>Module for Sigmoid2<D1, D2> {
    type Input = Tensor2<D1, D2>;
    type Output = Tensor2<D1, D2>;

    fn forward(&self, input: Self::Input) -> Self::Output {
        input.sigmoid()
    }
}

#[derive(Default)]
pub struct Sigmoid3<const D1: u16, const D2: u16, const D3: u16> {}

impl <const D1: u16, const D2: u16, const D3: u16>Module for Sigmoid3<D1, D2, D3> {
    type Input = Tensor3<D1, D2, D3>;
    type Output = Tensor3<D1, D2, D3>;

    fn forward(&self, input: Self::Input) -> Self::Output {
        input.sigmoid()
    }
}

// #[derive(Default)]
// pub struct Sigmoid4<const D1: u16, const D2: u16, const D3: u16, const D4: u16> {}

// impl <const D1: u16, const D2: u16, const D3: u16, const D4: u16>Module for Sigmoid4<D1, D2, D3, D4> {
//     type Input = Tensor4<D1, D2, D3, D4>;
//     type Output = Tensor4<D1, D2, D3, D4>;

//     fn forward(&self, input: Self::Input) -> Self::Output {
//         input.sigmoid()
//     }
// }

// #[derive(Default)]
// pub struct Sigmoid5<const D1: u16, const D2: u16, const D3: u16, const D4: u16, const D5: u16> {}

// impl <const D1: u16, const D2: u16, const D3: u16, const D4: u16, const D5: u16>Module for Sigmoid5<D1, D2, D3, D4, D5> {
//     type Input = Tensor5<D1, D2, D3, D4, D5>;
//     type Output = Tensor5<D1, D2, D3, D4, D5>;

//     fn forward(&self, input: Self::Input) -> Self::Output {
//         input.sigmoid()
//     }
// }

pub struct PReLU1<const D1: u16> {
    weight: Tensor1<1>
}

impl <const D1: u16>Default for PReLU1<D1> {
    fn default() -> Self {
        Self { weight: Tensor1::rand() }
    }
}

impl <const D1: u16>Module for PReLU1<D1> {
    type Input = Tensor1<D1>;
    type Output = Tensor1<D1>;

    fn forward(&self, input: Self::Input) -> Self::Output {
        input.prelu(&self.weight)
    }
}

pub struct PReLU2<const D1: u16, const D2: u16> {
    weight: Tensor1<1>
}

impl <const D1: u16, const D2: u16>Default for PReLU2<D1, D2> {
    fn default() -> Self {
        Self { weight: Tensor1::rand() }
    }
}

impl <const D1: u16, const D2: u16>Module for PReLU2<D1, D2> {
    type Input = Tensor2<D1, D2>;
    type Output = Tensor2<D1, D2>;

    fn forward(&self, input: Self::Input) -> Self::Output {
        input.prelu(&self.weight)
    }
}

pub struct PReLU3<const D1: u16, const D2: u16, const D3: u16> {
    weight: Tensor1<1>
}

impl <const D1: u16, const D2: u16, const D3: u16>Default for PReLU3<D1, D2, D3> {
    fn default() -> Self {
        Self { weight: Tensor1::rand() }
    }
}

impl <const D1: u16, const D2: u16, const D3: u16>Module for PReLU3<D1, D2, D3> {
    type Input = Tensor3<D1, D2, D3>;
    type Output = Tensor3<D1, D2, D3>;

    fn forward(&self, input: Self::Input) -> Self::Output {
        input.prelu(&self.weight)
    }
}

// #[derive(Default)]
// pub struct PReLU4<const D1: u16, const D2: u16, const D3: u16, const D4: u16> {
//     weight: Tensor1<1>
// }

// impl <const D1: u16, const D2: u16, const D3: u16, const D4: u16>Module for PReLU4<D1, D2, D3, D4> {
//     type Input = Tensor4<D1, D2, D3, D4>;
//     type Output = Tensor4<D1, D2, D3, D4>;

//     fn forward(&self, input: Self::Input) -> Self::Output {
//         input.prelu(&self.weight)
//     }
// }

// #[derive(Default)]
// pub struct PReLU5<const D1: u16, const D2: u16, const D3: u16, const D4: u16, const D5: u16> {
//     weight: Tensor1<1>
// }

// impl <const D1: u16, const D2: u16, const D3: u16, const D4: u16, const D5: u16>Module for PReLU5<D1, D2, D3, D4, D5> {
//     type Input = Tensor5<D1, D2, D3, D4, D5>;
//     type Output = Tensor5<D1, D2, D3, D4, D5>;

//     fn forward(&self, input: Self::Input) -> Self::Output {
//         input.prelu(&self.weight)
//     }
// }