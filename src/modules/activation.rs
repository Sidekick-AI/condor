use crate::{Tensor1, Tensor2, Tensor3, Module};

pub trait Activation {
    type Tensor;

    fn apply(&self, input: Self::Tensor) -> Self::Tensor;
}

/// Blanket impl module for all activations
impl <T>Module for T where T: Activation {
    type Input = T::Tensor;
    type Output = T::Tensor;

    fn forward(&self, input: Self::Input) -> Self::Output {
        self.apply(input)
    }
}

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
#[derive(Default, Clone, Copy)]
pub struct ReLU1<const D1: u16> {}

impl <const D1: u16>Activation for ReLU1<D1> {
    type Tensor = Tensor1<D1>;

    fn apply(&self, input: Self::Tensor) -> Self::Tensor {
        input.relu()
    }
}

#[derive(Default, Clone, Copy)]
pub struct ReLU2<const D1: u16, const D2: u16> {}

impl <const D1: u16, const D2: u16>Activation for ReLU2<D1, D2> {
    type Tensor = Tensor2<D1, D2>;

    fn apply(&self, input: Self::Tensor) -> Self::Tensor {
        input.relu()
    }
}

#[derive(Default, Clone, Copy)]
pub struct ReLU3<const D1: u16, const D2: u16, const D3: u16> {}

impl <const D1: u16, const D2: u16, const D3: u16>Activation for ReLU3<D1, D2, D3> {
    type Tensor = Tensor3<D1, D2, D3>;

    fn apply(&self, input: Self::Tensor) -> Self::Tensor {
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

#[derive(Default, Clone, Copy)]
pub struct GeLU1<const D1: u16> {}

impl <const D1: u16>Activation for GeLU1<D1> {
    type Tensor = Tensor1<D1>;

    fn apply(&self, input: Self::Tensor) -> Self::Tensor {
        input.gelu()
    }
}

#[derive(Default, Clone, Copy)]
pub struct GeLU2<const D1: u16, const D2: u16> {}

impl <const D1: u16, const D2: u16>Activation for GeLU2<D1, D2> {
    type Tensor = Tensor2<D1, D2>;

    fn apply(&self, input: Self::Tensor) -> Self::Tensor {
        input.gelu()
    }
}

#[derive(Default, Clone, Copy)]
pub struct GeLU3<const D1: u16, const D2: u16, const D3: u16> {}

impl <const D1: u16, const D2: u16, const D3: u16>Activation for GeLU3<D1, D2, D3> {
    type Tensor = Tensor3<D1, D2, D3>;

    fn apply(&self, input: Self::Tensor) -> Self::Tensor {
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

#[derive(Default, Clone, Copy)]
pub struct Sigmoid1<const D1: u16> {}

impl <const D1: u16>Activation for Sigmoid1<D1> {
    type Tensor = Tensor1<D1>;

    fn apply(&self, input: Self::Tensor) -> Self::Tensor {
        input.sigmoid()
    }
}

#[derive(Default, Clone, Copy)]
pub struct Sigmoid2<const D1: u16, const D2: u16> {}

impl <const D1: u16, const D2: u16>Activation for Sigmoid2<D1, D2> {
    type Tensor = Tensor2<D1, D2>;

    fn apply(&self, input: Self::Tensor) -> Self::Tensor {
        input.sigmoid()
    }
}

#[derive(Default, Clone, Copy)]
pub struct Sigmoid3<const D1: u16, const D2: u16, const D3: u16> {}

impl <const D1: u16, const D2: u16, const D3: u16>Activation for Sigmoid3<D1, D2, D3> {
    type Tensor = Tensor3<D1, D2, D3>;

    fn apply(&self, input: Self::Tensor) -> Self::Tensor {
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

#[derive(Clone)]
pub struct PReLU1<const D1: u16> {
    weight: Tensor1<1>
}

impl <const D1: u16>Default for PReLU1<D1> {
    fn default() -> Self {
        Self { weight: Tensor1::rand() }
    }
}

impl <const D1: u16>Activation for PReLU1<D1> {
    type Tensor = Tensor1<D1>;

    fn apply(&self, input: Self::Tensor) -> Self::Tensor {
        input.prelu(&self.weight)
    }
}

#[derive(Clone)]
pub struct PReLU2<const D1: u16, const D2: u16> {
    weight: Tensor1<1>
}

impl <const D1: u16, const D2: u16>Default for PReLU2<D1, D2> {
    fn default() -> Self {
        Self { weight: Tensor1::rand() }
    }
}

impl <const D1: u16, const D2: u16>Activation for PReLU2<D1, D2> {
    type Tensor = Tensor2<D1, D2>;

    fn apply(&self, input: Self::Tensor) -> Self::Tensor {
        input.prelu(&self.weight)
    }
}

#[derive(Clone)]
pub struct PReLU3<const D1: u16, const D2: u16, const D3: u16> {
    weight: Tensor1<1>
}

impl <const D1: u16, const D2: u16, const D3: u16>Default for PReLU3<D1, D2, D3> {
    fn default() -> Self {
        Self { weight: Tensor1::rand() }
    }
}

impl <const D1: u16, const D2: u16, const D3: u16>Activation for PReLU3<D1, D2, D3> {
    type Tensor = Tensor3<D1, D2, D3>;

    fn apply(&self, input: Self::Tensor) -> Self::Tensor {
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