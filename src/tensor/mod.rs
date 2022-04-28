mod ops;
pub use ops::*;

#[derive(Debug, Default, Clone)]
pub struct Tensor1<const D1: u16> {}

#[derive(Debug, Default, Clone)]
pub struct Tensor2<const D1: u16, const D2: u16> {}

#[derive(Debug, Default, Clone)]
pub struct Tensor3<const D1: u16, const D2: u16, const D3: u16> {}

// #[derive(Debug, Default, Clone)]
// pub struct Tensor4<const D1: u16, const D2: u16, const D3: u16, const D4: u16> {}

// #[derive(Debug, Default, Clone)]
// pub struct Tensor5<const D1: u16, const D2: u16, const D3: u16, const D4: u16, const D5: u16> {}
