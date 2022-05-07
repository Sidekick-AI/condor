use crate::*;

pub trait Repeat1<Output>{
    fn repeat_1(self) -> Output;
}

pub trait Repeat2<Output>{
    fn repeat_2(self) -> Output;
}

pub trait Repeat3<Output>{
    fn repeat_3(self) -> Output;
}

// Repeats and unsqueezes for Tensor1
impl <const REPEATS: u16>Repeat1<Tensor1<REPEATS>> for Tensor1<1> {
    fn repeat_1(self) -> Tensor1<REPEATS> {
        Tensor1::from_tch(self.data.repeat(&[REPEATS as i64]))
    }
}

impl <const D1: u16>Tensor1<D1>{
    pub fn unsqueeze_end(self) -> Tensor2<D1, 1> {
        Tensor2::from_tch(self.data)
    }

    pub fn unsqueeze_start(self) -> Tensor2<1, D1> {
        Tensor2::from_tch(self.data)
    }
}

// Repeats and unsqueezes for Tensor2
impl <const REPEATS: u16, const D2: u16>Repeat1<Tensor2<REPEATS, D2>> for Tensor2<1, D2> {
    fn repeat_1(self) -> Tensor2<REPEATS, D2> {
        Tensor2::from_tch(self.data.repeat(&[REPEATS as i64, 1]))
    }
}

impl <const REPEATS: u16, const D1: u16>Repeat2<Tensor2<D1, REPEATS>> for Tensor2<D1, 1> {
    fn repeat_2(self) -> Tensor2<D1, REPEATS> {
        Tensor2::from_tch(self.data.repeat(&[1, REPEATS as i64]))
    }
}

impl <const D1: u16, const D2: u16>Tensor2<D1, D2> {
    pub fn unsqueeze_end(self) -> Tensor3<D1, D2, 1> {
        Tensor3::from_tch(self.data)
    }

    pub fn unsqueeze_start(self) -> Tensor3<1, D1, D2> {
        Tensor3::from_tch(self.data)
    }
}

// Squeezes for Tensor2
impl <const D1: u16>Tensor2<D1, 1> {
    pub fn squeeze_end(self) -> Tensor1<D1> {
        Tensor1::from_tch(self.data)
    }
}

impl <const D1: u16>Tensor2<1, D1> {
    pub fn squeeze_start(self) -> Tensor1<D1> {
        Tensor1::from_tch(self.data)
    }
}

// Repeats for Tensor3
impl <const REPEATS: u16, const D2: u16, const D3: u16>Repeat1<Tensor3<REPEATS, D2, D3>> for Tensor3<1, D2, D3> {
    fn repeat_1(self) -> Tensor3<REPEATS, D2, D3> {
        Tensor3::from_tch(self.data.repeat(&[REPEATS as i64, 1, 1]))
    }
}

impl <const REPEATS: u16, const D1: u16, const D3: u16>Repeat2<Tensor3<D1, REPEATS, D3>> for Tensor3<D1, 1, D3> {
    fn repeat_2(self) -> Tensor3<D1, REPEATS, D3> {
        Tensor3::from_tch(self.data.repeat(&[1, REPEATS as i64, 1]))
    }
}

impl <const REPEATS: u16, const D1: u16, const D2: u16>Repeat3<Tensor3<D1, D2, REPEATS>> for Tensor3<D1, D2, 1> {
    fn repeat_3(self) -> Tensor3<D1, D2, REPEATS> {
        Tensor3::from_tch(self.data.repeat(&[1, 1, REPEATS as i64]))
    }
}

// Squeezes for Tensor3
impl <const D1: u16, const D2: u16>Tensor3<D1, D2, 1> {
    pub fn squeeze_end(self) -> Tensor2<D1, D2> {
        Tensor2::from_tch(self.data)
    }
}

impl <const D1: u16, const D2: u16>Tensor3<1, D1, D2> {
    pub fn squeeze_start(self) -> Tensor2<D1, D2> {
        Tensor2::from_tch(self.data)
    }
}