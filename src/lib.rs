#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(adt_const_params)]
#![feature(structural_match)]
#![feature(generic_associated_types)]

mod tensor;
mod modules;

pub use tensor::*;
pub use modules::*;

#[cfg(test)]
mod tests;
