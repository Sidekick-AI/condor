#![feature(generic_const_exprs)]
#![feature(adt_const_params)]

/// All NN modules
pub mod modules;

/// Common utilities for machine learning
pub mod utils;

/// Condor tensor
pub mod tensor;
pub use tensor::{Tensor, DimType};

mod other_crates;

#[cfg(test)]
mod tests;