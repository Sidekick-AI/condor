use super::NNModule;
use tch::nn::{Embedding, LayerNorm, Module};
use tch::{nn, Device, IndexOp, Kind, Tensor};

#[macro_export]
macro_rules! sequential {
    ( $( $x:expr ),* ) => {
        {
            let mut seq = nn:seq();
            $(
                seq.add($x);
            )*
            seq
        }
    };
}