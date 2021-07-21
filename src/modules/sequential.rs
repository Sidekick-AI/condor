use tch::{Tensor, nn::{self, Func, Module}};
use super::NNModule;

pub trait NeuralModule: Module + NNModule {}
impl NeuralModule for Func<'_> {}

/// A sequential layer combining multiple other layers.
#[derive(Debug)]
pub struct Sequential {
    pub layers: Vec<Box<dyn NNModule>>,
}

impl Default for Sequential {
    fn default() -> Self {
        Sequential {
            layers: vec![]
        }
    }
}

impl Sequential {
    pub fn new() -> Self {
        Sequential {
            layers: vec![]
        }
    }

    pub fn from_layers(layers: Vec<Box<dyn NNModule>>) -> Self {
        Sequential {
            layers
        }
    }

    /// The number of sub-layers embedded in this layer.
    pub fn len(&self) -> i64 {
        self.layers.len() as i64
    }

    /// Returns true if this layer does not have any sub-layer.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

impl Module for Sequential {
    fn forward(&self, xs: &Tensor) -> Tensor {
        if self.layers.is_empty() {
            xs.shallow_clone()
        } else {
            let xs = self.layers[0].forward(xs);
            self.layers
                .iter()
                .skip(1)
                .fold(xs, |xs, layer| layer.forward(&xs))
        }
    }
}

impl Sequential {
    /// Appends a layer after all the current layers.
    #[allow(clippy::should_implement_trait)]
    pub fn add<M: NNModule + 'static>(&mut self, layer: M) {
        self.layers.push(Box::new(layer));
    }

    /// Appends a closure after all the current layers.
    pub fn add_fn<F: 'static + Fn(&Tensor) -> Tensor + Send>(&mut self, f: F) {
        self.add(nn::func(f))
    }

    /// Applies the forward pass and returns the output for each layer.
    pub fn forward_all(&self, xs: &Tensor, n: Option<usize>) -> Vec<Tensor> {
        if self.layers.is_empty() {
            vec![xs.shallow_clone()]
        } else {
            let n = n.unwrap_or_else(|| self.layers.len());
            let xs = self.layers[0].forward(xs);
            let mut vec = vec![];
            let out = self.layers.iter().take(n).skip(1).fold(xs, |xs, layer| {
                let out = layer.forward(&xs);
                vec.push(xs);
                out
            });
            vec.push(out);
            vec
        }
    }
}

impl NNModule for Sequential {
    fn train(&mut self) {
        for layer in &mut self.layers {
            layer.train();
        }
    }

    fn eval(&mut self) {
        for layer in &mut self.layers {
            layer.eval();
        }
    }

    fn count_parameters(&self) -> u64 {
        self.layers.iter().map(|l| {l.count_parameters()}).sum()
    }
}

impl NNModule for Func<'_> {
    fn train(&mut self) {}

    fn eval(&mut self) {}

    fn count_parameters(&self) -> u64 {0}
}

// A macro for making sequentials
#[macro_use]
mod sequential_macro {
    #[macro_export]
    macro_rules! sequential {
        ( $( $x:expr ),* ) => {
            {
                let mut seq = Sequential::new();
                $(
                    seq.add($x);
                )*
                seq
            }
        };
    }
}
pub use sequential_macro::*;