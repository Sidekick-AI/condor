use tch::Tensor;
use super::{NNModule};

/// A sequential layer combining multiple other layers.
#[derive(Debug, Default)]
pub struct Sequential {
    pub layers: Vec<Box<dyn NNModule>>,
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

impl Sequential {
    /// Appends a layer after all the current layers.
    #[allow(clippy::should_implement_trait)]
    pub fn add<M: NNModule + 'static>(&mut self, layer: M) {
        self.layers.push(Box::new(layer));
    }

    /// Appends a closure after all the current layers.
    pub fn add_fn<F: 'static + Fn(&Tensor, bool) -> Tensor + Send>(&mut self, f: F) {
        self.add(super::func(f))
    }

    /// Applies the forward pass and returns the output for each layer.
    pub fn forward_all(&mut self, xs: &Tensor, n: Option<usize>) -> Vec<Tensor> {
        if self.layers.is_empty() {
            vec![xs.shallow_clone()]
        } else {
            let n = n.unwrap_or_else(|| self.layers.len());
            let xs = self.layers[0].forward(xs);
            let mut vec = vec![];
            let out = self.layers.iter_mut().take(n).skip(1).fold(xs, |xs, layer| {
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

    fn forward(&mut self, x: &tch::Tensor) -> tch::Tensor {
        if self.layers.is_empty() {
            x.shallow_clone()
        } else {
            let x = self.layers[0].forward(x);
            self.layers
                .iter_mut()
                .skip(1)
                .fold(x, |x, layer| layer.forward(&x))
        }
    }
}

// impl ModuleCopy for Sequential {
//     fn copy(&mut self, module: &Self) {
//         for i in 0..self.layers.len() {
//             self.layers[i].copy(module.layers[i])
//         }
//     }
// }


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