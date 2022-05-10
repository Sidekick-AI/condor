use super::{Module};

#[derive(Debug)]
pub struct Connector<M1: Module, M2: Module<Input = M1::Output>> {
    module1: M1,
    module2: M2
}

impl <M1: Module, M2: Module<Input = M1::Output>>Connector<M1, M2> {
    pub fn new(module1: M1, module2: M2) -> Self {
        Connector {
            module1,
            module2
        }
    }
}

impl <M1: Module, M2: Module<Input = M1::Output>>Module for Connector<M1, M2> {
    type Input = M1::Input;
    type Output = M2::Output;

    fn train(&mut self) {
        self.module1.train();
        self.module2.train();
    }

    fn eval(&mut self) {
        self.module1.eval();
        self.module2.eval();
    }

    fn forward(&mut self, input: Self::Input) -> Self::Output {
        self.module2.forward(self.module1.forward(input))
    }
}

// A macro for making sequentials
#[macro_use]
mod sequential_macro {
    #[macro_export]
    macro_rules! sequential {
        ($mod1:expr, $( $x:expr ),+ ) => {
            {
                use $crate::modules::Connector;
                let seq = $mod1;
                $(
                    let seq = Connector::new(seq, $x);
                )*
                seq
            }
        };
    }
}
pub use sequential_macro::*;