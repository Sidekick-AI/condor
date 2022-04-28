use crate::{Module};

pub struct Connector<M1: Module, M2: Module<Input = M1::Output>> {
    module_1: M1,
    module_2: M2
}

impl <M1: Module, M2: Module<Input = M1::Output>>Connector<M1, M2> {
    pub fn new(module_1: M1, module_2: M2) -> Self {
        Self {
            module_1,
            module_2
        }
    }
}

impl <M1: Module, M2: Module<Input = M1::Output>>Module for Connector<M1, M2> {
    type Input = M1::Input;
    type Output = M2::Output;

    fn train(&mut self) {
        self.module_1.train();
        self.module_2.train();
    }

    fn eval(&mut self) {
        self.module_1.eval();
        self.module_2.eval();
    }

    fn forward(&self, input: Self::Input) -> Self::Output {
        self.module_2.forward(self.module_1.forward(input))
    }
}

#[macro_use]
mod sequential_macro {
    #[macro_export]
    macro_rules! sequential {
        ($mod1:expr, $( $x:expr ),+ ) => {{
            use $crate::modules::Connector;
            let seq = $mod1;
            $(
                let seq = Connector::new(seq, $x);
            )*
            seq
        }};
    }
}
pub use sequential_macro::*;