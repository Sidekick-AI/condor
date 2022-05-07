use crate::*;

pub struct RNNCell<const BATCH: u16, const IN: u16, const OUT: u16, const HIDDEN: u16, A, B>
where A: Activation<Tensor = Tensor2<BATCH, HIDDEN>>, B: Activation<Tensor = Tensor2<BATCH, OUT>> {
    wx: Linear<BATCH, IN, HIDDEN>,
    wh: Linear<BATCH, HIDDEN, HIDDEN>,
    wy: Linear<BATCH, HIDDEN, OUT>,
    activation_h: A,
    activation_o: B,
}

impl <const BATCH: u16, const IN: u16, const OUT: u16, const HIDDEN: u16, A, B>RNNCell<BATCH, IN, OUT, HIDDEN, A, B>
where A: Activation<Tensor = Tensor2<BATCH, HIDDEN>>, B: Activation<Tensor = Tensor2<BATCH, OUT>> {
    pub fn new(activation_h: A, activation_o: B) -> Self {
        Self {
            activation_h,
            activation_o,
            wx: Linear::default(),
            wy: Linear::default(),
            wh: Linear::default()
        }
    }
}

impl <const BATCH: u16, const IN: u16, const OUT: u16, const HIDDEN: u16, A, B>Module for RNNCell<BATCH, IN, OUT, HIDDEN, A, B>
where A: Activation<Tensor = Tensor2<BATCH, HIDDEN>>,
B: Activation<Tensor = Tensor2<BATCH, OUT>> {
    type Input = (Tensor2<BATCH, IN>, Tensor2<BATCH, HIDDEN>);
    type Output = (Tensor2<BATCH, OUT>, Tensor2<BATCH, HIDDEN>);

    fn forward(&self, (input, hidden): Self::Input) -> Self::Output {
        let input = self.activation_h.apply(self.wx.forward(input));
        let hidden = self.activation_h.apply(self.wh.forward(hidden));

        let new_hidden = input + hidden;
        let output = self.activation_o.apply(self.wy.forward(new_hidden.clone()));
        (output, new_hidden)
    }
}

pub struct RNN<const BATCH: u16, const SEQ: u16, const HIDDEN: u16, const IN: u16, const OUT: u16, const BI: bool, A, B> 
where A: Activation<Tensor = Tensor2<BATCH, HIDDEN>>, B: Activation<Tensor = Tensor2<BATCH, OUT>> {
    cell: RNNCell<BATCH, IN, OUT, HIDDEN, A, B>,
    initial_hidden: Tensor1<HIDDEN>,
    reverse_cell: Option<RNNCell<BATCH, IN, OUT, HIDDEN, A, B>>,
    initial_reverse_hidden: Option<Tensor1<HIDDEN>>
}

impl <const BATCH: u16, const SEQ: u16, const HIDDEN: u16, const IN: u16, const OUT: u16, const BI: bool, A, B>RNN<BATCH, SEQ, HIDDEN, IN, OUT, BI, A, B>
where A: Activation<Tensor = Tensor2<BATCH, HIDDEN>> + Clone, B: Activation<Tensor = Tensor2<BATCH, OUT>> + Clone {
    pub fn new<const BIDIRECTIONAL: bool>(activation_h: A, activation_o: B) -> RNN<BATCH, SEQ, HIDDEN, IN, OUT, BIDIRECTIONAL, A, B> {
        if BIDIRECTIONAL {
            RNN {
                cell: RNNCell::new(activation_h.clone(), activation_o.clone()),
                initial_hidden: Tensor1::rand(),
                reverse_cell: Some(RNNCell::new(activation_h, activation_o)),
                initial_reverse_hidden: Some(Tensor1::rand()),
            }
        } else {
            RNN {
                cell: RNNCell::new(activation_h, activation_o),
                initial_hidden: Tensor1::rand(),
                reverse_cell: None,
                initial_reverse_hidden: None,
            }
        }
    }
}

// Single directional RNN
impl <const BATCH: u16, const SEQ: u16, const HIDDEN: u16, const IN: u16, const OUT: u16, A, B>Module for RNN<BATCH, SEQ, HIDDEN, IN, OUT, false, A, B>
where A: Activation<Tensor = Tensor2<BATCH, HIDDEN>>, B: Activation<Tensor = Tensor2<BATCH, OUT>> {
    type Input = Tensor3<SEQ, BATCH, IN>;
    type Output = Tensor3<SEQ, BATCH, OUT>;

    /// Standard encoding (input sequence, output sequence)
    fn forward(&self, input: Self::Input) -> Self::Output {
        // Feed through forward
        let initial_hidden: Tensor2<1, HIDDEN> = self.initial_hidden.clone().unsqueeze_start();
        let initial_hidden: Tensor2<BATCH, HIDDEN> = initial_hidden.repeat_1();
        let (mut curr_output, mut hidden) = self.cell.forward((
            input.get(0), 
            initial_hidden
        ));

        let mut output: Tensor3<SEQ, BATCH, OUT> = Tensor3::zeros();
        output.set(0, &curr_output);

        // Loop through each remaining input
        for i in 1..SEQ {
            (curr_output, hidden) = self.cell.forward((
                input.get(i), 
                hidden
            ));
            output.set(i, &curr_output);
        }
        
        output
    }
}

// Bi directional RNN
impl <const BATCH: u16, const SEQ: u16, const HIDDEN: u16, const IN: u16, const OUT: u16, A, B>Module for RNN<BATCH, SEQ, HIDDEN, IN, OUT, true, A, B>
where A: Activation<Tensor = Tensor2<BATCH, HIDDEN>>, B: Activation<Tensor = Tensor2<BATCH, OUT>>, Tensor3<SEQ, BATCH, {OUT * 2}>: Sized {
    type Input = Tensor3<SEQ, BATCH, IN>;
    type Output = Tensor3<SEQ, BATCH, {OUT * 2}>;

    /// Standard encoding (input sequence, output sequence)
    fn forward(&self, input: Self::Input) -> Self::Output {
        // Feed through forward
        let initial_hidden: Tensor2<1, HIDDEN> = self.initial_hidden.clone().unsqueeze_start();
        let initial_hidden: Tensor2<BATCH, HIDDEN> = initial_hidden.repeat_1();
        let (mut curr_output, mut hidden) = self.cell.forward((
            input.get(0), 
            initial_hidden
        ));

        let mut output: Tensor3<SEQ, BATCH, OUT> = Tensor3::zeros();
        output.set(0, &curr_output);

        // Loop through each remaining input
        for i in 1..SEQ {
            (curr_output, hidden) = self.cell.forward((
                input.get(i), 
                hidden
            ));
            output.set(i, &curr_output);
        }
        
        // Feed through backward
        let reverse_initial_hidden: Tensor2<1, HIDDEN> = self.initial_reverse_hidden.clone().unwrap().unsqueeze_start();
        let reverse_initial_hidden: Tensor2<BATCH, HIDDEN> = reverse_initial_hidden.repeat_1();
        let (mut curr_output, mut hidden) = self.reverse_cell.as_ref().unwrap().forward((
            input.get(SEQ - 1), 
            reverse_initial_hidden
        ));

        let mut reverse_output: Tensor3<SEQ, BATCH, OUT> = Tensor3::zeros();
        reverse_output.set(0, &curr_output);

        // Loop through each remaining input
        for i in (1..SEQ).rev() {
            (curr_output, hidden) = self.reverse_cell.as_ref().unwrap().forward((
                input.get(i), 
                hidden
            ));
            reverse_output.set(i, &curr_output);
        }
        
        // Concat forward and backward outputs on last dimension
        output.cat_3(&reverse_output)
    }
}