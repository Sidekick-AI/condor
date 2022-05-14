use std::ops::Div;

use crate::other_crates::indicatif::{ProgressBar, ProgressStyle};
use num::{Float, Zero};
use rand::{Rng, thread_rng};
use tch::{Kind, Tensor, nn::{Optimizer, VarStore}};


pub struct DecayingOptimizer {
    optimizer: Optimizer,
    lr: f64,
    decay: f64
}

impl DecayingOptimizer {
    pub fn new(optimizer: Optimizer, initial_lr: f64, decay: f64) -> Self {
        DecayingOptimizer {
            optimizer,
            lr: initial_lr,
            decay
        }
    }

    /// Decay the learning rate
    pub fn step_lr(&mut self) {
        self.lr *= self.decay;
        self.optimizer.set_lr(self.lr);
    }

    /// Get the current learning rate
    pub fn get_lr(&self) -> f64 {
        self.lr
    }
}

impl core::ops::Deref for DecayingOptimizer {
    type Target = Optimizer;

    fn deref(&self) -> &Self::Target {
        &self.optimizer
    }
}

impl core::ops::DerefMut for DecayingOptimizer {
    fn deref_mut(&mut self) -> &mut Optimizer {
        &mut self.optimizer
    }
}

/// Sample from distribution TODO: make more generalized sampling
pub fn sample_1d(logits: Tensor, temperature: f64) -> usize {
    assert_eq!(logits.size().len(), 1);
    let softmaxxed = Vec::<f64>::from(logits.div(temperature).softmax(0, Kind::Float));
    let num = thread_rng().gen_range((0.)..1.);
    let mut total = 0.;
    for (i, val) in softmaxxed.iter().enumerate() {
        total += val;
        if total > num {return i;}
    }
    0
}

/// Creates a training stylized progress bar
pub fn train_progress_bar(steps: u64) -> ProgressBar {
    let bar = ProgressBar::new(steps);
    bar.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green.bright} {percent}│{wide_bar:.green.bright/blue}│{pos:>7}/{len:7}({msg} | {rate} | {eta} | {elapsed_precise})")
        .with_key("eta", |state| {
            let secs = state.eta().as_secs();
            let mut string = format!("{:.1}s", state.eta().as_secs_f64() % 60.);
            if (secs / 60) % 60 > 0 {string = format!("{}m {}", (secs / 60) % 60, string);}
            if (secs / 3600) % 24 > 0 {string = format!("{}h {}", (secs / 3600) % 24, string);}
            if secs / 86400 > 0 {string = format!("{}d {}", secs / 86400, string);}
            string
        })
        .with_key("rate", |state| {
            if state.per_sec() < 1. {
                format!("{:.1}s/it", 1. / state.per_sec())
            } else {
                format!("{:.1}it/s", state.per_sec())
            }
        })
        .with_key("percent", |state| {
            format!("{:.1}%", (state.pos as f32 / state.len as f32) * 100.)
        })
        .progress_chars("█▉▊▋▌▍▎▏  "));
    bar
}

/// Creates a testing stylized progress bar
pub fn test_progress_bar(steps: u64) -> ProgressBar {
    let bar = ProgressBar::new(steps);
    bar.set_style(ProgressStyle::default_bar()
        .template("{spinner:.yellow.bright} {percent}│{wide_bar:.yellow.bright/blue}│{pos:>7}/{len:7}({rate} | {eta} | {elapsed_precise})")
        .with_key("eta", |state| {
            let secs = state.eta().as_secs();
            let mut string = format!("{:.1}s", state.eta().as_secs_f64() % 60.);
            if (secs / 60) % 60 > 0 {string = format!("{}m {}", (secs / 60) % 60, string);}
            if (secs / 3600) % 24 > 0 {string = format!("{}h {}", (secs / 3600) % 24, string);}
            if secs / 86400 > 0 {string = format!("{}d {}", secs / 86400, string);}
            string
        })
        .with_key("rate", |state| {
            if state.per_sec() < 1. {
                format!("{:.1}s/it", 1. / state.per_sec())
            } else {
                format!("{:.1}it/s", state.per_sec())
            }
        })
        .with_key("percent", |state| {
            format!("{:.1}%", (state.pos as f32 / state.len as f32) * 100.)
        })
        .progress_chars("█▉▊▋▌▍▎▏  "));
    bar
}

pub fn readable_number(num: i64) -> String {
    let abbreviations = vec!["K", "M", "B", "T", "Q"];
    for i in (1..5).rev() {
        if num - i64::pow(10, i * 3) > 0 {
            return format!("{:.1}{}", (num as f64) / f64::powi(10., i as i32 * 3), abbreviations[(i - 1) as usize]);
        }
    }
    num.to_string()
}

pub fn count_parameters(vs: &VarStore) -> u64 {
    vs.trainable_variables().iter().map(|tensor| {
        tensor.size().iter().map(|t| {*t as u64}).product::<u64>()
    }).sum::<u64>()
}

pub struct ExponentialAverage<T: Float> {
    beta: f64,
    moment: f64,
    pub value: T,
    t: i32
}

impl <T: Float> Default for ExponentialAverage<T> {
    fn default() -> Self {
        ExponentialAverage {
            beta: 0.99,
            moment: 0.,
            value: Zero::zero(),
            t: 0
        }
    }
}

impl <T: Float> ExponentialAverage<T> {
    pub fn new() -> Self {
        ExponentialAverage {
            beta: 0.99,
            moment: 0.,
            value: Zero::zero(),
            t: 0
        }
    }

    pub fn with_beta(beta: f64) -> Self {
        assert!((0. ..=1.).contains(&beta));
        ExponentialAverage {
            beta,
            moment: 0.,
            value: Zero::zero(),
            t: 0
        }
    }

    pub fn update(&mut self, value: T) {
        self.t += 1;
        self.moment = self.beta * self.moment + (1. - self.beta) * value.to_f64().unwrap();
        // bias correction
        self.value = T::from(self.moment / (1. - f64::powi(self.beta, self.t))).unwrap();
    }

    pub fn reset(&mut self) {
        self.moment = 0.;
        self.value = Zero::zero();
        self.t = 0;
    }
}

/// Constant decay a value by a factor
pub struct Decay {
    pub factor: f64,
    pub value: f64,
}

impl Decay {
    pub fn new(initial: f64, factor: f64) -> Self {
        Self { factor, value: initial }
    }

    pub fn step(&mut self) {
        self.value *= self.factor;
    }
}

/// Step decay a value at multiple steps
#[derive(Debug, Clone)]
pub struct StepDecay {
    pub value: f64,
    current_step: usize,
    steps: Vec<StepType>,
    initial_steps: Vec<StepType>,
}

impl StepDecay {
    pub fn new(initial: f64, steps: &[StepType]) -> Self {
        Self {
            value: initial,
            current_step: 0,
            steps: steps.to_vec(),
            initial_steps: steps.to_vec(),
        }
    }

    pub fn step(&mut self) {
        if self.current_step >= self.steps.len() {return;}
        match &mut self.steps[self.current_step] {
            StepType::Constant { value, steps } => {
                self.value = *value;
                *steps -= 1;
                if *steps == 0 {
                    self.current_step += 1;
                }
            },
            StepType::Until { factor, target } => {
                let initial_value = self.value;
                match factor {
                    DeltaFactor::Additive(i) => self.value += *i,
                    DeltaFactor::Multiplicitive(i) => self.value *= *i,
                }
                if (self.value > *target && initial_value < *target) || (self.value < *target && initial_value > *target) {
                    self.current_step += 1;
                }
            },
            StepType::NSteps { factor, steps } => {
                match factor {
                    DeltaFactor::Additive(i) => self.value += *i,
                    DeltaFactor::Multiplicitive(i) => self.value *= *i,
                }
                *steps -= 1;
                if *steps == 0 {
                    self.current_step += 1;
                }
            },
            StepType::Lerp {end, steps} => {
                let factor = (*end - self.value) / (*steps as f64);
                self.value += factor;
                *steps -= 1;
                if *steps == 0 {
                    self.current_step += 1;
                }
            }
        }
    }

    pub fn n_steps(&mut self, mut n_steps: usize) {

        while n_steps > 0 {
            if self.current_step >= self.steps.len() {return;}
            match &mut self.steps[self.current_step] {
                StepType::Constant { value, steps } => {
                    self.value = *value;
                    let sub_factor = n_steps.min(*steps);
                    *steps -= sub_factor;
                    n_steps -= sub_factor;
                    if *steps == 0 {
                        self.current_step += 1;
                    }
                },
                StepType::Until { factor, target } => {
                    let initial_value = self.value;
                    match factor {
                        DeltaFactor::Additive(i) => {
                            let steps_required = ((*target - self.value) / *i).ceil() as usize;
                            let steps_taken = steps_required.min(n_steps);
                            self.value += *i * steps_taken as f64;
                            n_steps -= steps_taken;
                        },
                        DeltaFactor::Multiplicitive(i) => {
                            let steps_required = (*target / self.value).log(*i).ceil() as usize;
                            let steps_taken = steps_required.min(n_steps);
                            self.value *= i.powf(steps_taken as f64);
                            n_steps -= steps_taken;
                        },
                    }
                    if (self.value > *target && initial_value < *target) || (self.value < *target && initial_value > *target) {
                        self.current_step += 1;
                    }
                },
                StepType::NSteps { factor, steps } => {
                    let n_factor = n_steps.min(*steps);
                    match factor {
                        DeltaFactor::Additive(i) => self.value += *i * n_factor as f64,
                        DeltaFactor::Multiplicitive(i) => self.value *= i.powf(n_factor as f64),
                    }
                    *steps -= n_factor;
                    n_steps -= n_factor;
                    if *steps == 0 {
                        self.current_step += 1;
                    }
                },
                StepType::Lerp {end, steps} => {
                    let factor = (*end - self.value) / (*steps as f64);
                    let steps_required = ((*end - self.value) / factor) as usize;
                    let steps_taken = steps_required.min(n_steps);
                    if steps_taken == 0 {return;}

                    self.value += factor * steps_taken as f64;
                    *steps -= steps_taken;
                    n_steps -= steps_taken;
                    
                    if *steps == 0 {
                        self.current_step += 1;
                    }
                },
            }
        }
    }

    pub fn reset(&mut self) {
        self.steps = self.initial_steps.clone();
    }
}

#[derive(Debug, Clone)]
pub enum StepType {
    Constant{value: f64, steps: usize},
    Until{factor: DeltaFactor, target: f64},
    NSteps{factor: DeltaFactor, steps: usize},
    Lerp{end: f64, steps: usize},
}

#[derive(Debug, Clone)]
pub enum DeltaFactor {
    Additive(f64),
    Multiplicitive(f64),
}