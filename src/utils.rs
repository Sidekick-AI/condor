use std::ops::Div;

use indicatif::{ProgressBar, ProgressStyle};
use num::{Float, Zero};
use rand::{Rng, thread_rng};
use tch::{Kind, Tensor, nn::{Optimizer, VarStore}};


pub struct DecayingOptimizer<T> {
    optimizer: Optimizer<T>,
    lr: f64,
    decay: f64
}

impl<T> DecayingOptimizer<T> {
    pub fn new(optimizer: Optimizer<T>, initial_lr: f64, decay: f64) -> Self {
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

impl<T> core::ops::Deref for DecayingOptimizer<T> {
    type Target = Optimizer<T>;

    fn deref(&self) -> &Self::Target {
        &self.optimizer
    }
}

impl<T> core::ops::DerefMut for DecayingOptimizer<T> {
    fn deref_mut(&mut self) -> &mut Optimizer<T> {
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
            if secs / 3600 > 0 {string = format!("{}h {}", secs / 3600, string);}
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
            if secs / 3600 > 0 {string = format!("{}h {}", secs / 3600, string);}
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