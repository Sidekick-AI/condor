use crate::other_crates::indicatif::{ProgressBar, ProgressStyle};
use num::{Float, Zero};
use tch::nn::VarStore;

/// Creates a training stylized progress bar
pub fn train_progress_bar(steps: u64) -> ProgressBar {
    let bar = ProgressBar::new(steps);
    bar.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green.bright} │{wide_bar:.green.bright/blue}│{pos:>7}/{len:7}({rate} | {eta} | {elapsed_precise})")
        .with_key("eta", |state| {
            let secs = state.eta().as_secs();
            let mut string = format!("{:.1}s", state.eta().as_secs_f64() % 60.);
            if secs / 60 > 0 {string = format!("{}m {}", secs / 60, string);}
            if secs / 3600 > 0 {string = format!("{}h {}", secs / 3600, string);}
            string
        })
        .with_key("rate", |state| {
            if state.per_sec() < 1. {
                format!("{:.1}s/it", state.per_sec())
            } else {
                format!("{:.1}it/s", state.per_sec())
            }
        })
        .progress_chars("█▉▊▋▌▍▎▏  "));
    bar
}

/// Creates a testing stylized progress bar
pub fn test_progress_bar(steps: u64) -> ProgressBar {
    let bar = ProgressBar::new(steps);
    bar.set_style(ProgressStyle::default_bar()
        .template("{spinner:.yellow.bright} │{wide_bar:.yellow.bright/blue}│{pos:>7}/{len:7}({rate} | {eta} | {elapsed_precise})")
        .with_key("eta", |state| {
            let secs = state.eta().as_secs();
            let mut string = format!("{:.1}s", state.eta().as_secs_f64() % 60.);
            if secs / 60 > 0 {string = format!("{}m {}", secs / 60, string);}
            if secs / 3600 > 0 {string = format!("{}h {}", secs / 3600, string);}
            string
        })
        .with_key("rate", |state| {
            if state.per_sec() < 1. {
                format!("{:.1}s/it", state.per_sec())
            } else {
                format!("{:.1}it/s", state.per_sec())
            }
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
        tensor.size().iter().map(|t| {*t as u64}).fold(1u64, |total, val| {total * val})
    }).sum::<u64>()
}

pub struct ExponentialAverage<T: Float> {
    beta: f64,
    moment: f64,
    pub value: T,
    t: i32
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