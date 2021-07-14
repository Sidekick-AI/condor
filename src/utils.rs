pub struct ExponentialAverage {
    beta: f32,
    moment: f32,
    pub value: f32,
    t: i32
}

impl ExponentialAverage {
    pub fn new() -> Self {
        ExponentialAverage {
            beta: 0.99,
            moment: 0.,
            value: 0.,
            t: 0
        }
    }

    pub fn update(&mut self, value: f32) {
        self.t += 1;
        self.moment = self.beta * self.moment + (1. - self.beta) * value;
        // bias correction
        self.value = self.moment / (f32::powi(1. - self.beta, self.t));
    }

    pub fn reset(&mut self) {
        self.moment = 0.;
        self.value = 0.;
        self.t = 0;
    }
}