use tensorboard_rs::summary_writer::SummaryWriter;

pub struct Tensorboard {
    writer: SummaryWriter,
}

impl Tensorboard {
    pub fn new(run_name: &str) -> Self {
        Self {
            writer: SummaryWriter::new(&format!("./logdir/{}", run_name))
        }
    }

    pub fn with_path(path: &str) -> Self {
        Self {
            writer: SummaryWriter::new(path)
        }
    }

    /// Log scalar
    pub fn log<T: Cast<f32>>(&mut self, name: &str, value: T, step: usize) {
        self.writer.add_scalar(&format!("data/default/{name}"), value.cast(), step);
    }

    /// Log scalar in graph
    pub fn log_in_graph<T: Cast<f32>>(&mut self, graph_name: &str, scalar_name: &str, value: T, step: usize) {
        self.writer.add_scalar(&format!("data/{graph_name}/{scalar_name}"), value.cast(), step);
    }

    /// Log scalar in group and graph
    pub fn log_in_group<T: Cast<f32>>(&mut self, group_name: &str, graph_name: &str, scalar_name: &str, value: T, step: usize) {
        self.writer.add_scalar(&format!("{group_name}/{graph_name}/{scalar_name}"), value.cast(), step);
    }
}

/// Trait for specifying availiable casts
pub trait Cast<A> {
    fn cast(self) -> A;
}

impl Cast<f32> for f32 {
    fn cast(self) -> f32 {
        self as f32
    }
}

impl Cast<f32> for f64 {
    fn cast(self) -> f32 {
        self as f32
    }
}

impl Cast<f32> for u8 {
    fn cast(self) -> f32 {
        self as f32
    }
}

impl Cast<f32> for u16 {
    fn cast(self) -> f32 {
        self as f32
    }
}

impl Cast<f32> for u32 {
    fn cast(self) -> f32 {
        self as f32
    }
}

impl Cast<f32> for u64 {
    fn cast(self) -> f32 {
        self as f32
    }
}

impl Cast<f32> for u128 {
    fn cast(self) -> f32 {
        self as f32
    }
}

impl Cast<f32> for i8 {
    fn cast(self) -> f32 {
        self as f32
    }
}

impl Cast<f32> for i16 {
    fn cast(self) -> f32 {
        self as f32
    }
}

impl Cast<f32> for i32 {
    fn cast(self) -> f32 {
        self as f32
    }
}

impl Cast<f32> for i64 {
    fn cast(self) -> f32 {
        self as f32
    }
}

impl Cast<f32> for i128 {
    fn cast(self) -> f32 {
        self as f32
    }
}

impl Cast<f32> for usize {
    fn cast(self) -> f32 {
        self as f32
    }
}