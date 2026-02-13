use pyo3::prelude::*;
use std::collections::VecDeque;
use super::trend::EMAStreaming;

// ============================================================================
// ATR (Average True Range)
// ============================================================================
#[pyclass]
pub struct ATRStreaming {
    window: usize,
    alpha: f64,
    prev_close: f64,
    current_value: f64,
    update_count: usize,
}

#[pymethods]
impl ATRStreaming {
    #[new]
    pub fn new(window: usize) -> Self {
        Self {
            window,
            alpha: 1.0 / window as f64,
            prev_close: f64::NAN,
            current_value: f64::NAN,
            update_count: 0,
        }
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> f64 {
        self.update_count += 1;

        let tr = if self.prev_close.is_nan() {
            high - low
        } else {
            let tr1 = high - low;
            let tr2 = (high - self.prev_close).abs();
            let tr3 = (low - self.prev_close).abs();
            tr1.max(tr2).max(tr3)
        };

        if self.current_value.is_nan() {
            self.current_value = tr;
        } else {
            self.current_value = (1.0 - self.alpha) * self.current_value + self.alpha * tr;
        }

        self.prev_close = close;

        if self.update_count >= self.window {
            self.current_value
        } else {
            f64::NAN
        }
    }

    pub fn reset(&mut self) {
        self.prev_close = f64::NAN;
        self.current_value = f64::NAN;
        self.update_count = 0;
    }
}

// ============================================================================
// Bollinger Bands
// ============================================================================
#[pyclass]
#[pyo3(name = "BBandsStreaming")]
pub struct BollingerBandsStreaming {
    window: usize,
    std_dev: f64,
    buffer: VecDeque<f64>,
}

#[pymethods]
impl BollingerBandsStreaming {
    #[new]
    pub fn new(window: usize, std_dev: f64) -> Self {
        Self {
            window,
            std_dev,
            buffer: VecDeque::with_capacity(window),
        }
    }

    /// Returns (upper, middle, lower)
    pub fn update(&mut self, value: f64) -> (f64, f64, f64) {
        self.buffer.push_back(value);

        if self.buffer.len() > self.window {
            self.buffer.pop_front();
        }

        if self.buffer.len() < self.window {
            (f64::NAN, f64::NAN, f64::NAN)
        } else {
            let mean: f64 = self.buffer.iter().sum::<f64>() / self.window as f64;
            let variance: f64 = self.buffer.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / self.window as f64;
            let std = variance.sqrt();

            let upper = mean + self.std_dev * std;
            let lower = mean - self.std_dev * std;

            (upper, mean, lower)
        }
    }

    pub fn reset(&mut self) {
        self.buffer.clear();
    }
}

// ============================================================================
// Keltner Channel
// ============================================================================
#[pyclass]
pub struct KeltnerChannelStreaming {
    multiplier: f64,
    ema: EMAStreaming,
    atr: ATRStreaming,
}

#[pymethods]
impl KeltnerChannelStreaming {
    #[new]
    pub fn new(window: usize, atr_period: usize, multiplier: f64) -> Self {
        Self {
            multiplier,
            ema: EMAStreaming::new(window),
            atr: ATRStreaming::new(atr_period),
        }
    }

    /// Returns (upper, middle, lower)
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> (f64, f64, f64) {
        let ema_value = self.ema.update(close);
        let atr_value = self.atr.update(high, low, close);

        if ema_value.is_nan() || atr_value.is_nan() {
            (f64::NAN, f64::NAN, f64::NAN)
        } else {
            let upper = ema_value + self.multiplier * atr_value;
            let lower = ema_value - self.multiplier * atr_value;
            (upper, ema_value, lower)
        }
    }

    pub fn reset(&mut self) {
        self.ema.reset();
        self.atr.reset();
    }
}

// ============================================================================
// Donchian Channel
// ============================================================================
#[pyclass]
pub struct DonchianChannelStreaming {
    window: usize,
    high_buffer: VecDeque<f64>,
    low_buffer: VecDeque<f64>,
}

#[pymethods]
impl DonchianChannelStreaming {
    #[new]
    pub fn new(window: usize) -> Self {
        Self {
            window,
            high_buffer: VecDeque::with_capacity(window),
            low_buffer: VecDeque::with_capacity(window),
        }
    }

    /// Returns (upper, middle, lower)
    pub fn update(&mut self, high: f64, low: f64) -> (f64, f64, f64) {
        self.high_buffer.push_back(high);
        self.low_buffer.push_back(low);

        if self.high_buffer.len() > self.window {
            self.high_buffer.pop_front();
            self.low_buffer.pop_front();
        }

        if self.high_buffer.len() < self.window {
            (f64::NAN, f64::NAN, f64::NAN)
        } else {
            let upper = self.high_buffer.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let lower = self.low_buffer.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let middle = (upper + lower) / 2.0;

            (upper, middle, lower)
        }
    }

    pub fn reset(&mut self) {
        self.high_buffer.clear();
        self.low_buffer.clear();
    }
}

// ============================================================================
// Ulcer Index
// ============================================================================
#[pyclass]
pub struct UlcerIndexStreaming {
    window: usize,
    close_buffer: VecDeque<f64>,
}

#[pymethods]
impl UlcerIndexStreaming {
    #[new]
    pub fn new(window: usize) -> Self {
        Self {
            window,
            close_buffer: VecDeque::with_capacity(window),
        }
    }

    pub fn update(&mut self, value: f64) -> f64 {
        self.close_buffer.push_back(value);

        if self.close_buffer.len() > self.window {
            self.close_buffer.pop_front();
        }

        if self.close_buffer.len() < self.window {
            f64::NAN
        } else {
            let close_vec: Vec<f64> = self.close_buffer.iter().copied().collect();
            let mut pct_drawdown_sq = vec![0.0; close_vec.len()];

            for i in 1..close_vec.len() {
                let max_close = close_vec[..=i].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                if max_close > 0.0 {
                    let pct_drawdown = ((close_vec[i] - max_close) / max_close) * 100.0;
                    pct_drawdown_sq[i] = pct_drawdown.powi(2);
                }
            }

            let mean_sq: f64 = pct_drawdown_sq.iter().sum::<f64>() / pct_drawdown_sq.len() as f64;
            mean_sq.sqrt()
        }
    }

    pub fn reset(&mut self) {
        self.close_buffer.clear();
    }
}
