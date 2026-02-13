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

// ============================================================================
// Standard Deviation (rolling, population)
// ============================================================================
#[pyclass]
pub struct StandardDeviationStreaming {
    window: usize,
    buffer: VecDeque<f64>,
}

#[pymethods]
impl StandardDeviationStreaming {
    #[new]
    pub fn new(window: usize) -> Self {
        Self {
            window,
            buffer: VecDeque::with_capacity(window),
        }
    }

    pub fn update(&mut self, value: f64) -> f64 {
        self.buffer.push_back(value);

        if self.buffer.len() > self.window {
            self.buffer.pop_front();
        }

        if self.buffer.len() < self.window {
            f64::NAN
        } else {
            let mean: f64 = self.buffer.iter().sum::<f64>() / self.window as f64;
            let variance: f64 = self.buffer.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / self.window as f64;
            variance.sqrt()
        }
    }

    pub fn reset(&mut self) {
        self.buffer.clear();
    }
}

// ============================================================================
// Variance (rolling, population)
// ============================================================================
#[pyclass]
pub struct VarianceStreaming {
    window: usize,
    buffer: VecDeque<f64>,
}

#[pymethods]
impl VarianceStreaming {
    #[new]
    pub fn new(window: usize) -> Self {
        Self {
            window,
            buffer: VecDeque::with_capacity(window),
        }
    }

    pub fn update(&mut self, value: f64) -> f64 {
        self.buffer.push_back(value);

        if self.buffer.len() > self.window {
            self.buffer.pop_front();
        }

        if self.buffer.len() < self.window {
            f64::NAN
        } else {
            let mean: f64 = self.buffer.iter().sum::<f64>() / self.window as f64;
            self.buffer.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / self.window as f64
        }
    }

    pub fn reset(&mut self) {
        self.buffer.clear();
    }
}

// ============================================================================
// Range (rolling High - Low range)
// ============================================================================
#[pyclass]
pub struct RangeStreaming {
    window: usize,
    high_buffer: VecDeque<f64>,
    low_buffer: VecDeque<f64>,
}

#[pymethods]
impl RangeStreaming {
    #[new]
    pub fn new(window: usize) -> Self {
        Self {
            window,
            high_buffer: VecDeque::with_capacity(window),
            low_buffer: VecDeque::with_capacity(window),
        }
    }

    pub fn update(&mut self, high: f64, low: f64) -> f64 {
        self.high_buffer.push_back(high);
        self.low_buffer.push_back(low);

        if self.high_buffer.len() > self.window {
            self.high_buffer.pop_front();
            self.low_buffer.pop_front();
        }

        if self.high_buffer.len() < self.window {
            f64::NAN
        } else {
            let max_high = self.high_buffer.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let min_low = self.low_buffer.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            max_high - min_low
        }
    }

    pub fn reset(&mut self) {
        self.high_buffer.clear();
        self.low_buffer.clear();
    }
}

// ============================================================================
// Historical Volatility (annualized rolling std of log returns)
// ============================================================================
#[pyclass]
pub struct HistoricalVolatilityStreaming {
    window: usize,
    annualize: bool,
    prev_value: f64,
    returns_buffer: VecDeque<f64>,
    update_count: usize,
}

#[pymethods]
impl HistoricalVolatilityStreaming {
    #[new]
    #[pyo3(signature = (window=20, annualize=true))]
    pub fn new(window: usize, annualize: bool) -> Self {
        Self {
            window,
            annualize,
            prev_value: f64::NAN,
            returns_buffer: VecDeque::with_capacity(window),
            update_count: 0,
        }
    }

    pub fn update(&mut self, value: f64) -> f64 {
        self.update_count += 1;

        if self.update_count == 1 {
            self.prev_value = value;
            return f64::NAN;
        }

        if self.prev_value > 0.0 && value > 0.0 {
            let log_return = (value / self.prev_value).ln();
            self.returns_buffer.push_back(log_return);

            if self.returns_buffer.len() > self.window {
                self.returns_buffer.pop_front();
            }
        }

        self.prev_value = value;

        if self.returns_buffer.len() < self.window {
            f64::NAN
        } else {
            let n = self.returns_buffer.len() as f64;
            let mean: f64 = self.returns_buffer.iter().sum::<f64>() / n;
            let variance: f64 = self.returns_buffer.iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f64>() / (n - 1.0);
            let mut volatility = variance.sqrt();

            if self.annualize {
                volatility *= (252.0_f64).sqrt();
            }

            volatility
        }
    }

    pub fn reset(&mut self) {
        self.prev_value = f64::NAN;
        self.returns_buffer.clear();
        self.update_count = 0;
    }
}
