use pyo3::prelude::*;
use std::collections::VecDeque;
use super::trend::{SMAStreaming, EMAStreaming};

// ============================================================================
// RSI (Relative Strength Index)
// ============================================================================
#[pyclass]
pub struct RSIStreaming {
    window: usize,
    alpha: f64,
    prev_close: f64,
    avg_gain: f64,
    avg_loss: f64,
    update_count: usize,
}

#[pymethods]
impl RSIStreaming {
    #[new]
    pub fn new(window: usize) -> Self {
        Self {
            window,
            alpha: 1.0 / window as f64,
            prev_close: f64::NAN,
            avg_gain: f64::NAN,
            avg_loss: f64::NAN,
            update_count: 0,
        }
    }

    pub fn update(&mut self, value: f64) -> f64 {
        self.update_count += 1;

        if self.update_count == 1 {
            self.prev_close = value;
            return f64::NAN;
        }

        let change = value - self.prev_close;
        let (current_gain, current_loss) = if change > 0.0 {
            (change, 0.0)
        } else {
            (0.0, -change)
        };

        if self.avg_gain.is_nan() {
            self.avg_gain = current_gain;
            self.avg_loss = current_loss;
        } else {
            self.avg_gain = self.alpha * current_gain + (1.0 - self.alpha) * self.avg_gain;
            self.avg_loss = self.alpha * current_loss + (1.0 - self.alpha) * self.avg_loss;
        }

        let rsi = if self.avg_loss == 0.0 {
            100.0
        } else {
            let rs = self.avg_gain / self.avg_loss;
            100.0 - (100.0 / (1.0 + rs))
        };

        self.prev_close = value;

        if self.update_count >= self.window {
            rsi
        } else {
            f64::NAN
        }
    }

    pub fn reset(&mut self) {
        self.prev_close = f64::NAN;
        self.avg_gain = f64::NAN;
        self.avg_loss = f64::NAN;
        self.update_count = 0;
    }
}

// ============================================================================
// Stochastic Oscillator
// ============================================================================
#[pyclass]
pub struct StochasticStreaming {
    k_period: usize,
    d_period: usize,
    high_buffer: VecDeque<f64>,
    low_buffer: VecDeque<f64>,
    percent_k_buffer: VecDeque<f64>,
}

#[pymethods]
impl StochasticStreaming {
    #[new]
    pub fn new(k_period: usize, d_period: usize) -> Self {
        Self {
            k_period,
            d_period,
            high_buffer: VecDeque::with_capacity(k_period),
            low_buffer: VecDeque::with_capacity(k_period),
            percent_k_buffer: VecDeque::with_capacity(d_period),
        }
    }

    /// Returns (percent_k, percent_d)
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> (f64, f64) {
        self.high_buffer.push_back(high);
        self.low_buffer.push_back(low);

        if self.high_buffer.len() > self.k_period {
            self.high_buffer.pop_front();
            self.low_buffer.pop_front();
        }

        if self.high_buffer.len() < self.k_period {
            return (f64::NAN, f64::NAN);
        }

        let highest_high = self.high_buffer.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let lowest_low = self.low_buffer.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        let percent_k = if highest_high != lowest_low {
            100.0 * (close - lowest_low) / (highest_high - lowest_low)
        } else {
            0.0
        };

        self.percent_k_buffer.push_back(percent_k);
        if self.percent_k_buffer.len() > self.d_period {
            self.percent_k_buffer.pop_front();
        }

        let percent_d = if self.percent_k_buffer.len() >= self.d_period {
            self.percent_k_buffer.iter().sum::<f64>() / self.percent_k_buffer.len() as f64
        } else {
            f64::NAN
        };

        (percent_k, percent_d)
    }

    pub fn reset(&mut self) {
        self.high_buffer.clear();
        self.low_buffer.clear();
        self.percent_k_buffer.clear();
    }
}

// ============================================================================
// Williams %R
// ============================================================================
#[pyclass]
pub struct WilliamsRStreaming {
    window: usize,
    high_buffer: VecDeque<f64>,
    low_buffer: VecDeque<f64>,
}

#[pymethods]
impl WilliamsRStreaming {
    #[new]
    pub fn new(window: usize) -> Self {
        Self {
            window,
            high_buffer: VecDeque::with_capacity(window),
            low_buffer: VecDeque::with_capacity(window),
        }
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> f64 {
        self.high_buffer.push_back(high);
        self.low_buffer.push_back(low);

        if self.high_buffer.len() > self.window {
            self.high_buffer.pop_front();
            self.low_buffer.pop_front();
        }

        if self.high_buffer.len() < self.window {
            f64::NAN
        } else {
            let highest_high = self.high_buffer.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let lowest_low = self.low_buffer.iter().fold(f64::INFINITY, |a, &b| a.min(b));

            if highest_high != lowest_low {
                -100.0 * (highest_high - close) / (highest_high - lowest_low)
            } else {
                -100.0
            }
        }
    }

    pub fn reset(&mut self) {
        self.high_buffer.clear();
        self.low_buffer.clear();
    }
}

// ============================================================================
// ROC (Rate of Change)
// ============================================================================
#[pyclass]
pub struct ROCStreaming {
    window: usize,
    buffer: VecDeque<f64>,
}

#[pymethods]
impl ROCStreaming {
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
            let old_value = self.buffer[0];
            if old_value != 0.0 {
                (value - old_value) / old_value * 100.0
            } else {
                0.0
            }
        }
    }

    pub fn reset(&mut self) {
        self.buffer.clear();
    }
}

// ============================================================================
// PPO (Percentage Price Oscillator)
// ============================================================================
#[pyclass]
pub struct PPOStreaming {
    fast_ema: EMAStreaming,
    slow_ema: EMAStreaming,
    signal_ema: EMAStreaming,
}

#[pymethods]
impl PPOStreaming {
    #[new]
    pub fn new(fast_period: usize, slow_period: usize, signal_period: usize) -> Self {
        Self {
            fast_ema: EMAStreaming::new(fast_period),
            slow_ema: EMAStreaming::new(slow_period),
            signal_ema: EMAStreaming::new(signal_period),
        }
    }

    /// Returns (ppo, signal, histogram)
    pub fn update(&mut self, value: f64) -> (f64, f64, f64) {
        let fast = self.fast_ema.update(value);
        let slow = self.slow_ema.update(value);

        if fast.is_nan() || slow.is_nan() || slow == 0.0 {
            return (f64::NAN, f64::NAN, f64::NAN);
        }

        let ppo_line = ((fast - slow) / slow) * 100.0;
        let signal_line = self.signal_ema.update(ppo_line);
        let histogram = if signal_line.is_nan() {
            f64::NAN
        } else {
            ppo_line - signal_line
        };

        (ppo_line, signal_line, histogram)
    }

    pub fn reset(&mut self) {
        self.fast_ema.reset();
        self.slow_ema.reset();
        self.signal_ema.reset();
    }
}

// ============================================================================
// PVO (Percentage Volume Oscillator) - same as PPO but for volume
// ============================================================================
#[pyclass]
pub struct PVOStreaming {
    fast_ema: EMAStreaming,
    slow_ema: EMAStreaming,
    signal_ema: EMAStreaming,
}

#[pymethods]
impl PVOStreaming {
    #[new]
    pub fn new(fast_period: usize, slow_period: usize, signal_period: usize) -> Self {
        Self {
            fast_ema: EMAStreaming::new(fast_period),
            slow_ema: EMAStreaming::new(slow_period),
            signal_ema: EMAStreaming::new(signal_period),
        }
    }

    /// Returns (pvo, signal, histogram)
    pub fn update(&mut self, volume: f64) -> (f64, f64, f64) {
        let fast = self.fast_ema.update(volume);
        let slow = self.slow_ema.update(volume);

        if fast.is_nan() || slow.is_nan() || slow == 0.0 {
            return (f64::NAN, f64::NAN, f64::NAN);
        }

        let pvo_line = ((fast - slow) / slow) * 100.0;
        let signal_line = self.signal_ema.update(pvo_line);
        let histogram = if signal_line.is_nan() {
            f64::NAN
        } else {
            pvo_line - signal_line
        };

        (pvo_line, signal_line, histogram)
    }

    pub fn reset(&mut self) {
        self.fast_ema.reset();
        self.slow_ema.reset();
        self.signal_ema.reset();
    }
}

// ============================================================================
// Ultimate Oscillator
// ============================================================================
#[pyclass]
pub struct UltimateOscillatorStreaming {
    period1: usize,
    period2: usize,
    period3: usize,
    bp_buffer: VecDeque<f64>,
    tr_buffer: VecDeque<f64>,
    prev_close: f64,
    update_count: usize,
}

#[pymethods]
impl UltimateOscillatorStreaming {
    #[new]
    pub fn new(period1: usize, period2: usize, period3: usize) -> Self {
        Self {
            period1,
            period2,
            period3,
            bp_buffer: VecDeque::with_capacity(period3),
            tr_buffer: VecDeque::with_capacity(period3),
            prev_close: f64::NAN,
            update_count: 0,
        }
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> f64 {
        self.update_count += 1;

        let (bp, tr) = if !self.prev_close.is_nan() {
            let bp = close - low.min(self.prev_close);
            let tr = (high - low).max((high - self.prev_close).abs()).max((low - self.prev_close).abs());
            (bp, tr)
        } else {
            (close - low, high - low)
        };

        self.bp_buffer.push_back(bp);
        self.tr_buffer.push_back(tr);

        if self.bp_buffer.len() > self.period3 {
            self.bp_buffer.pop_front();
            self.tr_buffer.pop_front();
        }

        self.prev_close = close;

        if self.bp_buffer.len() < self.period3 {
            f64::NAN
        } else {
            let bp_vec: Vec<f64> = self.bp_buffer.iter().copied().collect();
            let tr_vec: Vec<f64> = self.tr_buffer.iter().copied().collect();

            let avg1 = bp_vec[bp_vec.len() - self.period1..].iter().sum::<f64>()
                     / tr_vec[tr_vec.len() - self.period1..].iter().sum::<f64>();
            let avg2 = bp_vec[bp_vec.len() - self.period2..].iter().sum::<f64>()
                     / tr_vec[tr_vec.len() - self.period2..].iter().sum::<f64>();
            let avg3 = bp_vec.iter().sum::<f64>() / tr_vec.iter().sum::<f64>();

            100.0 * ((4.0 * avg1) + (2.0 * avg2) + avg3) / 7.0
        }
    }

    pub fn reset(&mut self) {
        self.bp_buffer.clear();
        self.tr_buffer.clear();
        self.prev_close = f64::NAN;
        self.update_count = 0;
    }
}

// ============================================================================
// Stochastic RSI
// ============================================================================
#[pyclass]
pub struct StochasticRSIStreaming {
    rsi_stream: RSIStreaming,
    rsi_buffer: VecDeque<f64>,
    k_sma: SMAStreaming,
    d_sma: SMAStreaming,
    stoch_period: usize,
}

#[pymethods]
impl StochasticRSIStreaming {
    #[new]
    pub fn new(rsi_period: usize, stoch_period: usize, k_period: usize, d_period: usize) -> Self {
        Self {
            rsi_stream: RSIStreaming::new(rsi_period),
            rsi_buffer: VecDeque::with_capacity(stoch_period),
            k_sma: SMAStreaming::new(k_period),
            d_sma: SMAStreaming::new(d_period),
            stoch_period,
        }
    }

    /// Returns (stochrsi, stochrsi_k, stochrsi_d)
    pub fn update(&mut self, value: f64) -> (f64, f64, f64) {
        let rsi_value = self.rsi_stream.update(value);

        if rsi_value.is_nan() {
            return (f64::NAN, f64::NAN, f64::NAN);
        }

        self.rsi_buffer.push_back(rsi_value);
        if self.rsi_buffer.len() > self.stoch_period {
            self.rsi_buffer.pop_front();
        }

        if self.rsi_buffer.len() < self.stoch_period {
            return (f64::NAN, f64::NAN, f64::NAN);
        }

        let low_rsi = self.rsi_buffer.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let high_rsi = self.rsi_buffer.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let stoch_rsi = if high_rsi > low_rsi {
            (rsi_value - low_rsi) / (high_rsi - low_rsi)
        } else {
            0.0
        };

        let k_value = self.k_sma.update(stoch_rsi);
        let d_value = self.d_sma.update(k_value);

        (stoch_rsi, k_value, d_value)
    }

    pub fn reset(&mut self) {
        self.rsi_stream.reset();
        self.rsi_buffer.clear();
        self.k_sma.reset();
        self.d_sma.reset();
    }
}

// ============================================================================
// TSI (True Strength Index)
// ============================================================================
#[pyclass]
pub struct TSIStreaming {
    momentum_ema1: EMAStreaming,
    momentum_ema2: EMAStreaming,
    abs_momentum_ema1: EMAStreaming,
    abs_momentum_ema2: EMAStreaming,
    prev_close: f64,
    update_count: usize,
}

#[pymethods]
impl TSIStreaming {
    #[new]
    pub fn new(first_smooth: usize, second_smooth: usize) -> Self {
        Self {
            momentum_ema1: EMAStreaming::new(first_smooth),
            momentum_ema2: EMAStreaming::new(second_smooth),
            abs_momentum_ema1: EMAStreaming::new(first_smooth),
            abs_momentum_ema2: EMAStreaming::new(second_smooth),
            prev_close: f64::NAN,
            update_count: 0,
        }
    }

    pub fn update(&mut self, value: f64) -> f64 {
        self.update_count += 1;

        if self.update_count == 1 {
            self.prev_close = value;
            return f64::NAN;
        }

        let momentum = value - self.prev_close;
        let abs_momentum = momentum.abs();

        let smooth1_momentum = self.momentum_ema1.update(momentum);
        let smooth2_momentum = self.momentum_ema2.update(smooth1_momentum);

        let smooth1_abs = self.abs_momentum_ema1.update(abs_momentum);
        let smooth2_abs = self.abs_momentum_ema2.update(smooth1_abs);

        self.prev_close = value;

        if !smooth2_momentum.is_nan() && !smooth2_abs.is_nan() && smooth2_abs != 0.0 {
            100.0 * (smooth2_momentum / smooth2_abs)
        } else {
            f64::NAN
        }
    }

    pub fn reset(&mut self) {
        self.momentum_ema1.reset();
        self.momentum_ema2.reset();
        self.abs_momentum_ema1.reset();
        self.abs_momentum_ema2.reset();
        self.prev_close = f64::NAN;
        self.update_count = 0;
    }
}

// ============================================================================
// Awesome Oscillator
// ============================================================================
#[pyclass]
pub struct AwesomeOscillatorStreaming {
    fast_sma: SMAStreaming,
    slow_sma: SMAStreaming,
}

#[pymethods]
impl AwesomeOscillatorStreaming {
    #[new]
    pub fn new(fast_period: usize, slow_period: usize) -> Self {
        Self {
            fast_sma: SMAStreaming::new(fast_period),
            slow_sma: SMAStreaming::new(slow_period),
        }
    }

    pub fn update(&mut self, high: f64, low: f64) -> f64 {
        let midpoint = (high + low) / 2.0;
        let fast = self.fast_sma.update(midpoint);
        let slow = self.slow_sma.update(midpoint);

        if fast.is_nan() || slow.is_nan() {
            f64::NAN
        } else {
            fast - slow
        }
    }

    pub fn reset(&mut self) {
        self.fast_sma.reset();
        self.slow_sma.reset();
    }
}

// ============================================================================
// KAMA (Kaufman's Adaptive Moving Average)
// ============================================================================
#[pyclass]
pub struct KAMAStreaming {
    window: usize,
    fast_sc: f64,
    slow_sc: f64,
    price_buffer: VecDeque<f64>,
    prev_kama: f64,
}

#[pymethods]
impl KAMAStreaming {
    #[new]
    pub fn new(window: usize, fast_period: usize, slow_period: usize) -> Self {
        let fast_sc = 2.0 / (fast_period as f64 + 1.0);
        let slow_sc = 2.0 / (slow_period as f64 + 1.0);

        Self {
            window,
            fast_sc,
            slow_sc,
            price_buffer: VecDeque::with_capacity(window + 1),
            prev_kama: f64::NAN,
        }
    }

    pub fn update(&mut self, value: f64) -> f64 {
        self.price_buffer.push_back(value);

        if self.price_buffer.len() > self.window + 1 {
            self.price_buffer.pop_front();
        }

        if self.price_buffer.len() < self.window + 1 {
            return f64::NAN;
        }

        let price_vec: Vec<f64> = self.price_buffer.iter().copied().collect();
        let direction = (price_vec[price_vec.len() - 1] - price_vec[0]).abs();

        let volatility: f64 = price_vec.windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .sum();

        let er = if volatility > 0.0 {
            direction / volatility
        } else {
            0.0
        };

        let sc = (er * (self.fast_sc - self.slow_sc) + self.slow_sc).powi(2);

        let result = if self.prev_kama.is_nan() {
            value
        } else {
            self.prev_kama + sc * (value - self.prev_kama)
        };

        self.prev_kama = result;
        result
    }

    pub fn reset(&mut self) {
        self.price_buffer.clear();
        self.prev_kama = f64::NAN;
    }
}

// ============================================================================
// Momentum (simple)
// ============================================================================
#[pyclass]
pub struct MomentumStreaming {
    window: usize,
    buffer: VecDeque<f64>,
}

#[pymethods]
impl MomentumStreaming {
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
            value - self.buffer[0]
        }
    }

    pub fn reset(&mut self) {
        self.buffer.clear();
    }
}
