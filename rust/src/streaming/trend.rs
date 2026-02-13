use pyo3::prelude::*;
use std::collections::VecDeque;

// ============================================================================
// Simple Moving Average (SMA)
// ============================================================================
#[pyclass]
pub struct SMAStreaming {
    window: usize,
    buffer: VecDeque<f64>,
    sum: f64,
}

#[pymethods]
impl SMAStreaming {
    #[new]
    pub fn new(window: usize) -> Self {
        Self {
            window,
            buffer: VecDeque::with_capacity(window),
            sum: 0.0,
        }
    }

    pub fn update(&mut self, value: f64) -> f64 {
        if self.buffer.len() >= self.window {
            self.sum -= self.buffer.pop_front().unwrap();
        }
        self.buffer.push_back(value);
        self.sum += value;

        if self.buffer.len() < self.window {
            f64::NAN
        } else {
            self.sum / self.window as f64
        }
    }

    pub fn reset(&mut self) {
        self.buffer.clear();
        self.sum = 0.0;
    }
}

// ============================================================================
// Exponential Moving Average (EMA)
// ============================================================================
#[pyclass]
pub struct EMAStreaming {
    window: usize,
    alpha: f64,
    current_value: f64,
}

#[pymethods]
impl EMAStreaming {
    #[new]
    pub fn new(window: usize) -> Self {
        let alpha = 2.0 / (window as f64 + 1.0);
        Self {
            window,
            alpha,
            current_value: f64::NAN,
        }
    }

    pub fn update(&mut self, value: f64) -> f64 {
        if self.current_value.is_nan() {
            self.current_value = value;
        } else {
            self.current_value = self.alpha * value + (1.0 - self.alpha) * self.current_value;
        }
        self.current_value
    }

    pub fn reset(&mut self) {
        self.current_value = f64::NAN;
    }
}

// ============================================================================
// Weighted Moving Average (WMA)
// ============================================================================
#[pyclass]
pub struct WMAStreaming {
    window: usize,
    buffer: VecDeque<f64>,
    weights: Vec<f64>,
    sum_weights: f64,
}

#[pymethods]
impl WMAStreaming {
    #[new]
    pub fn new(window: usize) -> Self {
        let weights: Vec<f64> = (1..=window).map(|i| i as f64).collect();
        let sum_weights: f64 = weights.iter().sum();

        Self {
            window,
            buffer: VecDeque::with_capacity(window),
            weights,
            sum_weights,
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
            let weighted_sum: f64 = self.buffer.iter()
                .zip(self.weights.iter())
                .map(|(v, w)| v * w)
                .sum();
            weighted_sum / self.sum_weights
        }
    }

    pub fn reset(&mut self) {
        self.buffer.clear();
    }
}

// ============================================================================
// MACD (Moving Average Convergence Divergence)
// ============================================================================
#[pyclass]
pub struct MACDStreaming {
    fast_ema: EMAStreaming,
    slow_ema: EMAStreaming,
    signal_ema: EMAStreaming,
}

#[pymethods]
impl MACDStreaming {
    #[new]
    pub fn new(fast_period: usize, slow_period: usize, signal_period: usize) -> Self {
        Self {
            fast_ema: EMAStreaming::new(fast_period),
            slow_ema: EMAStreaming::new(slow_period),
            signal_ema: EMAStreaming::new(signal_period),
        }
    }

    /// Returns (macd_line, signal_line, histogram)
    pub fn update(&mut self, value: f64) -> (f64, f64, f64) {
        let fast = self.fast_ema.update(value);
        let slow = self.slow_ema.update(value);

        if fast.is_nan() || slow.is_nan() {
            return (f64::NAN, f64::NAN, f64::NAN);
        }

        let macd_line = fast - slow;
        let signal_line = self.signal_ema.update(macd_line);
        let histogram = if signal_line.is_nan() {
            f64::NAN
        } else {
            macd_line - signal_line
        };

        (macd_line, signal_line, histogram)
    }

    pub fn reset(&mut self) {
        self.fast_ema.reset();
        self.slow_ema.reset();
        self.signal_ema.reset();
    }
}

// ============================================================================
// ADX (Average Directional Index)
// ============================================================================
#[pyclass]
pub struct ADXStreaming {
    window: usize,
    alpha: f64,
    prev_high: f64,
    prev_low: f64,
    prev_close: f64,
    smoothed_plus_dm: f64,
    smoothed_minus_dm: f64,
    smoothed_tr: f64,
    smoothed_dx: f64,
    update_count: usize,
}

#[pymethods]
impl ADXStreaming {
    #[new]
    pub fn new(window: usize) -> Self {
        Self {
            window,
            alpha: 1.0 / window as f64,
            prev_high: f64::NAN,
            prev_low: f64::NAN,
            prev_close: f64::NAN,
            smoothed_plus_dm: f64::NAN,
            smoothed_minus_dm: f64::NAN,
            smoothed_tr: f64::NAN,
            smoothed_dx: f64::NAN,
            update_count: 0,
        }
    }

    /// Returns (adx, plus_di, minus_di)
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> (f64, f64, f64) {
        self.update_count += 1;

        if self.update_count == 1 {
            self.prev_high = high;
            self.prev_low = low;
            self.prev_close = close;
            return (f64::NAN, f64::NAN, f64::NAN);
        }

        // Calculate directional movement
        let high_diff = high - self.prev_high;
        let low_diff = self.prev_low - low;

        let plus_dm = if high_diff > low_diff && high_diff > 0.0 { high_diff } else { 0.0 };
        let minus_dm = if low_diff > high_diff && low_diff > 0.0 { low_diff } else { 0.0 };

        // Calculate true range
        let tr = (high - low).max((high - self.prev_close).abs()).max((low - self.prev_close).abs());

        // Smooth using Wilder's method
        if self.smoothed_plus_dm.is_nan() {
            self.smoothed_plus_dm = plus_dm;
            self.smoothed_minus_dm = minus_dm;
            self.smoothed_tr = tr;
        } else {
            self.smoothed_plus_dm = (1.0 - self.alpha) * self.smoothed_plus_dm + self.alpha * plus_dm;
            self.smoothed_minus_dm = (1.0 - self.alpha) * self.smoothed_minus_dm + self.alpha * minus_dm;
            self.smoothed_tr = (1.0 - self.alpha) * self.smoothed_tr + self.alpha * tr;
        }

        let mut adx = f64::NAN;
        let mut plus_di = f64::NAN;
        let mut minus_di = f64::NAN;

        if self.smoothed_tr > 0.0 {
            plus_di = 100.0 * (self.smoothed_plus_dm / self.smoothed_tr);
            minus_di = 100.0 * (self.smoothed_minus_dm / self.smoothed_tr);

            let di_sum = plus_di + minus_di;
            if di_sum > 0.0 {
                let dx = 100.0 * (plus_di - minus_di).abs() / di_sum;

                if self.smoothed_dx.is_nan() {
                    self.smoothed_dx = dx;
                } else {
                    self.smoothed_dx = (1.0 - self.alpha) * self.smoothed_dx + self.alpha * dx;
                }

                if self.update_count >= self.window {
                    adx = self.smoothed_dx;
                }
            }
        }

        self.prev_high = high;
        self.prev_low = low;
        self.prev_close = close;

        (adx, plus_di, minus_di)
    }

    pub fn reset(&mut self) {
        self.prev_high = f64::NAN;
        self.prev_low = f64::NAN;
        self.prev_close = f64::NAN;
        self.smoothed_plus_dm = f64::NAN;
        self.smoothed_minus_dm = f64::NAN;
        self.smoothed_tr = f64::NAN;
        self.smoothed_dx = f64::NAN;
        self.update_count = 0;
    }
}

// ============================================================================
// CCI (Commodity Channel Index)
// ============================================================================
#[pyclass]
pub struct CCIStreaming {
    window: usize,
    constant: f64,
    tp_buffer: VecDeque<f64>,
}

#[pymethods]
impl CCIStreaming {
    #[new]
    pub fn new(window: usize, constant: f64) -> Self {
        Self {
            window,
            constant,
            tp_buffer: VecDeque::with_capacity(window),
        }
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> f64 {
        let typical_price = (high + low + close) / 3.0;
        self.tp_buffer.push_back(typical_price);

        if self.tp_buffer.len() > self.window {
            self.tp_buffer.pop_front();
        }

        if self.tp_buffer.len() < self.window {
            f64::NAN
        } else {
            let sma: f64 = self.tp_buffer.iter().sum::<f64>() / self.window as f64;
            let mad: f64 = self.tp_buffer.iter().map(|tp| (tp - sma).abs()).sum::<f64>() / self.window as f64;

            if mad == 0.0 {
                0.0
            } else {
                (typical_price - sma) / (self.constant * mad)
            }
        }
    }

    pub fn reset(&mut self) {
        self.tp_buffer.clear();
    }
}

// ============================================================================
// DPO (Detrended Price Oscillator)
// ============================================================================
#[pyclass]
pub struct DPOStreaming {
    window: usize,
    displacement: usize,
    sma_stream: SMAStreaming,
    price_buffer: VecDeque<f64>,
}

#[pymethods]
impl DPOStreaming {
    #[new]
    pub fn new(window: usize) -> Self {
        let displacement = window / 2 + 1;
        Self {
            window,
            displacement,
            sma_stream: SMAStreaming::new(window),
            price_buffer: VecDeque::with_capacity(window),
        }
    }

    pub fn update(&mut self, value: f64) -> f64 {
        self.price_buffer.push_back(value);
        if self.price_buffer.len() > self.window {
            self.price_buffer.pop_front();
        }

        let sma_value = self.sma_stream.update(value);

        if self.price_buffer.len() >= self.displacement && !sma_value.is_nan() {
            let displaced_price = self.price_buffer[self.price_buffer.len() - self.displacement];
            displaced_price - sma_value
        } else {
            f64::NAN
        }
    }

    pub fn reset(&mut self) {
        self.sma_stream.reset();
        self.price_buffer.clear();
    }
}

// ============================================================================
// Vortex Indicator
// ============================================================================
#[pyclass]
#[pyo3(name = "VortexIndicatorStreaming")]
pub struct VortexStreaming {
    window: usize,
    vm_plus_buffer: VecDeque<f64>,
    vm_minus_buffer: VecDeque<f64>,
    tr_buffer: VecDeque<f64>,
    prev_high: f64,
    prev_low: f64,
    prev_close: f64,
    update_count: usize,
}

#[pymethods]
impl VortexStreaming {
    #[new]
    pub fn new(window: usize) -> Self {
        Self {
            window,
            vm_plus_buffer: VecDeque::with_capacity(window),
            vm_minus_buffer: VecDeque::with_capacity(window),
            tr_buffer: VecDeque::with_capacity(window),
            prev_high: f64::NAN,
            prev_low: f64::NAN,
            prev_close: f64::NAN,
            update_count: 0,
        }
    }

    /// Returns (vi_plus, vi_minus)
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> (f64, f64) {
        self.update_count += 1;

        if self.update_count == 1 {
            self.prev_high = high;
            self.prev_low = low;
            self.prev_close = close;
            return (f64::NAN, f64::NAN);
        }

        let vm_plus = (high - self.prev_low).abs();
        let vm_minus = (low - self.prev_high).abs();
        let tr = (high - low).max((high - self.prev_close).abs()).max((low - self.prev_close).abs());

        self.vm_plus_buffer.push_back(vm_plus);
        self.vm_minus_buffer.push_back(vm_minus);
        self.tr_buffer.push_back(tr);

        if self.vm_plus_buffer.len() > self.window {
            self.vm_plus_buffer.pop_front();
            self.vm_minus_buffer.pop_front();
            self.tr_buffer.pop_front();
        }

        let result = if self.vm_plus_buffer.len() >= self.window {
            let sum_vm_plus: f64 = self.vm_plus_buffer.iter().sum();
            let sum_vm_minus: f64 = self.vm_minus_buffer.iter().sum();
            let sum_tr: f64 = self.tr_buffer.iter().sum();

            if sum_tr > 0.0 {
                (sum_vm_plus / sum_tr, sum_vm_minus / sum_tr)
            } else {
                (f64::NAN, f64::NAN)
            }
        } else {
            (f64::NAN, f64::NAN)
        };

        self.prev_high = high;
        self.prev_low = low;
        self.prev_close = close;

        result
    }

    pub fn reset(&mut self) {
        self.vm_plus_buffer.clear();
        self.vm_minus_buffer.clear();
        self.tr_buffer.clear();
        self.prev_high = f64::NAN;
        self.prev_low = f64::NAN;
        self.prev_close = f64::NAN;
        self.update_count = 0;
    }
}

// ============================================================================
// TRIX
// ============================================================================
#[pyclass]
pub struct TRIXStreaming {
    ema1: EMAStreaming,
    ema2: EMAStreaming,
    ema3: EMAStreaming,
    prev_ema3: f64,
}

#[pymethods]
impl TRIXStreaming {
    #[new]
    pub fn new(window: usize) -> Self {
        Self {
            ema1: EMAStreaming::new(window),
            ema2: EMAStreaming::new(window),
            ema3: EMAStreaming::new(window),
            prev_ema3: f64::NAN,
        }
    }

    pub fn update(&mut self, value: f64) -> f64 {
        let ema1_val = self.ema1.update(value);
        let ema2_val = self.ema2.update(ema1_val);
        let ema3_val = self.ema3.update(ema2_val);

        let result = if !self.prev_ema3.is_nan() && !ema3_val.is_nan() && self.prev_ema3 != 0.0 {
            100.0 * (ema3_val - self.prev_ema3) / self.prev_ema3
        } else {
            f64::NAN
        };

        self.prev_ema3 = ema3_val;
        result
    }

    pub fn reset(&mut self) {
        self.ema1.reset();
        self.ema2.reset();
        self.ema3.reset();
        self.prev_ema3 = f64::NAN;
    }
}

// ============================================================================
// Aroon Indicator
// ============================================================================
#[pyclass]
pub struct AroonStreaming {
    window: usize,
    high_buffer: VecDeque<f64>,
    low_buffer: VecDeque<f64>,
}

#[pymethods]
impl AroonStreaming {
    #[new]
    pub fn new(window: usize) -> Self {
        Self {
            window,
            high_buffer: VecDeque::with_capacity(window + 1),
            low_buffer: VecDeque::with_capacity(window + 1),
        }
    }

    /// Returns (aroon_up, aroon_down)
    pub fn update(&mut self, high: f64, low: f64) -> (f64, f64) {
        self.high_buffer.push_back(high);
        self.low_buffer.push_back(low);

        if self.high_buffer.len() > self.window + 1 {
            self.high_buffer.pop_front();
            self.low_buffer.pop_front();
        }

        if self.high_buffer.len() < self.window + 1 {
            (f64::NAN, f64::NAN)
        } else {
            let max_high = self.high_buffer.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let min_low = self.low_buffer.iter().fold(f64::INFINITY, |a, &b| a.min(b));

            let max_idx = self.high_buffer.iter().rposition(|&h| h == max_high).unwrap();
            let min_idx = self.low_buffer.iter().rposition(|&l| l == min_low).unwrap();

            let periods_since_high = self.high_buffer.len() - 1 - max_idx;
            let periods_since_low = self.low_buffer.len() - 1 - min_idx;

            let aroon_up = ((self.window - periods_since_high) as f64 / self.window as f64) * 100.0;
            let aroon_down = ((self.window - periods_since_low) as f64 / self.window as f64) * 100.0;

            (aroon_up, aroon_down)
        }
    }

    pub fn reset(&mut self) {
        self.high_buffer.clear();
        self.low_buffer.clear();
    }
}

// ============================================================================
// Parabolic SAR
// ============================================================================
#[pyclass]
#[pyo3(name = "ParabolicSARStreaming")]
pub struct PSARStreaming {
    af_start: f64,
    af_inc: f64,
    af_max: f64,
    up_trend: bool,
    acceleration_factor: f64,
    up_trend_high: f64,
    down_trend_low: f64,
    prev_sar: f64,
    prev_high: f64,
    prev_low: f64,
    buffer: VecDeque<f64>,
    update_count: usize,
}

#[pymethods]
impl PSARStreaming {
    #[new]
    pub fn new(af_start: f64, af_inc: f64, af_max: f64) -> Self {
        Self {
            af_start,
            af_inc,
            af_max,
            up_trend: true,
            acceleration_factor: af_start,
            up_trend_high: f64::NAN,
            down_trend_low: f64::NAN,
            prev_sar: f64::NAN,
            prev_high: f64::NAN,
            prev_low: f64::NAN,
            buffer: VecDeque::with_capacity(2),
            update_count: 0,
        }
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> f64 {
        self.update_count += 1;

        if self.update_count == 1 {
            self.prev_sar = close;
            self.up_trend_high = high;
            self.down_trend_low = low;
            self.prev_high = high;
            self.prev_low = low;
            return close;
        }

        if self.update_count == 2 {
            self.prev_high = high;
            self.prev_low = low;
            return close;
        }

        let mut current_sar: f64;
        let mut reversal = false;

        if self.up_trend {
            current_sar = self.prev_sar + self.acceleration_factor * (self.up_trend_high - self.prev_sar);

            if low < current_sar {
                reversal = true;
                current_sar = self.up_trend_high;
                self.down_trend_low = low;
                self.acceleration_factor = self.af_start;
            } else {
                if high > self.up_trend_high {
                    self.up_trend_high = high;
                    self.acceleration_factor = (self.acceleration_factor + self.af_inc).min(self.af_max);
                }

                if self.prev_low < current_sar {
                    current_sar = self.prev_low;
                }
            }
        } else {
            current_sar = self.prev_sar - self.acceleration_factor * (self.prev_sar - self.down_trend_low);

            if high > current_sar {
                reversal = true;
                current_sar = self.down_trend_low;
                self.up_trend_high = high;
                self.acceleration_factor = self.af_start;
            } else {
                if low < self.down_trend_low {
                    self.down_trend_low = low;
                    self.acceleration_factor = (self.acceleration_factor + self.af_inc).min(self.af_max);
                }

                if self.prev_high > current_sar {
                    current_sar = self.prev_high;
                }
            }
        }

        self.up_trend = self.up_trend != reversal;
        self.prev_sar = current_sar;
        self.prev_high = high;
        self.prev_low = low;

        let store_val = if self.up_trend { low } else { high };
        self.buffer.push_back(store_val);
        if self.buffer.len() > 2 {
            self.buffer.pop_front();
        }

        current_sar
    }

    pub fn reset(&mut self) {
        self.up_trend = true;
        self.acceleration_factor = self.af_start;
        self.up_trend_high = f64::NAN;
        self.down_trend_low = f64::NAN;
        self.prev_sar = f64::NAN;
        self.prev_high = f64::NAN;
        self.prev_low = f64::NAN;
        self.buffer.clear();
        self.update_count = 0;
    }
}
