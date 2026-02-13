use pyo3::prelude::*;
use std::collections::VecDeque;
use super::trend::{EMAStreaming, SMAStreaming};

// ============================================================================
// MFI (Money Flow Index)
// ============================================================================
#[pyclass]
#[pyo3(name = "MoneyFlowIndexStreaming")]
pub struct MFIStreaming {
    window: usize,
    positive_mf_buffer: VecDeque<f64>,
    negative_mf_buffer: VecDeque<f64>,
    prev_tp: f64,
}

#[pymethods]
impl MFIStreaming {
    #[new]
    pub fn new(window: usize) -> Self {
        Self {
            window,
            positive_mf_buffer: VecDeque::with_capacity(window),
            negative_mf_buffer: VecDeque::with_capacity(window),
            prev_tp: f64::NAN,
        }
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> f64 {
        let typical_price = (high + low + close) / 3.0;
        let rmf = typical_price * volume;

        let (positive_mf, negative_mf) = if self.prev_tp.is_nan() {
            (0.0, 0.0)
        } else if typical_price > self.prev_tp {
            (rmf, 0.0)
        } else if typical_price < self.prev_tp {
            (0.0, rmf)
        } else {
            (0.0, 0.0)
        };

        self.positive_mf_buffer.push_back(positive_mf);
        self.negative_mf_buffer.push_back(negative_mf);

        if self.positive_mf_buffer.len() > self.window {
            self.positive_mf_buffer.pop_front();
            self.negative_mf_buffer.pop_front();
        }

        self.prev_tp = typical_price;

        if self.positive_mf_buffer.len() < self.window {
            f64::NAN
        } else {
            let pos_sum: f64 = self.positive_mf_buffer.iter().sum();
            let neg_sum: f64 = self.negative_mf_buffer.iter().sum();

            if neg_sum == 0.0 {
                100.0
            } else {
                let mfr = pos_sum / neg_sum;
                100.0 - (100.0 / (1.0 + mfr))
            }
        }
    }

    pub fn reset(&mut self) {
        self.positive_mf_buffer.clear();
        self.negative_mf_buffer.clear();
        self.prev_tp = f64::NAN;
    }
}

// ============================================================================
// Accumulation/Distribution Index
// ============================================================================
#[pyclass]
#[pyo3(name = "AccDistIndexStreaming")]
pub struct AccDistStreaming {
    ad_line: f64,
}

#[pymethods]
impl AccDistStreaming {
    #[new]
    pub fn new() -> Self {
        Self {
            ad_line: 0.0,
        }
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> f64 {
        let mfm = if high != low {
            ((close - low) - (high - close)) / (high - low)
        } else {
            0.0
        };

        let mfv = mfm * volume;
        self.ad_line += mfv;

        self.ad_line
    }

    pub fn reset(&mut self) {
        self.ad_line = 0.0;
    }
}

// ============================================================================
// OBV (On-Balance Volume)
// ============================================================================
#[pyclass]
#[pyo3(name = "OnBalanceVolumeStreaming")]
pub struct OBVStreaming {
    obv_line: f64,
    prev_close: f64,
    update_count: usize,
}

#[pymethods]
impl OBVStreaming {
    #[new]
    pub fn new() -> Self {
        Self {
            obv_line: 0.0,
            prev_close: f64::NAN,
            update_count: 0,
        }
    }

    pub fn update(&mut self, close: f64, volume: f64) -> f64 {
        self.update_count += 1;

        if self.update_count == 1 {
            self.obv_line = volume;
        } else {
            if close > self.prev_close {
                self.obv_line += volume;
            } else if close < self.prev_close {
                self.obv_line -= volume;
            }
        }

        self.prev_close = close;
        self.obv_line
    }

    pub fn reset(&mut self) {
        self.obv_line = 0.0;
        self.prev_close = f64::NAN;
        self.update_count = 0;
    }
}

// ============================================================================
// CMF (Chaikin Money Flow)
// ============================================================================
#[pyclass]
#[pyo3(name = "ChaikinMoneyFlowStreaming")]
pub struct CMFStreaming {
    window: usize,
    mfv_buffer: VecDeque<f64>,
    volume_buffer: VecDeque<f64>,
}

#[pymethods]
impl CMFStreaming {
    #[new]
    pub fn new(window: usize) -> Self {
        Self {
            window,
            mfv_buffer: VecDeque::with_capacity(window),
            volume_buffer: VecDeque::with_capacity(window),
        }
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> f64 {
        let mfm = if high != low {
            ((close - low) - (high - close)) / (high - low)
        } else {
            0.0
        };

        let mfv = mfm * volume;

        self.mfv_buffer.push_back(mfv);
        self.volume_buffer.push_back(volume);

        if self.mfv_buffer.len() > self.window {
            self.mfv_buffer.pop_front();
            self.volume_buffer.pop_front();
        }

        if self.mfv_buffer.len() < self.window {
            f64::NAN
        } else {
            let sum_mfv: f64 = self.mfv_buffer.iter().sum();
            let sum_volume: f64 = self.volume_buffer.iter().sum();

            if sum_volume != 0.0 {
                sum_mfv / sum_volume
            } else {
                0.0
            }
        }
    }

    pub fn reset(&mut self) {
        self.mfv_buffer.clear();
        self.volume_buffer.clear();
    }
}

// ============================================================================
// Force Index
// ============================================================================
#[pyclass]
pub struct ForceIndexStreaming {
    window: usize,
    alpha: f64,
    prev_close: f64,
    current_value: f64,
    update_count: usize,
}

#[pymethods]
impl ForceIndexStreaming {
    #[new]
    pub fn new(window: usize) -> Self {
        Self {
            window,
            alpha: 2.0 / (window as f64 + 1.0),
            prev_close: f64::NAN,
            current_value: f64::NAN,
            update_count: 0,
        }
    }

    pub fn update(&mut self, close: f64, volume: f64) -> f64 {
        self.update_count += 1;

        if self.update_count == 1 {
            self.prev_close = close;
            return f64::NAN;
        }

        let force_value = (close - self.prev_close) * volume;

        if self.current_value.is_nan() {
            self.current_value = force_value;
        } else {
            self.current_value = self.alpha * force_value + (1.0 - self.alpha) * self.current_value;
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
// EOM (Ease of Movement)
// ============================================================================
#[pyclass]
#[pyo3(name = "EaseOfMovementStreaming")]
pub struct EOMStreaming {
    prev_high: f64,
    prev_low: f64,
    update_count: usize,
}

#[pymethods]
impl EOMStreaming {
    #[new]
    pub fn new() -> Self {
        Self {
            prev_high: f64::NAN,
            prev_low: f64::NAN,
            update_count: 0,
        }
    }

    pub fn update(&mut self, high: f64, low: f64, volume: f64) -> f64 {
        self.update_count += 1;

        if self.update_count == 1 {
            self.prev_high = high;
            self.prev_low = low;
            return f64::NAN;
        }

        let result = if volume != 0.0 {
            let distance_moved = ((high - self.prev_high) + (low - self.prev_low)) / 2.0;
            let box_height = high - low;
            distance_moved * box_height / volume * 100_000_000.0
        } else {
            f64::NAN
        };

        self.prev_high = high;
        self.prev_low = low;

        result
    }

    pub fn reset(&mut self) {
        self.prev_high = f64::NAN;
        self.prev_low = f64::NAN;
        self.update_count = 0;
    }
}

// ============================================================================
// VPT (Volume Price Trend)
// ============================================================================
#[pyclass]
#[pyo3(name = "VolumePriceTrendStreaming")]
pub struct VPTStreaming {
    vpt_line: f64,
    prev_close: f64,
    update_count: usize,
}

#[pymethods]
impl VPTStreaming {
    #[new]
    pub fn new() -> Self {
        Self {
            vpt_line: 0.0,
            prev_close: f64::NAN,
            update_count: 0,
        }
    }

    pub fn update(&mut self, close: f64, volume: f64) -> f64 {
        self.update_count += 1;

        if self.update_count == 1 {
            self.vpt_line = 0.0;
        } else if self.prev_close != 0.0 {
            let pct_change = (close - self.prev_close) / self.prev_close;
            let vpt_change = volume * pct_change;
            self.vpt_line += vpt_change;
        }

        self.prev_close = close;
        self.vpt_line
    }

    pub fn reset(&mut self) {
        self.vpt_line = 0.0;
        self.prev_close = f64::NAN;
        self.update_count = 0;
    }
}

// ============================================================================
// NVI (Negative Volume Index)
// ============================================================================
#[pyclass]
#[pyo3(name = "NegativeVolumeIndexStreaming")]
pub struct NVIStreaming {
    nvi_line: f64,
    prev_close: f64,
    prev_volume: f64,
    update_count: usize,
}

#[pymethods]
impl NVIStreaming {
    #[new]
    pub fn new() -> Self {
        Self {
            nvi_line: 1000.0,
            prev_close: f64::NAN,
            prev_volume: f64::NAN,
            update_count: 0,
        }
    }

    pub fn update(&mut self, close: f64, volume: f64) -> f64 {
        self.update_count += 1;

        if self.update_count == 1 {
            self.nvi_line = 1000.0;
        } else if volume < self.prev_volume && self.prev_close != 0.0 {
            let pct_change = (close - self.prev_close) / self.prev_close;
            self.nvi_line *= 1.0 + pct_change;
        }

        self.prev_close = close;
        self.prev_volume = volume;

        self.nvi_line
    }

    pub fn reset(&mut self) {
        self.nvi_line = 1000.0;
        self.prev_close = f64::NAN;
        self.prev_volume = f64::NAN;
        self.update_count = 0;
    }
}

// ============================================================================
// VWAP (Volume Weighted Average Price)
// ============================================================================
#[pyclass]
pub struct VWAPStreaming {
    window: usize,
    tpv_buffer: VecDeque<f64>,
    volume_buffer: VecDeque<f64>,
}

#[pymethods]
impl VWAPStreaming {
    #[new]
    pub fn new(window: usize) -> Self {
        Self {
            window,
            tpv_buffer: VecDeque::with_capacity(window),
            volume_buffer: VecDeque::with_capacity(window),
        }
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> f64 {
        let typical_price = (high + low + close) / 3.0;
        let tpv = typical_price * volume;

        self.tpv_buffer.push_back(tpv);
        self.volume_buffer.push_back(volume);

        if self.tpv_buffer.len() > self.window {
            self.tpv_buffer.pop_front();
            self.volume_buffer.pop_front();
        }

        if self.tpv_buffer.len() < self.window {
            f64::NAN
        } else {
            let sum_tpv: f64 = self.tpv_buffer.iter().sum();
            let sum_volume: f64 = self.volume_buffer.iter().sum();

            if sum_volume != 0.0 {
                sum_tpv / sum_volume
            } else {
                0.0
            }
        }
    }

    pub fn reset(&mut self) {
        self.tpv_buffer.clear();
        self.volume_buffer.clear();
    }
}

// ============================================================================
// VWEMA (Volume Weighted EMA)
// ============================================================================
#[pyclass]
pub struct VWEMAStreaming {
    vwap_stream: VWAPStreaming,
    ema_stream: EMAStreaming,
}

#[pymethods]
impl VWEMAStreaming {
    #[new]
    pub fn new(vwma_period: usize, ema_period: usize) -> Self {
        Self {
            vwap_stream: VWAPStreaming::new(vwma_period),
            ema_stream: EMAStreaming::new(ema_period),
        }
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> f64 {
        let vwap_value = self.vwap_stream.update(high, low, close, volume);

        if vwap_value.is_nan() {
            f64::NAN
        } else {
            self.ema_stream.update(vwap_value)
        }
    }

    pub fn reset(&mut self) {
        self.vwap_stream.reset();
        self.ema_stream.reset();
    }
}

// ============================================================================
// Volume Ratio: volume / SMA(volume, window)
// ============================================================================
#[pyclass]
pub struct VolumeRatioStreaming {
    sma: SMAStreaming,
    #[allow(dead_code)]
    window: usize,
    update_count: usize,
}

#[pymethods]
impl VolumeRatioStreaming {
    #[new]
    pub fn new(window: usize) -> Self {
        Self {
            sma: SMAStreaming::new(window),
            window,
            update_count: 0,
        }
    }

    pub fn update(&mut self, volume: f64) -> f64 {
        self.update_count += 1;
        let sma_value = self.sma.update(volume);

        if sma_value.is_nan() || sma_value == 0.0 {
            f64::NAN
        } else {
            volume / sma_value
        }
    }

    pub fn reset(&mut self) {
        self.sma.reset();
        self.update_count = 0;
    }
}
