use pyo3::prelude::*;
use std::collections::VecDeque;

// ============================================================================
// Daily Return
// ============================================================================
#[pyclass]
pub struct DailyReturnStreaming {
    prev_close: f64,
    update_count: usize,
}

#[pymethods]
impl DailyReturnStreaming {
    #[new]
    pub fn new() -> Self {
        Self {
            prev_close: f64::NAN,
            update_count: 0,
        }
    }

    pub fn update(&mut self, close: f64) -> f64 {
        self.update_count += 1;

        if self.update_count == 1 {
            self.prev_close = close;
            return f64::NAN;
        }

        let result = if self.prev_close != 0.0 {
            (close - self.prev_close) / self.prev_close * 100.0
        } else {
            f64::NAN
        };

        self.prev_close = close;
        result
    }

    pub fn reset(&mut self) {
        self.prev_close = f64::NAN;
        self.update_count = 0;
    }
}

// ============================================================================
// Daily Log Return
// ============================================================================
#[pyclass]
pub struct DailyLogReturnStreaming {
    prev_close: f64,
    update_count: usize,
}

#[pymethods]
impl DailyLogReturnStreaming {
    #[new]
    pub fn new() -> Self {
        Self {
            prev_close: f64::NAN,
            update_count: 0,
        }
    }

    pub fn update(&mut self, close: f64) -> f64 {
        self.update_count += 1;

        if self.update_count == 1 {
            self.prev_close = close;
            return f64::NAN;
        }

        let result = if self.prev_close > 0.0 && close > 0.0 {
            (close / self.prev_close).ln() * 100.0
        } else {
            f64::NAN
        };

        self.prev_close = close;
        result
    }

    pub fn reset(&mut self) {
        self.prev_close = f64::NAN;
        self.update_count = 0;
    }
}

// ============================================================================
// Cumulative Return
// ============================================================================
#[pyclass]
pub struct CumulativeReturnStreaming {
    initial_price: f64,
    update_count: usize,
}

#[pymethods]
impl CumulativeReturnStreaming {
    #[new]
    pub fn new() -> Self {
        Self {
            initial_price: f64::NAN,
            update_count: 0,
        }
    }

    pub fn update(&mut self, close: f64) -> f64 {
        self.update_count += 1;

        if self.update_count == 1 {
            self.initial_price = close;
            return 0.0;
        }

        if self.initial_price != 0.0 {
            ((close / self.initial_price) - 1.0) * 100.0
        } else {
            f64::NAN
        }
    }

    pub fn reset(&mut self) {
        self.initial_price = f64::NAN;
        self.update_count = 0;
    }
}

// ============================================================================
// Rolling Return
// ============================================================================
#[pyclass]
pub struct RollingReturnStreaming {
    window: usize,
    close_buffer: VecDeque<f64>,
}

#[pymethods]
impl RollingReturnStreaming {
    #[new]
    pub fn new(window: usize) -> Self {
        Self {
            window,
            close_buffer: VecDeque::with_capacity(window),
        }
    }

    pub fn update(&mut self, close: f64) -> f64 {
        self.close_buffer.push_back(close);

        if self.close_buffer.len() > self.window {
            self.close_buffer.pop_front();
        }

        if self.close_buffer.len() < self.window {
            f64::NAN
        } else {
            let start_price = self.close_buffer[0];
            let end_price = self.close_buffer[self.close_buffer.len() - 1];

            if start_price != 0.0 {
                (end_price - start_price) / start_price * 100.0
            } else {
                0.0
            }
        }
    }

    pub fn reset(&mut self) {
        self.close_buffer.clear();
    }
}

// ============================================================================
// Maximum Drawdown
// ============================================================================
#[pyclass]
pub struct MaxDrawdownStreaming {
    window: usize,
    close_buffer: VecDeque<f64>,
}

#[pymethods]
impl MaxDrawdownStreaming {
    #[new]
    pub fn new(window: usize) -> Self {
        Self {
            window,
            close_buffer: VecDeque::with_capacity(window),
        }
    }

    pub fn update(&mut self, close: f64) -> f64 {
        self.close_buffer.push_back(close);

        if self.close_buffer.len() > self.window {
            self.close_buffer.pop_front();
        }

        if self.close_buffer.len() < 2 {
            f64::NAN
        } else {
            let close_vec: Vec<f64> = self.close_buffer.iter().copied().collect();
            let mut running_max = close_vec[0];
            let mut max_drawdown: f64 = 0.0;

            for &price in close_vec.iter().skip(1) {
                running_max = running_max.max(price);
                let drawdown = (price - running_max) / running_max;
                max_drawdown = max_drawdown.min(drawdown);
            }

            max_drawdown * 100.0
        }
    }

    pub fn reset(&mut self) {
        self.close_buffer.clear();
    }
}

// ============================================================================
// Sharpe Ratio
// ============================================================================
#[pyclass]
pub struct SharpeRatioStreaming {
    window: usize,
    risk_free_rate: f64,
    annualization_factor: f64,
    prev_close: f64,
    returns_buffer: VecDeque<f64>,
    update_count: usize,
}

#[pymethods]
impl SharpeRatioStreaming {
    #[new]
    pub fn new(window: usize, risk_free_rate: f64, annualization_factor: f64) -> Self {
        Self {
            window,
            risk_free_rate,
            annualization_factor,
            prev_close: f64::NAN,
            returns_buffer: VecDeque::with_capacity(window),
            update_count: 0,
        }
    }

    pub fn update(&mut self, close: f64) -> f64 {
        self.update_count += 1;

        if self.update_count == 1 {
            self.prev_close = close;
            return f64::NAN;
        }

        if self.prev_close > 0.0 && close > 0.0 {
            let log_return = (close / self.prev_close).ln();
            self.returns_buffer.push_back(log_return);

            if self.returns_buffer.len() > self.window {
                self.returns_buffer.pop_front();
            }
        }

        self.prev_close = close;

        if self.returns_buffer.len() < self.window {
            f64::NAN
        } else {
            let returns_vec: Vec<f64> = self.returns_buffer.iter().copied().collect();

            let avg_return = returns_vec.iter().sum::<f64>() / returns_vec.len() as f64;
            let annualized_return = avg_return * self.annualization_factor;

            let variance = returns_vec.iter()
                .map(|r| (r - avg_return).powi(2))
                .sum::<f64>() / (returns_vec.len() - 1) as f64;
            let volatility = variance.sqrt() * self.annualization_factor.sqrt();

            if volatility > 0.0 {
                (annualized_return - self.risk_free_rate) / volatility
            } else {
                0.0
            }
        }
    }

    pub fn reset(&mut self) {
        self.prev_close = f64::NAN;
        self.returns_buffer.clear();
        self.update_count = 0;
    }
}

// ============================================================================
// Compound Log Return
// ============================================================================
#[pyclass]
pub struct CompoundLogReturnStreaming {
    cumulative_log_return: f64,
    prev_close: f64,
    update_count: usize,
}

#[pymethods]
impl CompoundLogReturnStreaming {
    #[new]
    pub fn new() -> Self {
        Self {
            cumulative_log_return: 0.0,
            prev_close: f64::NAN,
            update_count: 0,
        }
    }

    pub fn update(&mut self, close: f64) -> f64 {
        self.update_count += 1;

        if self.update_count == 1 {
            self.prev_close = close;
            return 0.0;
        }

        if self.prev_close > 0.0 && close > 0.0 {
            let log_return = (close / self.prev_close).ln();
            self.cumulative_log_return += log_return;
        }

        self.prev_close = close;
        (self.cumulative_log_return.exp() - 1.0) * 100.0
    }

    pub fn reset(&mut self) {
        self.cumulative_log_return = 0.0;
        self.prev_close = f64::NAN;
        self.update_count = 0;
    }
}

// ============================================================================
// Rolling Z-Score
// ============================================================================
#[pyclass]
pub struct RollingZScoreStreaming {
    window: usize,
    buffer: VecDeque<f64>,
}

#[pymethods]
impl RollingZScoreStreaming {
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
            let std = variance.sqrt();

            if std != 0.0 {
                (value - mean) / std
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
// Linear Regression Slope
// ============================================================================
#[pyclass]
pub struct LinearRegressionSlopeStreaming {
    window: usize,
    sum_x: f64,
    #[allow(dead_code)]
    sum_x2: f64,
    denom: f64,
    buffer: VecDeque<f64>,
}

#[pymethods]
impl LinearRegressionSlopeStreaming {
    #[new]
    pub fn new(window: usize) -> Self {
        let w = window as f64;
        let sum_x = w * (w - 1.0) / 2.0;
        let sum_x2 = w * (w - 1.0) * (2.0 * w - 1.0) / 6.0;
        let denom = w * sum_x2 - sum_x * sum_x;

        Self {
            window,
            sum_x,
            sum_x2,
            denom,
            buffer: VecDeque::with_capacity(window),
        }
    }

    pub fn update(&mut self, value: f64) -> f64 {
        self.buffer.push_back(value);

        if self.buffer.len() > self.window {
            self.buffer.pop_front();
        }

        if self.buffer.len() < self.window || self.denom == 0.0 {
            f64::NAN
        } else {
            let mut sum_y = 0.0;
            let mut sum_xy = 0.0;

            for (j, &y) in self.buffer.iter().enumerate() {
                sum_y += y;
                sum_xy += j as f64 * y;
            }

            (self.window as f64 * sum_xy - self.sum_x * sum_y) / self.denom
        }
    }

    pub fn reset(&mut self) {
        self.buffer.clear();
    }
}

// ============================================================================
// Rolling Percentile
// ============================================================================
#[pyclass]
pub struct RollingPercentileStreaming {
    window: usize,
    buffer: VecDeque<f64>,
}

#[pymethods]
impl RollingPercentileStreaming {
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
            let count = self.buffer.iter().filter(|&&x| x <= value).count();
            count as f64 / self.window as f64
        }
    }

    pub fn reset(&mut self) {
        self.buffer.clear();
    }
}

// ============================================================================
// Calmar Ratio
// ============================================================================
#[pyclass]
pub struct CalmarRatioStreaming {
    window: usize,
    close_buffer: VecDeque<f64>,
}

#[pymethods]
impl CalmarRatioStreaming {
    #[new]
    pub fn new(window: usize) -> Self {
        Self {
            window,
            close_buffer: VecDeque::with_capacity(window),
        }
    }

    pub fn update(&mut self, close: f64) -> f64 {
        self.close_buffer.push_back(close);

        if self.close_buffer.len() > self.window {
            self.close_buffer.pop_front();
        }

        if self.close_buffer.len() < self.window {
            f64::NAN
        } else {
            let close_vec: Vec<f64> = self.close_buffer.iter().copied().collect();

            // Calculate annualized return
            let total_return = (close_vec[close_vec.len() - 1] / close_vec[0]) - 1.0;
            let annual_return = total_return * (252.0 / close_vec.len() as f64);

            // Calculate maximum drawdown
            let mut running_max = close_vec[0];
            let mut max_drawdown: f64 = 0.0;

            for &price in close_vec.iter().skip(1) {
                running_max = running_max.max(price);
                let drawdown = (price - running_max) / running_max;
                max_drawdown = max_drawdown.min(drawdown);
            }

            let max_drawdown: f64 = max_drawdown.abs();

            if max_drawdown > 0.0 {
                annual_return / max_drawdown
            } else {
                0.0
            }
        }
    }

    pub fn reset(&mut self) {
        self.close_buffer.clear();
    }
}
