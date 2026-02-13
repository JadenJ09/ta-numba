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
    prev_close: f64,
    returns_buffer: VecDeque<f64>,
    update_count: usize,
}

#[pymethods]
impl SharpeRatioStreaming {
    #[new]
    pub fn new(window: usize, risk_free_rate: f64) -> Self {
        Self {
            window,
            risk_free_rate,
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
            let annualized_return = avg_return * 252.0;

            let variance = returns_vec.iter()
                .map(|r| (r - avg_return).powi(2))
                .sum::<f64>() / (returns_vec.len() - 1) as f64;
            let volatility = variance.sqrt() * (252.0_f64).sqrt();

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
