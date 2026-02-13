/// Volatility indicators: ATR, Bollinger Bands, Keltner Channel, Donchian Channel, Ulcer Index

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use crate::helpers::{sma_kernel, sma_kernel_min0, wilders_ema_kernel, true_range, rolling_std, rolling_min, rolling_max};

/// ATR - Average True Range (Wilder's method)
///
/// # Arguments
/// * `high` - High price series
/// * `low` - Low price series
/// * `close` - Close price series
/// * `n` - ATR period (default: 14)
///
/// # Returns
/// Numpy array with ATR values
#[pyfunction]
#[pyo3(name = "average_true_range_numba", signature = (high, low, close, n=14))]
pub fn atr<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    n: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;

    let tr = true_range(high_slice, low_slice, close_slice);
    let atr_values = wilders_ema_kernel(&tr, n);

    Ok(PyArray1::from_vec(py, atr_values))
}

/// Bollinger Bands
///
/// # Arguments
/// * `close` - Close price series
/// * `n` - Period for moving average and std (default: 20)
/// * `k` - Number of standard deviations (default: 2.0)
///
/// # Returns
/// Tuple of (upper_band, middle_band, lower_band) as numpy arrays
#[pyfunction]
#[pyo3(name = "bollinger_bands_numba", signature = (close, n=20, k=2.0))]
pub fn bollinger_bands<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    n: usize,
    k: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let close_slice = close.as_slice()?;
    let len = close_slice.len();

    let middle = sma_kernel(close_slice, n);
    let std = rolling_std(close_slice, n);

    let mut upper = vec![f64::NAN; len];
    let mut lower = vec![f64::NAN; len];

    for i in 0..len {
        if !middle[i].is_nan() && !std[i].is_nan() {
            upper[i] = middle[i] + k * std[i];
            lower[i] = middle[i] - k * std[i];
        }
    }

    Ok((
        PyArray1::from_vec(py, upper),
        PyArray1::from_vec(py, middle),
        PyArray1::from_vec(py, lower),
    ))
}

/// Keltner Channel (matching ta library original_version=True)
///
/// # Arguments
/// * `high` - High price series
/// * `low` - Low price series
/// * `close` - Close price series
/// * `n_ema` - Period for typical price moving average (default: 20)
/// * `n_atr` - Unused (kept for API compatibility)
/// * `k` - Unused (kept for API compatibility)
///
/// # Returns
/// Tuple of (upper_band, middle_band, lower_band) as numpy arrays
///
/// Uses ta library original_version formulas:
/// - Middle: SMA(typical_price, n_ema) with min_periods=window
/// - High band: SMA((4*H - 2*L + C)/3, n_ema) with min_periods=0
/// - Low band: SMA((-2*H + 4*L + C)/3, n_ema) with min_periods=0
#[pyfunction]
#[pyo3(name = "keltner_channel_numba", signature = (high, low, close, n_ema=20, n_atr=10, k=2.0))]
pub fn keltner_channel<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    n_ema: usize,
    #[allow(unused_variables)]
    n_atr: usize,
    #[allow(unused_variables)]
    k: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    let len = high_slice.len();

    // Middle: SMA(typical_price, n_ema) with standard min_periods=window
    let mut typical_price = vec![0.0; len];
    for i in 0..len {
        typical_price[i] = (high_slice[i] + low_slice[i] + close_slice[i]) / 3.0;
    }
    let middle = sma_kernel(&typical_price, n_ema);

    // High band: SMA((4*H - 2*L + C)/3, n_ema) with min_periods=0
    let mut high_tp = vec![0.0; len];
    for i in 0..len {
        high_tp[i] = (4.0 * high_slice[i] - 2.0 * low_slice[i] + close_slice[i]) / 3.0;
    }
    let upper = sma_kernel_min0(&high_tp, n_ema);

    // Low band: SMA((-2*H + 4*L + C)/3, n_ema) with min_periods=0
    let mut low_tp = vec![0.0; len];
    for i in 0..len {
        low_tp[i] = (-2.0 * high_slice[i] + 4.0 * low_slice[i] + close_slice[i]) / 3.0;
    }
    let lower = sma_kernel_min0(&low_tp, n_ema);

    Ok((
        PyArray1::from_vec(py, upper),
        PyArray1::from_vec(py, middle),
        PyArray1::from_vec(py, lower),
    ))
}

/// Donchian Channel
///
/// # Arguments
/// * `high` - High price series
/// * `low` - Low price series
/// * `n` - Period for channel calculation (default: 20)
///
/// # Returns
/// Tuple of (upper_band, middle_band, lower_band) as numpy arrays
#[pyfunction]
#[pyo3(name = "donchian_channel_numba", signature = (high, low, n=20))]
pub fn donchian_channel<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    n: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let len = high_slice.len();

    let upper = rolling_max(high_slice, n);
    let lower = rolling_min(low_slice, n);

    let mut middle = vec![f64::NAN; len];
    for i in 0..len {
        if !upper[i].is_nan() && !lower[i].is_nan() {
            middle[i] = (upper[i] + lower[i]) / 2.0;
        }
    }

    Ok((
        PyArray1::from_vec(py, upper),
        PyArray1::from_vec(py, middle),
        PyArray1::from_vec(py, lower),
    ))
}

/// Ulcer Index
///
/// # Arguments
/// * `data` - Price series (typically close)
/// * `n` - Period for Ulcer Index calculation (default: 14)
///
/// # Returns
/// Numpy array with Ulcer Index values
#[pyfunction]
#[pyo3(name = "ulcer_index_numba", signature = (close, n=14))]
pub fn ulcer_index<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    n: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let close_slice = close.as_slice()?;
    let len = close_slice.len();

    let mut pct_drawdown_sq = vec![0.0; len];

    for i in 1..len {
        let start_idx = if i >= n { i - n + 1 } else { 0 };
        let max_close = close_slice[start_idx..=i]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        let pct_dd = ((close_slice[i] - max_close) / max_close) * 100.0;
        pct_drawdown_sq[i] = pct_dd * pct_dd;
    }

    let mut ui = vec![f64::NAN; len];
    for i in (n - 1)..len {
        let mean_sq: f64 = pct_drawdown_sq[(i + 1 - n)..=i].iter().sum::<f64>() / n as f64;
        ui[i] = mean_sq.sqrt();
    }

    Ok(PyArray1::from_vec(py, ui))
}
