/// Trend indicators: SMA, EMA, MACD, ADX, CCI, DPO, Vortex, Parabolic SAR

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use crate::helpers::{sma_kernel, ema_kernel, wilders_ema_kernel, true_range, rolling_sum};

/// Simple Moving Average
///
/// # Arguments
/// * `data` - Input price series
/// * `n` - Period for moving average
///
/// # Returns
/// Numpy array with SMA values (NaN for first n-1 elements)
#[pyfunction]
#[pyo3(name = "sma_numba", signature = (data, n=20))]
pub fn sma<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    n: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data_slice = data.as_slice()?;
    let result = sma_kernel(data_slice, n);
    Ok(PyArray1::from_vec(py, result))
}

/// Exponential Moving Average
///
/// # Arguments
/// * `data` - Input price series
/// * `n` - Period for EMA
/// * `adjusted` - Use pandas-style adjusted EMA (default: true)
///
/// # Returns
/// Numpy array with EMA values
#[pyfunction]
#[pyo3(name = "ema_numba", signature = (data, n=20, adjusted=true))]
pub fn ema<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    n: usize,
    adjusted: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data_slice = data.as_slice()?;
    let alpha = 2.0 / (n as f64 + 1.0);
    let result = ema_kernel(data_slice, alpha, adjusted);
    Ok(PyArray1::from_vec(py, result))
}

/// Weighted Moving Average
///
/// # Arguments
/// * `data` - Input price series
/// * `n` - Period for WMA
///
/// # Returns
/// Numpy array with WMA values
#[pyfunction]
#[pyo3(name = "weighted_moving_average", signature = (data, n=14))]
pub fn wma<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    n: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data_slice = data.as_slice()?;
    let len = data_slice.len();
    let mut result = vec![f64::NAN; len];

    if len < n {
        return Ok(PyArray1::from_vec(py, result));
    }

    let weights: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let weight_sum: f64 = weights.iter().sum();

    for i in (n - 1)..len {
        let window_start = i + 1 - n;
        let weighted_sum: f64 = data_slice[window_start..=i]
            .iter()
            .zip(&weights)
            .map(|(price, weight)| price * weight)
            .sum();
        result[i] = weighted_sum / weight_sum;
    }

    Ok(PyArray1::from_vec(py, result))
}

/// MACD - Moving Average Convergence Divergence
///
/// # Arguments
/// * `close` - Close price series
/// * `n_fast` - Fast EMA period (default: 12)
/// * `n_slow` - Slow EMA period (default: 26)
/// * `n_signal` - Signal line EMA period (default: 9)
/// * `adjusted` - Use adjusted EMA for MACD line (default: false)
///
/// # Returns
/// Tuple of (macd_line, signal_line, histogram) as numpy arrays
#[pyfunction]
#[pyo3(name = "macd_numba", signature = (close, n_fast=12, n_slow=26, n_signal=9, adjusted=false))]
pub fn macd<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    n_fast: usize,
    n_slow: usize,
    n_signal: usize,
    adjusted: bool,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let close_slice = close.as_slice()?;
    let n_param = close_slice.len();

    let alpha_fast = 2.0 / (n_fast as f64 + 1.0);
    let alpha_slow = 2.0 / (n_slow as f64 + 1.0);

    let ema_fast = ema_kernel(close_slice, alpha_fast, adjusted);
    let ema_slow = ema_kernel(close_slice, alpha_slow, adjusted);

    let mut macd_line = vec![f64::NAN; n_param];
    for i in 0..n_param {
        if !ema_fast[i].is_nan() && !ema_slow[i].is_nan() {
            macd_line[i] = ema_fast[i] - ema_slow[i];
        }
    }

    let alpha_signal = 2.0 / (n_signal as f64 + 1.0);
    let signal_line = ema_kernel(&macd_line, alpha_signal, true);

    let mut histogram = vec![f64::NAN; n_param];
    for i in 0..n_param {
        if !macd_line[i].is_nan() && !signal_line[i].is_nan() {
            histogram[i] = macd_line[i] - signal_line[i];
        }
    }

    Ok((
        PyArray1::from_vec(py, macd_line),
        PyArray1::from_vec(py, signal_line),
        PyArray1::from_vec(py, histogram),
    ))
}

/// ADX - Average Directional Index
///
/// # Arguments
/// * `high` - High price series
/// * `low` - Low price series
/// * `close` - Close price series
/// * `n` - ADX period (default: 14)
///
/// # Returns
/// Tuple of (ADX, +DI, -DI) as numpy arrays
#[pyfunction]
#[pyo3(name = "adx_numba", signature = (high, low, close, n=14))]
pub fn adx<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    n: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    let len = high_slice.len();

    let mut plus_dm = vec![0.0; len];
    let mut minus_dm = vec![0.0; len];

    for i in 1..len {
        let high_diff = high_slice[i] - high_slice[i - 1];
        let low_diff = low_slice[i - 1] - low_slice[i];

        if high_diff > low_diff && high_diff > 0.0 {
            plus_dm[i] = high_diff;
        }
        if low_diff > high_diff && low_diff > 0.0 {
            minus_dm[i] = low_diff;
        }
    }

    let tr = true_range(high_slice, low_slice, close_slice);
    let atr_values = wilders_ema_kernel(&tr, n);

    let mut plus_di = vec![f64::NAN; len];
    let mut minus_di = vec![f64::NAN; len];

    let smoothed_plus_dm = wilders_ema_kernel(&plus_dm, n);
    let smoothed_minus_dm = wilders_ema_kernel(&minus_dm, n);

    for i in 0..len {
        if !atr_values[i].is_nan() && atr_values[i] != 0.0 {
            plus_di[i] = (smoothed_plus_dm[i] / atr_values[i]) * 100.0;
            minus_di[i] = (smoothed_minus_dm[i] / atr_values[i]) * 100.0;
        }
    }

    let mut dx = vec![f64::NAN; len];
    for i in 0..len {
        if !plus_di[i].is_nan() && !minus_di[i].is_nan() {
            let di_sum = plus_di[i] + minus_di[i];
            if di_sum != 0.0 {
                dx[i] = ((plus_di[i] - minus_di[i]).abs() / di_sum) * 100.0;
            }
        }
    }

    let adx_values = wilders_ema_kernel(&dx, n);

    Ok((
        PyArray1::from_vec(py, adx_values),
        PyArray1::from_vec(py, plus_di),
        PyArray1::from_vec(py, minus_di),
    ))
}

/// CCI - Commodity Channel Index
///
/// # Arguments
/// * `high` - High price series
/// * `low` - Low price series
/// * `close` - Close price series
/// * `n` - CCI period (default: 20)
/// * `constant` - Scaling constant (default: 0.015)
///
/// # Returns
/// Numpy array with CCI values
#[pyfunction]
#[pyo3(name = "cci_numba", signature = (high, low, close, n=20, c=0.015))]
pub fn cci<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    n: usize,
    c: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    let len = high_slice.len();

    let mut typical_price = vec![0.0; len];
    for i in 0..len {
        typical_price[i] = (high_slice[i] + low_slice[i] + close_slice[i]) / 3.0;
    }

    let sma_tp = sma_kernel(&typical_price, n);

    let mut cci_values = vec![f64::NAN; len];
    for i in (n - 1)..len {
        let window_start = i + 1 - n;
        let tp_slice = &typical_price[window_start..=i];

        let mean_dev: f64 = tp_slice.iter()
            .map(|&tp| (tp - sma_tp[i]).abs())
            .sum::<f64>() / n as f64;

        if mean_dev != 0.0 {
            cci_values[i] = (typical_price[i] - sma_tp[i]) / (c * mean_dev);
        }
    }

    Ok(PyArray1::from_vec(py, cci_values))
}

/// DPO - Detrended Price Oscillator
///
/// # Arguments
/// * `close` - Close price series
/// * `n` - DPO period (default: 20)
///
/// # Returns
/// Numpy array with DPO values
#[pyfunction]
#[pyo3(name = "dpo_numba", signature = (close, n=20))]
pub fn dpo<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    n: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let close_slice = close.as_slice()?;
    let len = close_slice.len();

    let sma_values = sma_kernel(close_slice, n);
    let shift = (n / 2) + 1;

    let mut dpo_values = vec![f64::NAN; len];
    for i in 0..len {
        if i >= shift - 1 {
            let sma_idx = i + 1 - shift;
            if sma_idx < len && !sma_values[sma_idx].is_nan() {
                dpo_values[i] = close_slice[i] - sma_values[sma_idx];
            }
        }
    }

    Ok(PyArray1::from_vec(py, dpo_values))
}

/// Vortex Indicator
///
/// # Arguments
/// * `high` - High price series
/// * `low` - Low price series
/// * `close` - Close price series
/// * `n` - VI period (default: 14)
///
/// # Returns
/// Tuple of (VI+, VI-) as numpy arrays
#[pyfunction]
#[pyo3(name = "vortex_indicator_numba", signature = (high, low, close, n=14))]
pub fn vortex_indicator<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    n: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    let len = high_slice.len();

    let mut vm_plus = vec![0.0; len];
    let mut vm_minus = vec![0.0; len];

    for i in 1..len {
        vm_plus[i] = (high_slice[i] - low_slice[i - 1]).abs();
        vm_minus[i] = (low_slice[i] - high_slice[i - 1]).abs();
    }

    let tr = true_range(high_slice, low_slice, close_slice);

    let sum_vm_plus = rolling_sum(&vm_plus, n);
    let sum_vm_minus = rolling_sum(&vm_minus, n);
    let sum_tr = rolling_sum(&tr, n);

    let mut vi_plus = vec![f64::NAN; len];
    let mut vi_minus = vec![f64::NAN; len];

    for i in (n - 1)..len {
        if sum_tr[i] != 0.0 && !sum_tr[i].is_nan() {
            vi_plus[i] = sum_vm_plus[i] / sum_tr[i];
            vi_minus[i] = sum_vm_minus[i] / sum_tr[i];
        }
    }

    Ok((
        PyArray1::from_vec(py, vi_plus),
        PyArray1::from_vec(py, vi_minus),
    ))
}

/// Parabolic SAR
///
/// # Arguments
/// * `high` - High price series
/// * `low` - Low price series
/// * `close` - Close price series
/// * `af_start` - Initial acceleration factor (default: 0.02)
/// * `af_increment` - AF increment per extreme point (default: 0.02)
/// * `af_max` - Maximum AF (default: 0.2)
///
/// # Returns
/// Numpy array with SAR values
#[pyfunction]
#[pyo3(name = "parabolic_sar_numba", signature = (high, low, close, af_start=0.02, af_inc=0.02, af_max=0.2))]
pub fn parabolic_sar<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    af_start: f64,
    af_inc: f64,
    af_max: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    let len = high_slice.len();

    if len < 2 {
        return Ok(PyArray1::from_vec(py, vec![f64::NAN; len]));
    }

    let mut sar = vec![f64::NAN; len];
    let mut is_long = close_slice[1] > close_slice[0];
    let mut af = af_start;
    let mut extreme_point = if is_long { high_slice[1] } else { low_slice[1] };

    sar[0] = if is_long { low_slice[0] } else { high_slice[0] };
    sar[1] = sar[0];

    for i in 2..len {
        let prev_sar = sar[i - 1];
        let mut new_sar = prev_sar + af * (extreme_point - prev_sar);

        let mut trend_changed = false;

        if is_long {
            new_sar = new_sar.min(low_slice[i - 1]).min(low_slice[i - 2]);

            if low_slice[i] < new_sar {
                trend_changed = true;
                is_long = false;
                new_sar = extreme_point;
                extreme_point = low_slice[i];
                af = af_start;
            } else {
                if high_slice[i] > extreme_point {
                    extreme_point = high_slice[i];
                    af = (af + af_inc).min(af_max);
                }
            }
        } else {
            new_sar = new_sar.max(high_slice[i - 1]).max(high_slice[i - 2]);

            if high_slice[i] > new_sar {
                trend_changed = true;
                is_long = true;
                new_sar = extreme_point;
                extreme_point = high_slice[i];
                af = af_start;
            } else {
                if low_slice[i] < extreme_point {
                    extreme_point = low_slice[i];
                    af = (af + af_inc).min(af_max);
                }
            }
        }

        sar[i] = new_sar;
    }

    Ok(PyArray1::from_vec(py, sar))
}

/// TRIX - Triple Exponential Average
///
/// # Arguments
/// * `close` - Close price series
/// * `n` - TRIX period (default: 15)
///
/// # Returns
/// Numpy array with TRIX values (percentage change)
#[pyfunction]
#[pyo3(name = "trix_numba", signature = (close, n=14))]
pub fn trix<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    n: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let close_slice = close.as_slice()?;
    let len = close_slice.len();

    let alpha = 2.0 / (n as f64 + 1.0);

    let ema1 = ema_kernel(close_slice, alpha, true);
    let ema2 = ema_kernel(&ema1, alpha, true);
    let ema3 = ema_kernel(&ema2, alpha, true);

    let mut trix_values = vec![f64::NAN; len];
    for i in 1..len {
        if !ema3[i].is_nan() && !ema3[i - 1].is_nan() && ema3[i - 1] != 0.0 {
            trix_values[i] = ((ema3[i] - ema3[i - 1]) / ema3[i - 1]) * 100.0;
        }
    }

    Ok(PyArray1::from_vec(py, trix_values))
}

/// Mass Index
///
/// # Arguments
/// * `high` - High price series
/// * `low` - Low price series
/// * `n_ema` - EMA period for range (default: 9)
/// * `n_sum` - Summation period (default: 25)
///
/// # Returns
/// Numpy array with Mass Index values
#[pyfunction]
#[pyo3(name = "mass_index_numba", signature = (high, low, n_ema=9, n_sum=25))]
pub fn mass_index<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    n_ema: usize,
    n_sum: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let len = high_slice.len();

    let mut range = vec![0.0; len];
    for i in 0..len {
        range[i] = high_slice[i] - low_slice[i];
    }

    let alpha = 2.0 / (n_ema as f64 + 1.0);
    let ema1 = ema_kernel(&range, alpha, true);
    let ema2 = ema_kernel(&ema1, alpha, true);

    let mut ratio = vec![f64::NAN; len];
    for i in 0..len {
        if !ema2[i].is_nan() && ema2[i] != 0.0 {
            ratio[i] = ema1[i] / ema2[i];
        }
    }

    let mut mi = vec![f64::NAN; len];
    for i in (n_sum - 1)..len {
        let sum: f64 = ratio[(i + 1 - n_sum)..=i].iter()
            .filter(|x| !x.is_nan())
            .sum();
        mi[i] = sum;
    }

    Ok(PyArray1::from_vec(py, mi))
}

/// KST - Know Sure Thing
///
/// # Arguments
/// * `close` - Close price series
/// * `r1, r2, r3, r4` - ROC periods (defaults: 10, 15, 20, 30)
/// * `s1, s2, s3, s4` - SMA smoothing periods (defaults: 10, 10, 10, 15)
/// * `n_sig` - Signal line period (default: 9)
///
/// # Returns
/// Tuple of (KST, signal) as numpy arrays
#[pyfunction]
#[pyo3(name = "kst_numba", signature = (close, r1=10, r2=15, r3=20, r4=30, s1=10, s2=10, s3=10, s4=15, n_sig=9))]
pub fn kst<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    r1: usize,
    r2: usize,
    r3: usize,
    r4: usize,
    s1: usize,
    s2: usize,
    s3: usize,
    s4: usize,
    n_sig: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let close_slice = close.as_slice()?;
    let len = close_slice.len();

    let calc_roc = |window: usize| -> Vec<f64> {
        let mut roc = vec![f64::NAN; len];
        for i in window..len {
            if close_slice[i - window] != 0.0 {
                roc[i] = (close_slice[i] - close_slice[i - window]) / close_slice[i - window] * 100.0;
            }
        }
        roc
    };

    let roc1 = calc_roc(r1);
    let roc2 = calc_roc(r2);
    let roc3 = calc_roc(r3);
    let roc4 = calc_roc(r4);

    let rcma1 = sma_kernel(&roc1, s1);
    let rcma2 = sma_kernel(&roc2, s2);
    let rcma3 = sma_kernel(&roc3, s3);
    let rcma4 = sma_kernel(&roc4, s4);

    let mut kst_values = vec![f64::NAN; len];
    for i in 0..len {
        if !rcma1[i].is_nan() && !rcma2[i].is_nan() && !rcma3[i].is_nan() && !rcma4[i].is_nan() {
            kst_values[i] = rcma1[i] + 2.0 * rcma2[i] + 3.0 * rcma3[i] + 4.0 * rcma4[i];
        }
    }

    let signal = sma_kernel(&kst_values, n_sig);

    Ok((
        PyArray1::from_vec(py, kst_values),
        PyArray1::from_vec(py, signal),
    ))
}

/// Ichimoku Cloud
///
/// # Arguments
/// * `high` - High price series
/// * `low` - Low price series
/// * `n1` - Tenkan period (default: 9)
/// * `n2` - Kijun period (default: 26)
/// * `n3` - Senkou B period (default: 52)
///
/// # Returns
/// Tuple of (tenkan, kijun, senkou_a, senkou_b, chikou) as numpy arrays
#[pyfunction]
#[pyo3(name = "ichimoku_numba", signature = (high, low, close, n1=9, n2=26, n3=52))]
pub fn ichimoku<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    n1: usize,
    n2: usize,
    n3: usize,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let high_slice = high.as_slice()?;
    let close_slice = close.as_slice()?;
    let low_slice = low.as_slice()?;
    let len = high_slice.len();

    let calc_midpoint = |window: usize| -> Vec<f64> {
        let mut result = vec![f64::NAN; len];
        for i in (window - 1)..len {
            let window_high = high_slice[(i + 1 - window)..=i].iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let window_low = low_slice[(i + 1 - window)..=i].iter().copied().fold(f64::INFINITY, f64::min);
            result[i] = (window_high + window_low) / 2.0;
        }
        result
    };

    let tenkan = calc_midpoint(n1);
    let kijun = calc_midpoint(n2);

    let mut senkou_a = vec![f64::NAN; len];
    for i in 0..len {
        if !tenkan[i].is_nan() && !kijun[i].is_nan() {
            senkou_a[i] = (tenkan[i] + kijun[i]) / 2.0;
        }
    }

    let senkou_b = calc_midpoint(n3);

    let mut chikou = vec![f64::NAN; len];
    for i in n2..len {
        chikou[i - n2] = close_slice[i];
    }

    Ok((
        PyArray1::from_vec(py, tenkan),
        PyArray1::from_vec(py, kijun),
        PyArray1::from_vec(py, senkou_a),
        PyArray1::from_vec(py, senkou_b),
        PyArray1::from_vec(py, chikou),
    ))
}

/// Schaff Trend Cycle
///
/// # Arguments
/// * `close` - Close price series
/// * `n_fast` - Fast MACD period (default: 23)
/// * `n_slow` - Slow MACD period (default: 50)
/// * `n_stoch` - Stochastic period (default: 10)
/// * `n_smooth` - Smoothing period (default: 3)
///
/// # Returns
/// Numpy array with STC values
#[pyfunction]
#[pyo3(name = "schaff_trend_cycle_numba", signature = (close, n_fast=23, n_slow=50, n_stoch=10, n_smooth=3))]
pub fn schaff_trend_cycle<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    n_fast: usize,
    n_slow: usize,
    n_stoch: usize,
    n_smooth: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let close_slice = close.as_slice()?;
    let len = close_slice.len();

    let alpha_fast = 2.0 / (n_fast as f64 + 1.0);
    let alpha_slow = 2.0 / (n_slow as f64 + 1.0);

    let ema_fast = ema_kernel(close_slice, alpha_fast, false);
    let ema_slow = ema_kernel(close_slice, alpha_slow, false);

    let mut macd_line = vec![f64::NAN; len];
    for i in 0..len {
        if !ema_fast[i].is_nan() && !ema_slow[i].is_nan() {
            macd_line[i] = ema_fast[i] - ema_slow[i];
        }
    }

    let mut pf = vec![f64::NAN; len];
    for i in (n_stoch - 1)..len {
        let window_start = i + 1 - n_stoch;
        let macd_slice = &macd_line[window_start..=i];
        let lowest = macd_slice.iter().filter(|x| !x.is_nan()).copied().fold(f64::INFINITY, f64::min);
        let highest = macd_slice.iter().filter(|x| !x.is_nan()).copied().fold(f64::NEG_INFINITY, f64::max);

        if highest > lowest {
            pf[i] = 100.0 * (macd_line[i] - lowest) / (highest - lowest);
        } else {
            pf[i] = 0.0;
        }
    }

    let alpha_smooth = 2.0 / (n_smooth as f64 + 1.0);
    let pf_smooth = ema_kernel(&pf, alpha_smooth, false);

    let mut pff = vec![f64::NAN; len];
    for i in (n_stoch - 1)..len {
        let window_start = i + 1 - n_stoch;
        let pf_slice = &pf_smooth[window_start..=i];
        let lowest = pf_slice.iter().filter(|x| !x.is_nan()).copied().fold(f64::INFINITY, f64::min);
        let highest = pf_slice.iter().filter(|x| !x.is_nan()).copied().fold(f64::NEG_INFINITY, f64::max);

        if highest > lowest {
            pff[i] = 100.0 * (pf_smooth[i] - lowest) / (highest - lowest);
        } else {
            pff[i] = 0.0;
        }
    }

    let stc = ema_kernel(&pff, alpha_smooth, false);

    Ok(PyArray1::from_vec(py, stc))
}

/// Aroon Indicator
///
/// # Arguments
/// * `high` - High price series
/// * `low` - Low price series
/// * `n` - Aroon period (default: 25)
///
/// # Returns
/// Tuple of (aroon_up, aroon_down) as numpy arrays
#[pyfunction]
#[pyo3(name = "aroon_numba", signature = (high, low, n=25))]
pub fn aroon<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    n: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let len = high_slice.len();

    let mut aroon_up = vec![f64::NAN; len];
    let mut aroon_down = vec![f64::NAN; len];

    for i in (n - 1)..len {
        let window_start = i + 1 - n;
        let high_window = &high_slice[window_start..=i];
        let low_window = &low_slice[window_start..=i];

        let (max_idx, _) = high_window.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        let (min_idx, _) = low_window.iter().enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        aroon_up[i] = ((n - 1 - max_idx) as f64 / (n - 1) as f64) * 100.0;
        aroon_down[i] = ((n - 1 - min_idx) as f64 / (n - 1) as f64) * 100.0;
    }

    Ok((
        PyArray1::from_vec(py, aroon_up),
        PyArray1::from_vec(py, aroon_down),
    ))
}
