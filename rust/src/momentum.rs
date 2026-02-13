/// Momentum indicators: RSI, Stochastic, Williams %R, PPO, Ultimate Oscillator

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use crate::helpers::{sma_kernel, sma_kernel_nan_aware, rolling_min, rolling_max, ema_kernel, true_range, rolling_sum};

/// RSI - Relative Strength Index (Wilder's method)
///
/// # Arguments
/// * `close` - Close price series
/// * `n` - RSI period (default: 14)
///
/// # Returns
/// Numpy array with RSI values (0-100)
#[pyfunction]
#[pyo3(name = "relative_strength_index_numba", signature = (close, n=14))]
pub fn rsi<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    n: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let close_slice = close.as_slice()?;
    let len = close_slice.len();
    let mut rsi_values = vec![f64::NAN; len];

    if len < 2 {
        return Ok(PyArray1::from_vec(py, rsi_values));
    }

    let mut gains = vec![0.0; len];
    let mut losses = vec![0.0; len];

    for i in 1..len {
        let delta = close_slice[i] - close_slice[i - 1];
        if delta > 0.0 {
            gains[i] = delta;
            losses[i] = 0.0;
        } else {
            gains[i] = 0.0;
            losses[i] = -delta;
        }
    }

    let alpha = 1.0 / n as f64;
    let mut avg_gain = vec![f64::NAN; len];
    let mut avg_loss = vec![f64::NAN; len];

    if len > n {
        let mut sum_gain = 0.0;
        let mut sum_loss = 0.0;
        for i in 1..=n {
            sum_gain += gains[i];
            sum_loss += losses[i];
        }
        avg_gain[n] = sum_gain / n as f64;
        avg_loss[n] = sum_loss / n as f64;

        for i in (n + 1)..len {
            avg_gain[i] = alpha * gains[i] + (1.0 - alpha) * avg_gain[i - 1];
            avg_loss[i] = alpha * losses[i] + (1.0 - alpha) * avg_loss[i - 1];
        }
    }

    for i in n..len {
        if avg_loss[i] == 0.0 {
            rsi_values[i] = 100.0;
        } else {
            let rs = avg_gain[i] / avg_loss[i];
            rsi_values[i] = 100.0 - (100.0 / (1.0 + rs));
        }
    }

    Ok(PyArray1::from_vec(py, rsi_values))
}

/// Stochastic Oscillator
///
/// # Arguments
/// * `high` - High price series
/// * `low` - Low price series
/// * `close` - Close price series
/// * `k_period` - %K period (default: 14)
/// * `d_period` - %D smoothing period (default: 3)
///
/// # Returns
/// Tuple of (%K, %D) as numpy arrays
#[pyfunction]
#[pyo3(name = "stochastic_oscillator_numba", signature = (high, low, close, n=14, d=3))]
pub fn stochastic<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    n: usize,
    d: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    let len = high_slice.len();

    let lowest_low = rolling_min(low_slice, n);
    let highest_high = rolling_max(high_slice, n);

    let mut percent_k = vec![f64::NAN; len];
    for i in (n - 1)..len {
        let range = highest_high[i] - lowest_low[i];
        if range != 0.0 {
            percent_k[i] = 100.0 * (close_slice[i] - lowest_low[i]) / range;
        } else {
            percent_k[i] = 50.0;
        }
    }

    let percent_d = sma_kernel_nan_aware(&percent_k, d);

    Ok((
        PyArray1::from_vec(py, percent_k),
        PyArray1::from_vec(py, percent_d),
    ))
}

/// Williams %R
///
/// # Arguments
/// * `high` - High price series
/// * `low` - Low price series
/// * `close` - Close price series
/// * `n` - Period for calculation (default: 14)
///
/// # Returns
/// Numpy array with Williams %R values (-100 to 0)
#[pyfunction]
#[pyo3(name = "williams_r_numba", signature = (high, low, close, n=14))]
pub fn williams_r<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    n: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    let len = high_slice.len();

    let lowest_low = rolling_min(low_slice, n);
    let highest_high = rolling_max(high_slice, n);

    let mut wr = vec![f64::NAN; len];
    for i in (n - 1)..len {
        let range = highest_high[i] - lowest_low[i];
        if range != 0.0 {
            wr[i] = -100.0 * (highest_high[i] - close_slice[i]) / range;
        } else {
            wr[i] = -100.0;
        }
    }

    Ok(PyArray1::from_vec(py, wr))
}

/// PPO - Percentage Price Oscillator
///
/// # Arguments
/// * `close` - Close price series
/// * `n_fast` - Fast EMA period (default: 12)
/// * `n_slow` - Slow EMA period (default: 26)
/// * `n_signal` - Signal line EMA period (default: 9)
///
/// # Returns
/// Tuple of (ppo_line, signal, histogram) as numpy arrays
#[pyfunction]
#[pyo3(name = "percentage_price_oscillator_numba", signature = (close, n_fast=12, n_slow=26, n_signal=9))]
pub fn ppo<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    n_fast: usize,
    n_slow: usize,
    n_signal: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let close_slice = close.as_slice()?;
    let len = close_slice.len();

    let alpha_fast = 2.0 / (n_fast as f64 + 1.0);
    let alpha_slow = 2.0 / (n_slow as f64 + 1.0);

    let ema_fast = ema_kernel(close_slice, alpha_fast, false);
    let ema_slow = ema_kernel(close_slice, alpha_slow, false);

    let mut ppo_line = vec![f64::NAN; len];
    for i in 0..len {
        if ema_slow[i] != 0.0 && !ema_slow[i].is_nan() {
            ppo_line[i] = (ema_fast[i] - ema_slow[i]) / ema_slow[i] * 100.0;
        }
    }

    let alpha_signal = 2.0 / (n_signal as f64 + 1.0);
    let signal = ema_kernel(&ppo_line, alpha_signal, false);

    let mut histogram = vec![f64::NAN; len];
    for i in 0..len {
        if !ppo_line[i].is_nan() && !signal[i].is_nan() {
            histogram[i] = ppo_line[i] - signal[i];
        }
    }

    Ok((
        PyArray1::from_vec(py, ppo_line),
        PyArray1::from_vec(py, signal),
        PyArray1::from_vec(py, histogram),
    ))
}

/// Ultimate Oscillator
///
/// # Arguments
/// * `high` - High price series
/// * `low` - Low price series
/// * `close` - Close price series
/// * `n1` - Period 1 (default: 7)
/// * `n2` - Period 2 (default: 14)
/// * `n3` - Period 3 (default: 28)
///
/// # Returns
/// Numpy array with Ultimate Oscillator values (0-100)
#[pyfunction]
#[pyo3(name = "ultimate_oscillator_numba", signature = (high, low, close, n1=7, n2=14, n3=28))]
pub fn ultimate_oscillator<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    n1: usize,
    n2: usize,
    n3: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    let len = high_slice.len();

    let mut bp = vec![f64::NAN; len];
    bp[0] = 0.0;

    for i in 1..len {
        let min_low_close = low_slice[i].min(close_slice[i - 1]);
        bp[i] = close_slice[i] - min_low_close;
    }

    let tr = true_range(high_slice, low_slice, close_slice);

    let sum_bp1 = rolling_sum(&bp, n1);
    let sum_tr1 = rolling_sum(&tr, n1);
    let sum_bp2 = rolling_sum(&bp, n2);
    let sum_tr2 = rolling_sum(&tr, n2);
    let sum_bp3 = rolling_sum(&bp, n3);
    let sum_tr3 = rolling_sum(&tr, n3);

    let mut uo = vec![f64::NAN; len];
    for i in (n3 - 1)..len {
        if sum_tr1[i] != 0.0 && sum_tr2[i] != 0.0 && sum_tr3[i] != 0.0 &&
           !sum_tr1[i].is_nan() && !sum_tr2[i].is_nan() && !sum_tr3[i].is_nan() {
            let avg1 = sum_bp1[i] / sum_tr1[i];
            let avg2 = sum_bp2[i] / sum_tr2[i];
            let avg3 = sum_bp3[i] / sum_tr3[i];
            uo[i] = 100.0 * (4.0 * avg1 + 2.0 * avg2 + avg3) / 7.0;
        }
    }

    Ok(PyArray1::from_vec(py, uo))
}

/// Stochastic RSI - Apply stochastic oscillator to RSI values
///
/// # Arguments
/// * `data` - Price data series (typically close prices)
/// * `rsi_window` - RSI period (default: 14)
/// * `stoch_window` - Stochastic period for RSI (default: 14)
/// * `smooth_k` - %K smoothing period (default: 3)
/// * `smooth_d` - %D smoothing period (default: 3)
///
/// # Returns
/// Tuple of (%K, %D) as numpy arrays
#[pyfunction]
#[pyo3(name = "stochastic_rsi_numba", signature = (close, n=14, k=3, d=3))]
pub fn stochastic_rsi<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    n: usize,
    k: usize,
    d: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let close_slice = close.as_slice()?;
    let len = close_slice.len();

    let mut rsi_values = vec![f64::NAN; len];

    if len < 2 {
        return Ok((
            PyArray1::from_vec(py, rsi_values.clone()),
            PyArray1::from_vec(py, rsi_values.clone()),
            PyArray1::from_vec(py, rsi_values),
        ));
    }

    let mut gains = vec![0.0; len];
    let mut losses = vec![0.0; len];

    for i in 1..len {
        let delta = close_slice[i] - close_slice[i - 1];
        if delta > 0.0 {
            gains[i] = delta;
            losses[i] = 0.0;
        } else {
            gains[i] = 0.0;
            losses[i] = -delta;
        }
    }

    let alpha = 1.0 / n as f64;
    let mut avg_gain = vec![f64::NAN; len];
    let mut avg_loss = vec![f64::NAN; len];

    if len > n {
        let mut sum_gain = 0.0;
        let mut sum_loss = 0.0;
        for i in 1..=n {
            sum_gain += gains[i];
            sum_loss += losses[i];
        }
        avg_gain[n] = sum_gain / n as f64;
        avg_loss[n] = sum_loss / n as f64;

        for i in (n + 1)..len {
            avg_gain[i] = alpha * gains[i] + (1.0 - alpha) * avg_gain[i - 1];
            avg_loss[i] = alpha * losses[i] + (1.0 - alpha) * avg_loss[i - 1];
        }
    }

    for i in n..len {
        if avg_loss[i] == 0.0 {
            rsi_values[i] = 100.0;
        } else {
            let rs = avg_gain[i] / avg_loss[i];
            rsi_values[i] = 100.0 - (100.0 / (1.0 + rs));
        }
    }

    let mut stoch_rsi = vec![f64::NAN; len];
    let start_idx = (n - 1) + (n - 1);

    for i in start_idx..len {
        let window_start = i + 1 - n;
        let rsi_window_slice = &rsi_values[window_start..=i];

        let low_rsi = rsi_window_slice.iter()
            .filter(|x| !x.is_nan())
            .copied()
            .fold(f64::INFINITY, f64::min);
        let high_rsi = rsi_window_slice.iter()
            .filter(|x| !x.is_nan())
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        if !rsi_values[i].is_nan() && !low_rsi.is_infinite() && !high_rsi.is_infinite() {
            if high_rsi > low_rsi {
                stoch_rsi[i] = (rsi_values[i] - low_rsi) / (high_rsi - low_rsi);
            } else {
                stoch_rsi[i] = 0.0;
            }
        }
    }

    let stoch_k = sma_kernel_nan_aware(&stoch_rsi, k);
    let stoch_d = sma_kernel_nan_aware(&stoch_k, d);

    Ok((
        PyArray1::from_vec(py, stoch_rsi),
        PyArray1::from_vec(py, stoch_k),
        PyArray1::from_vec(py, stoch_d),
    ))
}

/// TSI - True Strength Index
///
/// # Arguments
/// * `data` - Price data series (typically close prices)
/// * `long_window` - Long period for double smoothing (default: 25)
/// * `short_window` - Short period for double smoothing (default: 13)
///
/// # Returns
/// Numpy array with TSI values
#[pyfunction]
#[pyo3(name = "true_strength_index_numba", signature = (close, r=25, s=13))]
pub fn tsi<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    r: usize,
    s: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let close_slice = close.as_slice()?;
    let len = close_slice.len();

    let mut price_change = vec![0.0; len];
    for i in 1..len {
        price_change[i] = close_slice[i] - close_slice[i - 1];
    }

    let abs_price_change: Vec<f64> = price_change.iter().map(|x| x.abs()).collect();

    let alpha_long = 2.0 / (r as f64 + 1.0);
    let alpha_short = 2.0 / (s as f64 + 1.0);

    let ema1_pc = ema_kernel(&price_change, alpha_long, true);
    let ema2_pc = ema_kernel(&ema1_pc, alpha_short, true);

    let ema1_abspc = ema_kernel(&abs_price_change, alpha_long, true);
    let ema2_abspc = ema_kernel(&ema1_abspc, alpha_short, true);

    let mut tsi_values = vec![f64::NAN; len];
    for i in 0..len {
        if ema2_abspc[i] != 0.0 && !ema2_abspc[i].is_nan() {
            tsi_values[i] = 100.0 * (ema2_pc[i] / ema2_abspc[i]);
        }
    }

    Ok(PyArray1::from_vec(py, tsi_values))
}

/// Awesome Oscillator
///
/// # Arguments
/// * `high` - High price series
/// * `low` - Low price series
/// * `fast_window` - Fast SMA period (default: 5)
/// * `slow_window` - Slow SMA period (default: 34)
///
/// # Returns
/// Numpy array with Awesome Oscillator values
#[pyfunction]
#[pyo3(name = "awesome_oscillator_numba", signature = (high, low, n1=5, n2=34))]
pub fn awesome_oscillator<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    n1: usize,
    n2: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let len = high_slice.len();

    let mut midpoint = vec![0.0; len];
    for i in 0..len {
        midpoint[i] = (high_slice[i] + low_slice[i]) / 2.0;
    }

    let sma_fast = sma_kernel(&midpoint, n1);
    let sma_slow = sma_kernel(&midpoint, n2);

    let mut ao = vec![f64::NAN; len];
    for i in 0..len {
        if !sma_fast[i].is_nan() && !sma_slow[i].is_nan() {
            ao[i] = sma_fast[i] - sma_slow[i];
        }
    }

    Ok(PyArray1::from_vec(py, ao))
}

/// KAMA - Kaufman's Adaptive Moving Average
///
/// # Arguments
/// * `data` - Price data series (typically close prices)
/// * `n` - Efficiency Ratio period (default: 10)
/// * `fast_period` - Fast smoothing constant period (default: 2)
/// * `slow_period` - Slow smoothing constant period (default: 30)
///
/// # Returns
/// Numpy array with KAMA values
#[pyfunction]
#[pyo3(name = "kaufmans_adaptive_moving_average_numba", signature = (close, n=10, n_fast=2, n_slow=30))]
pub fn kama<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    n: usize,
    n_fast: usize,
    n_slow: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let close_slice = close.as_slice()?;
    let len = close_slice.len();
    let mut kama_values = vec![f64::NAN; len];

    if len <= n {
        return Ok(PyArray1::from_vec(py, kama_values));
    }

    let mut direction = vec![f64::NAN; len];
    for i in n..len {
        direction[i] = (close_slice[i] - close_slice[i - n]).abs();
    }

    let mut diffs = vec![0.0; len];
    for i in 1..len {
        diffs[i] = (close_slice[i] - close_slice[i - 1]).abs();
    }

    let mut volatility = vec![f64::NAN; len];
    for i in n..len {
        let mut sum = 0.0;
        for j in (i + 1 - n)..=i {
            sum += diffs[j];
        }
        volatility[i] = sum;
    }

    let mut er = vec![f64::NAN; len];
    for i in n..len {
        if volatility[i] != 0.0 {
            er[i] = direction[i] / volatility[i];
        } else {
            er[i] = 0.0;
        }
    }

    let fast_sc = 2.0 / (n_fast as f64 + 1.0);
    let slow_sc = 2.0 / (n_slow as f64 + 1.0);

    let mut sc = vec![f64::NAN; len];
    for i in n..len {
        if !er[i].is_nan() {
            sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc).powi(2);
        }
    }

    if len > n {
        kama_values[n - 1] = close_slice[n - 1];
        for i in n..len {
            if !sc[i].is_nan() {
                kama_values[i] = kama_values[i - 1] + sc[i] * (close_slice[i] - kama_values[i - 1]);
            }
        }
    }

    Ok(PyArray1::from_vec(py, kama_values))
}

/// ROC - Rate of Change
///
/// # Arguments
/// * `data` - Price data series (typically close prices)
/// * `n` - Period for rate of change calculation (default: 12)
///
/// # Returns
/// Numpy array with ROC values (percentage)
#[pyfunction]
#[pyo3(name = "rate_of_change_numba", signature = (close, n=12))]
pub fn roc<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    n: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let close_slice = close.as_slice()?;
    let len = close_slice.len();
    let mut roc_values = vec![f64::NAN; len];

    for i in n..len {
        if close_slice[i - n] != 0.0 {
            roc_values[i] = (close_slice[i] - close_slice[i - n]) / close_slice[i - n] * 100.0;
        }
    }

    Ok(PyArray1::from_vec(py, roc_values))
}

/// PVO - Percentage Volume Oscillator
///
/// # Arguments
/// * `volume` - Volume series
/// * `n_fast` - Fast EMA period (default: 12)
/// * `n_slow` - Slow EMA period (default: 26)
/// * `n_signal` - Signal line EMA period (default: 9)
///
/// # Returns
/// Tuple of (pvo_line, signal, histogram) as numpy arrays
#[pyfunction]
#[pyo3(name = "percentage_volume_oscillator_numba", signature = (volume, n_fast=12, n_slow=26, n_signal=9))]
pub fn pvo<'py>(
    py: Python<'py>,
    volume: PyReadonlyArray1<'py, f64>,
    n_fast: usize,
    n_slow: usize,
    n_signal: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let volume_slice = volume.as_slice()?;
    let len = volume_slice.len();

    let alpha_fast = 2.0 / (n_fast as f64 + 1.0);
    let alpha_slow = 2.0 / (n_slow as f64 + 1.0);

    let ema_fast = ema_kernel(volume_slice, alpha_fast, false);
    let ema_slow = ema_kernel(volume_slice, alpha_slow, false);

    let mut pvo_line = vec![f64::NAN; len];
    for i in 0..len {
        if ema_slow[i] != 0.0 && !ema_slow[i].is_nan() {
            pvo_line[i] = (ema_fast[i] - ema_slow[i]) / ema_slow[i] * 100.0;
        }
    }

    let alpha_signal = 2.0 / (n_signal as f64 + 1.0);
    let signal = ema_kernel(&pvo_line, alpha_signal, true);

    let mut histogram = vec![f64::NAN; len];
    for i in 0..len {
        if !pvo_line[i].is_nan() && !signal[i].is_nan() {
            histogram[i] = pvo_line[i] - signal[i];
        }
    }

    Ok((
        PyArray1::from_vec(py, pvo_line),
        PyArray1::from_vec(py, signal),
        PyArray1::from_vec(py, histogram),
    ))
}

/// Momentum - Simple Momentum Indicator
///
/// # Arguments
/// * `data` - Price data series
/// * `n` - Momentum period (default: 10)
///
/// # Returns
/// Numpy array with momentum values
#[pyfunction]
#[pyo3(name = "momentum_numba", signature = (close, n=10))]
pub fn momentum<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    n: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let close_slice = close.as_slice()?;
    let len = close_slice.len();
    let mut mom_values = vec![f64::NAN; len];

    for i in n..len {
        mom_values[i] = close_slice[i] - close_slice[i - n];
    }

    Ok(PyArray1::from_vec(py, mom_values))
}
