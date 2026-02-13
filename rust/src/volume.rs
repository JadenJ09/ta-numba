/// Volume indicators: VWEMA, CMF, Force Index, MFI, A/D, OBV, EOM, VPT, NVI, VWAP

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use crate::helpers::{ema_kernel, rolling_sum};

/// Money Flow Index (MFI)
///
/// # Arguments
/// * `high` - High price series
/// * `low` - Low price series
/// * `close` - Close price series
/// * `volume` - Volume series
/// * `n` - Period for MFI calculation (default: 14)
///
/// # Returns
/// Numpy array with MFI values (0 to 100)
#[pyfunction]
#[pyo3(name = "money_flow_index_numba", signature = (high, low, close, volume, n=14))]
pub fn mfi<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    n: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    let volume_slice = volume.as_slice()?;
    let len = high_slice.len();

    let mut tp = vec![0.0; len];
    for i in 0..len {
        tp[i] = (high_slice[i] + low_slice[i] + close_slice[i]) / 3.0;
    }

    let mut positive_mf = vec![0.0; len];
    let mut negative_mf = vec![0.0; len];

    for i in 1..len {
        let rmf = tp[i] * volume_slice[i];
        if tp[i] > tp[i - 1] {
            positive_mf[i] = rmf;
        } else if tp[i] < tp[i - 1] {
            negative_mf[i] = rmf;
        }
    }

    let mut mfi_values = vec![f64::NAN; len];
    for i in (n - 1)..len {
        let pos_sum: f64 = positive_mf[(i + 1 - n)..=i].iter().sum();
        let neg_sum: f64 = negative_mf[(i + 1 - n)..=i].iter().sum();

        if neg_sum == 0.0 {
            mfi_values[i] = 100.0;
        } else {
            let mfr = pos_sum / neg_sum;
            mfi_values[i] = 100.0 - (100.0 / (1.0 + mfr));
        }
    }

    Ok(PyArray1::from_vec(py, mfi_values))
}

/// Accumulation/Distribution Index
///
/// # Arguments
/// * `high` - High price series
/// * `low` - Low price series
/// * `close` - Close price series
/// * `volume` - Volume series
///
/// # Returns
/// Numpy array with A/D values
#[pyfunction]
#[pyo3(name = "acc_dist_index_numba", signature = (high, low, close, volume))]
pub fn acc_dist_index<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    let volume_slice = volume.as_slice()?;
    let len = high_slice.len();

    let mut clv = vec![0.0; len];
    for i in 0..len {
        let range = high_slice[i] - low_slice[i];
        if range != 0.0 {
            clv[i] = ((close_slice[i] - low_slice[i]) - (high_slice[i] - close_slice[i])) / range;
        } else {
            clv[i] = 0.0;
        }
    }

    let mut mfv = vec![0.0; len];
    for i in 0..len {
        mfv[i] = clv[i] * volume_slice[i];
    }

    let mut ad = vec![0.0; len];
    ad[0] = mfv[0];
    for i in 1..len {
        ad[i] = ad[i - 1] + mfv[i];
    }

    Ok(PyArray1::from_vec(py, ad))
}

/// On-Balance Volume (OBV)
///
/// # Arguments
/// * `close` - Close price series
/// * `volume` - Volume series
///
/// # Returns
/// Numpy array with OBV values
#[pyfunction]
#[pyo3(name = "on_balance_volume_numba", signature = (close, volume))]
pub fn obv<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let close_slice = close.as_slice()?;
    let volume_slice = volume.as_slice()?;
    let len = close_slice.len();

    let mut obv_values = vec![0.0; len];
    obv_values[0] = volume_slice[0];

    for i in 1..len {
        if close_slice[i] < close_slice[i - 1] {
            obv_values[i] = obv_values[i - 1] - volume_slice[i];
        } else {
            obv_values[i] = obv_values[i - 1] + volume_slice[i];
        }
    }

    Ok(PyArray1::from_vec(py, obv_values))
}

/// Chaikin Money Flow (CMF)
///
/// # Arguments
/// * `high` - High price series
/// * `low` - Low price series
/// * `close` - Close price series
/// * `volume` - Volume series
/// * `n` - Period for CMF calculation (default: 20)
///
/// # Returns
/// Numpy array with CMF values (-1 to 1)
#[pyfunction]
#[pyo3(name = "chaikin_money_flow_numba", signature = (high, low, close, volume, n=20))]
pub fn chaikin_money_flow<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    n: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    let volume_slice = volume.as_slice()?;
    let len = high_slice.len();

    let mut mfm = vec![0.0; len];
    let mut mfv = vec![0.0; len];

    for i in 0..len {
        let range = high_slice[i] - low_slice[i];
        if range != 0.0 {
            mfm[i] = ((close_slice[i] - low_slice[i]) - (high_slice[i] - close_slice[i])) / range;
        } else {
            mfm[i] = 0.0;
        }
        mfv[i] = mfm[i] * volume_slice[i];
    }

    let sum_mfv = rolling_sum(&mfv, n);
    let sum_volume = rolling_sum(volume_slice, n);

    let mut cmf = vec![f64::NAN; len];
    for i in (n - 1)..len {
        if sum_volume[i] != 0.0 && !sum_volume[i].is_nan() {
            cmf[i] = sum_mfv[i] / sum_volume[i];
        }
    }

    Ok(PyArray1::from_vec(py, cmf))
}

/// Force Index
///
/// # Arguments
/// * `close` - Close price series
/// * `volume` - Volume series
/// * `n` - Period for EMA smoothing (default: 13)
///
/// # Returns
/// Numpy array with Force Index values
#[pyfunction]
#[pyo3(name = "force_index_numba", signature = (close, volume, n=13))]
pub fn force_index<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    n: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let close_slice = close.as_slice()?;
    let volume_slice = volume.as_slice()?;
    let len = close_slice.len();

    let mut fi_raw = vec![f64::NAN; len];
    fi_raw[0] = 0.0;

    for i in 1..len {
        fi_raw[i] = (close_slice[i] - close_slice[i - 1]) * volume_slice[i];
    }

    let alpha = 2.0 / (n as f64 + 1.0);
    let result = ema_kernel(&fi_raw, alpha, false);

    Ok(PyArray1::from_vec(py, result))
}

/// Ease of Movement (EOM)
///
/// # Arguments
/// * `high` - High price series
/// * `low` - Low price series
/// * `volume` - Volume series
/// * `_n` - Period for SMA (default: 14, currently unused)
///
/// # Returns
/// Numpy array with EOM values
#[pyfunction]
#[pyo3(name = "ease_of_movement_numba", signature = (high, low, volume, n=14))]
#[allow(unused_variables)]
pub fn eom<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    n: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let volume_slice = volume.as_slice()?;
    let len = high_slice.len();

    let mut emv_raw = vec![f64::NAN; len];

    for i in 1..len {
        if volume_slice[i] != 0.0 {
            let distance_moved = ((high_slice[i] - high_slice[i - 1]) + (low_slice[i] - low_slice[i - 1])) / 2.0;
            let box_height = high_slice[i] - low_slice[i];
            emv_raw[i] = distance_moved * box_height / volume_slice[i] * 100000000.0;
        }
    }

    // Return raw EMV values (no SMA applied) to match ta library
    Ok(PyArray1::from_vec(py, emv_raw))
}

/// Volume Price Trend (VPT)
///
/// # Arguments
/// * `close` - Close price series
/// * `volume` - Volume series
///
/// # Returns
/// Numpy array with VPT values
#[pyfunction]
#[pyo3(name = "volume_price_trend_numba", signature = (close, volume))]
pub fn vpt<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let close_slice = close.as_slice()?;
    let volume_slice = volume.as_slice()?;
    let len = close_slice.len();

    let mut pct_change = vec![0.0; len];
    for i in 1..len {
        pct_change[i] = (close_slice[i] - close_slice[i - 1]) / close_slice[i - 1];
    }

    let mut vpt_change = vec![0.0; len];
    for i in 0..len {
        vpt_change[i] = volume_slice[i] * pct_change[i];
    }

    let mut vpt_values = vec![0.0; len];
    vpt_values[0] = vpt_change[0];
    for i in 1..len {
        vpt_values[i] = vpt_values[i - 1] + vpt_change[i];
    }

    Ok(PyArray1::from_vec(py, vpt_values))
}

/// Negative Volume Index (NVI)
///
/// # Arguments
/// * `close` - Close price series
/// * `volume` - Volume series
///
/// # Returns
/// Numpy array with NVI values
#[pyfunction]
#[pyo3(name = "negative_volume_index_numba", signature = (close, volume))]
pub fn nvi<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let close_slice = close.as_slice()?;
    let volume_slice = volume.as_slice()?;
    let len = close_slice.len();

    let mut nvi_values = vec![f64::NAN; len];
    nvi_values[0] = 1000.0;

    let mut pct_change = vec![0.0; len];
    for i in 1..len {
        pct_change[i] = (close_slice[i] - close_slice[i - 1]) / close_slice[i - 1];
    }

    for i in 1..len {
        if volume_slice[i] < volume_slice[i - 1] {
            nvi_values[i] = nvi_values[i - 1] * (1.0 + pct_change[i]);
        } else {
            nvi_values[i] = nvi_values[i - 1];
        }
    }

    Ok(PyArray1::from_vec(py, nvi_values))
}

/// Volume Weighted Average Price (VWAP)
///
/// # Arguments
/// * `high` - High price series
/// * `low` - Low price series
/// * `close` - Close price series
/// * `volume` - Volume series
///
/// # Returns
/// Numpy array with VWAP values
#[pyfunction]
#[pyo3(name = "volume_weighted_average_price_numba", signature = (high, low, close, volume, n=14))]
pub fn vwap<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    n: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    let volume_slice = volume.as_slice()?;
    let len = high_slice.len();

    let mut tp = vec![0.0; len];
    for i in 0..len {
        tp[i] = (high_slice[i] + low_slice[i] + close_slice[i]) / 3.0;
    }

    let mut vwap_values = vec![f64::NAN; len];
    
    if len < n {
        return Ok(PyArray1::from_vec(py, vwap_values));
    }

    for i in (n - 1)..len {
        let window_start = i + 1 - n;
        let mut sum_tpv = 0.0;
        let mut sum_vol = 0.0;
        
        for j in window_start..=i {
            sum_tpv += tp[j] * volume_slice[j];
            sum_vol += volume_slice[j];
        }
        
        if sum_vol != 0.0 {
            vwap_values[i] = sum_tpv / sum_vol;
        }
    }

    Ok(PyArray1::from_vec(py, vwap_values))
}

/// VWEMA - Volume-Weighted Exponential Moving Average
///
/// # Arguments
/// * `high` - High price series
/// * `low` - Low price series
/// * `close` - Close price series
/// * `volume` - Volume series
/// * `vwma_period` - Period for VWAP calculation (default: 14)
/// * `ema_period` - Period for EMA of VWAP (default: 20)
///
/// # Returns
/// Numpy array with VWEMA values
#[pyfunction]
#[pyo3(name = "volume_weighted_exponential_moving_average_numba", signature = (high, low, close, volume, n_vwma=14, n_ema=20))]
pub fn vwema<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    n_vwma: usize,
    n_ema: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    let volume_slice = volume.as_slice()?;
    let len = high_slice.len();

    let mut typical_price = vec![0.0; len];
    for i in 0..len {
        typical_price[i] = (high_slice[i] + low_slice[i] + close_slice[i]) / 3.0;
    }

    let mut vwap = vec![f64::NAN; len];
    for i in (n_vwma - 1)..len {
        let start_idx = i + 1 - n_vwma;
        let mut sum_tp_vol = 0.0;
        let mut sum_vol = 0.0;

        for j in start_idx..=i {
            sum_tp_vol += typical_price[j] * volume_slice[j];
            sum_vol += volume_slice[j];
        }

        if sum_vol != 0.0 {
            vwap[i] = sum_tp_vol / sum_vol;
        } else {
            vwap[i] = typical_price[start_idx..=i].iter().sum::<f64>() / n_vwma as f64;
        }
    }

    let alpha = 2.0 / (n_ema as f64 + 1.0);
    let vwema_values = ema_kernel(&vwap, alpha, true);

    Ok(PyArray1::from_vec(py, vwema_values))
}

/// Volume Ratio: volume / SMA(volume, window)
///
/// # Arguments
/// * `volume` - Volume series
/// * `window` - SMA window for volume averaging (default: 50)
///
/// # Returns
/// Numpy array with volume ratio values
#[pyfunction]
#[pyo3(name = "volume_ratio_numba", signature = (volume, window=50))]
pub fn volume_ratio<'py>(
    py: Python<'py>,
    volume: PyReadonlyArray1<'py, f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let volume_slice = volume.as_slice()?;
    let len = volume_slice.len();

    let sma = crate::helpers::sma_kernel(volume_slice, window);

    let mut result = vec![f64::NAN; len];
    for i in 0..len {
        if !sma[i].is_nan() && sma[i] != 0.0 {
            result[i] = volume_slice[i] / sma[i];
        }
    }

    Ok(PyArray1::from_vec(py, result))
}
