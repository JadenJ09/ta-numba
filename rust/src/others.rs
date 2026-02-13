/// Other utility indicators: Daily Returns, Log Returns, Cumulative Returns,
/// Rolling Z-Score, Linear Regression Slope, Rolling Percentile

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Daily Return
///
/// # Arguments
/// * `data` - Price series (typically close)
///
/// # Returns
/// Numpy array with daily return values (percentage)
#[pyfunction]
#[pyo3(name = "daily_return_numba", signature = (close,))]
pub fn daily_return<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let close_slice = close.as_slice()?;
    let len = close_slice.len();

    let mut dr = vec![f64::NAN; len];

    for i in 1..len {
        dr[i] = (close_slice[i] - close_slice[i - 1]) / close_slice[i - 1] * 100.0;
    }

    Ok(PyArray1::from_vec(py, dr))
}

/// Daily Log Return
///
/// # Arguments
/// * `data` - Price series (typically close)
///
/// # Returns
/// Numpy array with daily log return values (percentage)
#[pyfunction]
#[pyo3(name = "daily_log_return_numba", signature = (close,))]
pub fn daily_log_return<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let close_slice = close.as_slice()?;
    let len = close_slice.len();

    let mut dlr = vec![f64::NAN; len];

    for i in 1..len {
        let ratio: f64 = close_slice[i] / close_slice[i - 1];
        dlr[i] = ratio.ln() * 100.0;
    }

    Ok(PyArray1::from_vec(py, dlr))
}

/// Cumulative Return
///
/// # Arguments
/// * `data` - Price series (typically close)
///
/// # Returns
/// Numpy array with cumulative return values (percentage)
#[pyfunction]
#[pyo3(name = "cumulative_return_numba", signature = (close,))]
pub fn cumulative_return<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let close_slice = close.as_slice()?;
    let len = close_slice.len();

    let mut cr = vec![f64::NAN; len];

    if len > 0 {
        let initial_price = close_slice[0];
        if initial_price != 0.0 {
            for i in 0..len {
                cr[i] = ((close_slice[i] / initial_price) - 1.0) * 100.0;
            }
        }
    }

    Ok(PyArray1::from_vec(py, cr))
}

/// Compound Log Return
///
/// Cumulative sum of log returns, exponentiated and converted to percentage.
///
/// # Arguments
/// * `close` - Price series (typically close)
///
/// # Returns
/// Numpy array with compound log return values (percentage)
#[pyfunction]
#[pyo3(name = "compound_log_return_numba", signature = (close,))]
pub fn compound_log_return<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let close_slice = close.as_slice()?;
    let len = close_slice.len();

    let mut clr = vec![f64::NAN; len];

    if len == 0 {
        return Ok(PyArray1::from_vec(py, clr));
    }

    let mut cumulative_log_return = 0.0;

    for i in 1..len {
        if close_slice[i] > 0.0 && close_slice[i - 1] > 0.0 {
            let log_ret = (close_slice[i] / close_slice[i - 1]).ln();
            cumulative_log_return += log_ret;
            clr[i] = (cumulative_log_return.exp() - 1.0) * 100.0;
        }
    }

    Ok(PyArray1::from_vec(py, clr))
}

/// Rolling Z-Score
///
/// (x - rolling_mean(x, w)) / rolling_std(x, w)
///
/// # Arguments
/// * `data` - Data series
/// * `window` - Rolling window size (default: 20)
///
/// # Returns
/// Numpy array with z-score values
#[pyfunction]
#[pyo3(name = "rolling_zscore_numba", signature = (data, window=20))]
pub fn rolling_zscore<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data_slice = data.as_slice()?;
    let len = data_slice.len();
    let mut result = vec![f64::NAN; len];

    if window == 0 || window > len {
        return Ok(PyArray1::from_vec(py, result));
    }

    for i in (window - 1)..len {
        let start = i + 1 - window;
        let slice = &data_slice[start..=i];

        let mean: f64 = slice.iter().sum::<f64>() / window as f64;
        let variance: f64 = slice.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / window as f64;
        let std = variance.sqrt();

        if std != 0.0 {
            result[i] = (data_slice[i] - mean) / std;
        } else {
            result[i] = 0.0;
        }
    }

    Ok(PyArray1::from_vec(py, result))
}

/// Linear Regression Slope
///
/// Rolling linear regression slope using least squares.
/// slope = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)
///
/// # Arguments
/// * `data` - Data series
/// * `window` - Rolling window size (default: 14)
///
/// # Returns
/// Numpy array with slope values
#[pyfunction]
#[pyo3(name = "linear_regression_slope_numba", signature = (data, window=14))]
pub fn linear_regression_slope<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data_slice = data.as_slice()?;
    let len = data_slice.len();
    let mut result = vec![f64::NAN; len];

    if window == 0 || window > len {
        return Ok(PyArray1::from_vec(py, result));
    }

    let w = window as f64;
    let sum_x = w * (w - 1.0) / 2.0;
    let sum_x2 = w * (w - 1.0) * (2.0 * w - 1.0) / 6.0;
    let denom = w * sum_x2 - sum_x * sum_x;

    if denom == 0.0 {
        return Ok(PyArray1::from_vec(py, result));
    }

    for i in (window - 1)..len {
        let start = i + 1 - window;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;

        for j in 0..window {
            let y = data_slice[start + j];
            sum_y += y;
            sum_xy += j as f64 * y;
        }

        result[i] = (w * sum_xy - sum_x * sum_y) / denom;
    }

    Ok(PyArray1::from_vec(py, result))
}

/// Rolling Percentile
///
/// Fraction of values in the window that are <= current value.
///
/// # Arguments
/// * `data` - Data series
/// * `window` - Rolling window size (default: 120)
///
/// # Returns
/// Numpy array with percentile values (0.0 to 1.0)
#[pyfunction]
#[pyo3(name = "rolling_percentile_numba", signature = (data, window=120))]
pub fn rolling_percentile<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data_slice = data.as_slice()?;
    let len = data_slice.len();
    let mut result = vec![f64::NAN; len];

    if window == 0 || window > len {
        return Ok(PyArray1::from_vec(py, result));
    }

    for i in (window - 1)..len {
        let start = i + 1 - window;
        let current = data_slice[i];
        let mut count = 0usize;

        for j in start..=i {
            if data_slice[j] <= current {
                count += 1;
            }
        }

        result[i] = count as f64 / window as f64;
    }

    Ok(PyArray1::from_vec(py, result))
}
