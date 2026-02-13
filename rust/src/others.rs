/// Other utility indicators: Daily Returns, Log Returns, Cumulative Returns

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
