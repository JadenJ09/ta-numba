# src/ta_numba/others.py

import numpy as np
from numba import njit

# ==============================================================================
# Other Indicator Functions (Returns)
# ==============================================================================

@njit(fastmath=True)
def daily_return_numba(close: np.ndarray) -> np.ndarray:
    dr = np.full_like(close, np.nan)
    dr[1:] = (close[1:] - close[:-1]) / close[:-1] * 100.0
    return dr

@njit(fastmath=True)
def daily_log_return_numba(close: np.ndarray) -> np.ndarray:
    dlr = np.full_like(close, np.nan)
    dlr[1:] = np.log(close[1:] / close[:-1]) * 100.0
    return dlr

@njit(fastmath=True)
def cumulative_return_numba(close: np.ndarray) -> np.ndarray:
    cr = np.full_like(close, np.nan)
    if len(close) > 0:
        initial_price = close[0]
        if initial_price != 0:
            cr = ((close / initial_price) - 1) * 100.0
    return cr

@njit(fastmath=True)
def compound_log_return_numba(close: np.ndarray) -> np.ndarray:
    clr = np.full_like(close, np.nan)
    log_returns = np.full_like(close, np.nan)
    log_returns[1:] = np.log(close[1:] / close[:-1])
    
    for i in range(1, len(close)):
        clr[i] = np.exp(np.nansum(log_returns[1:i+1])) - 1
    clr = clr * 100.0
    return clr

# ==============================================================================
# Clean Public API Aliases
# ==============================================================================

daily_return = daily_return_numba
daily_log_return = daily_log_return_numba
cumulative_return = cumulative_return_numba
compound_log_return = compound_log_return_numba


@njit(fastmath=True)
def rolling_zscore_numba(data: np.ndarray, window: int = 20) -> np.ndarray:
    """Rolling Z-Score: (x - rolling_mean(x, w)) / rolling_std(x, w)."""
    result = np.full_like(data, np.nan)

    for i in range(window - 1, len(data)):
        window_slice = data[i - window + 1:i + 1]
        mean = 0.0
        for j in range(window):
            mean += window_slice[j]
        mean /= window

        var = 0.0
        for j in range(window):
            var += (window_slice[j] - mean) ** 2
        var /= window
        std = np.sqrt(var)

        if std != 0.0:
            result[i] = (data[i] - mean) / std
        else:
            result[i] = 0.0

    return result


@njit(fastmath=True)
def linear_regression_slope_numba(data: np.ndarray, window: int = 14) -> np.ndarray:
    """Rolling Linear Regression Slope.

    slope = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)
    where x = 0, 1, ..., n-1 (time indices within window).
    """
    n = len(data)
    result = np.full(n, np.nan)

    # Precompute constants for x = 0..window-1
    w = window
    sum_x = w * (w - 1) / 2.0
    sum_x2 = w * (w - 1) * (2 * w - 1) / 6.0
    denom = w * sum_x2 - sum_x * sum_x

    if denom == 0.0:
        return result

    for i in range(window - 1, n):
        sum_y = 0.0
        sum_xy = 0.0
        for j in range(window):
            y = data[i - window + 1 + j]
            sum_y += y
            sum_xy += j * y
        result[i] = (w * sum_xy - sum_x * sum_y) / denom

    return result


@njit(fastmath=True)
def rolling_percentile_numba(data: np.ndarray, window: int = 120) -> np.ndarray:
    """Rolling Percentile: fraction of values in window <= current value."""
    n = len(data)
    result = np.full(n, np.nan)

    for i in range(window - 1, n):
        current = data[i]
        count = 0
        for j in range(i - window + 1, i + 1):
            if data[j] <= current:
                count += 1
        result[i] = count / window

    return result


rolling_zscore = rolling_zscore_numba
linear_regression_slope = linear_regression_slope_numba
rolling_percentile = rolling_percentile_numba


# --- Rust backend dispatch (transparent acceleration) ---
