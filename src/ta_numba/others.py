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
