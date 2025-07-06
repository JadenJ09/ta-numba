# src/ta_numba/helpers.py

import numpy as np
from numba import njit

# ==============================================================================
# Core Numba Helper Functions
# These functions are the building blocks for many indicators and are optimized
# to match the specific logic of the 'ta' library for 1-to-1 output.
# (Corrected to match `ta` library's `adjust=False` EMA)
# ==============================================================================

@njit
def _sma(arr, window):
    """Helper function for calculating Simple Moving Average."""
    out = np.full_like(arr, np.nan, dtype=np.float64)
    for i in range(window - 1, len(arr)):
        window_slice = arr[i-window+1:i+1]
        if not np.any(np.isnan(window_slice)):
            out[i] = np.mean(window_slice)
    return out

@njit
def _sma_numba(data: np.ndarray, window: int = 20, min_periods: int = 1) -> np.ndarray:
    """Numba-compatible SMA calculation with configurable min_periods behavior."""
    sma = np.full_like(data, np.nan)
    for i in range(len(data)):
        # Use expanding window until we have window periods, then use rolling window
        start_idx = max(0, i - window + 1)
        window_size = i - start_idx + 1
        if window_size >= min_periods:
            sma[i] = np.mean(data[start_idx:i+1])
    return sma

@njit
def _ema(arr, window, adjust=False):
    """
    Helper function for calculating Exponential Moving Average.
    Can replicate both pandas `adjust=True` and `adjust=False`.
    """
    alpha = 2.0 / (window + 1.0)
    
    if adjust:
        out = np.full_like(arr, np.nan, dtype=np.float64)
        one_minus_alpha = 1.0 - alpha
        weighted_sum = 0.0
        weight = 0.0
        for i in range(len(arr)):
            if not np.isnan(arr[i]):
                weighted_sum = arr[i] + one_minus_alpha * weighted_sum
                weight = 1.0 + one_minus_alpha * weight
                if weight != 0:
                    out[i] = weighted_sum / weight
        return out
    else: # adjust=False (ta library default)
        out = np.full_like(arr, np.nan, dtype=np.float64)
        first_valid_idx = -1
        for i in range(len(arr)):
            if not np.isnan(arr[i]):
                first_valid_idx = i
                break
        if first_valid_idx == -1: return out
        
        out[first_valid_idx] = arr[first_valid_idx]
        for i in range(first_valid_idx + 1, len(arr)):
            prev_val = out[i-1]
            curr_val = arr[i]
            if np.isnan(curr_val):
                out[i] = prev_val
            else:
                out[i] = curr_val * alpha + prev_val * (1 - alpha)
        return out

@njit
def _ema_numba_unadjusted(data: np.ndarray, n: int) -> np.ndarray:
    """Numba EMA calculation matching pandas.ewm(span=n, adjust=False) exactly."""
    ema = np.full_like(data, np.nan)
    if len(data) == 0:
        return ema

    alpha = 2.0 / (n + 1.0)
    
    # Find the first non-NaN value to start the EMA
    first_valid_idx = -1
    for i in range(len(data)):
        if not np.isnan(data[i]):
            first_valid_idx = i
            break

    if first_valid_idx == -1: # All NaNs
        return ema

    # Initialize EMA with first valid value
    ema_val = data[first_valid_idx]
    ema[first_valid_idx] = ema_val

    # Calculate EMA for subsequent values
    for i in range(first_valid_idx + 1, len(data)):
        if np.isnan(data[i]):
            # Carry forward previous EMA if current value is NaN
            ema[i] = ema_val
        else:
            # Standard EMA formula: new_ema = alpha * current + (1 - alpha) * prev_ema
            ema_val = alpha * data[i] + (1 - alpha) * ema_val
            ema[i] = ema_val

    return ema

@njit
def _ema_numba_adjusted(data: np.ndarray, n: int) -> np.ndarray:
    """Numba EMA calculation matching pandas.ewm(adjust=True)."""
    ema = np.full_like(data, np.nan)
    if len(data) == 0:
        return ema

    alpha = 2.0 / (n + 1.0)
    one_minus_alpha = 1.0 - alpha
    
    # Find the first non-NaN value to start calculation
    first_valid_idx = -1
    for i in range(len(data)):
        if not np.isnan(data[i]):
            first_valid_idx = i
            break
    
    if first_valid_idx == -1:  # All NaN values
        return ema
    
    # Initialize state variables for adjust=True calculation
    weighted_sum = 0.0
    weight_sum = 0.0
    
    for i in range(first_valid_idx, len(data)):
        if not np.isnan(data[i]):
            # For adjust=True, we maintain cumulative weighted sum and weights
            weighted_sum = data[i] + one_minus_alpha * weighted_sum
            weight_sum = 1.0 + one_minus_alpha * weight_sum
            
            if weight_sum > 0:
                ema[i] = weighted_sum / weight_sum
        else:
            # For NaN values, carry forward the previous EMA value
            if i > first_valid_idx:
                ema[i] = ema[i-1]
    
    return ema

@njit
def _wilders_ema_numba_adjusted(data: np.ndarray, n: int) -> np.ndarray:
    """
    Optimized O(n) Wilder's EMA (SMMA) matching pandas ewm(alpha=1/n, adjust=True).
    
    Key behavior:
    - Maintains cumulative weighted sum with proper time decay
    - NaN values are skipped but time periods still count for decay
    - Carries forward values during NaN periods
    - Proper adjust=True weighting scheme
    
    Optimization: Uses iterative calculation instead of recalculating from scratch each time.
    """
    wema = np.full_like(data, np.nan)
    if len(data) == 0:
        return wema
        
    alpha = 1.0 / n
    one_minus_alpha = 1.0 - alpha
    
    # State variables for iterative calculation
    weighted_sum = 0.0
    weight_sum = 0.0
    last_valid_idx = -1
    
    for i in range(len(data)):
        if not np.isnan(data[i]):
            if last_valid_idx == -1:
                # First valid value
                weighted_sum = data[i]
                weight_sum = 1.0
                wema[i] = data[i]
            else:
                # Apply time decay to existing weighted sum and weight sum
                time_gap = i - last_valid_idx
                decay_factor = one_minus_alpha ** time_gap
                
                # Update existing components with decay
                weighted_sum *= decay_factor
                weight_sum *= decay_factor
                
                # Add current value
                weighted_sum += data[i]
                weight_sum += 1.0
                
                # Calculate new EMA value
                wema[i] = weighted_sum / weight_sum
            
            last_valid_idx = i
    
    # Fill NaN positions with carry-forward values
    last_valid_value = np.nan
    for i in range(len(data)):
        if not np.isnan(wema[i]):
            last_valid_value = wema[i]
        elif not np.isnan(last_valid_value):
            wema[i] = last_valid_value
    
    return wema

@njit
def _wilders_ema_ta_style(data: np.ndarray, n: int) -> np.ndarray:
    wema = np.full_like(data, np.nan)
    if len(data) < n:
        return wema

    # Find the first n non-NaN values to calculate initial SMA
    non_nan_values = []
    non_nan_indices = []
    
    for i in range(len(data)):
        if not np.isnan(data[i]):
            non_nan_values.append(data[i])
            non_nan_indices.append(i)
            if len(non_nan_values) == n:
                break
    
    if len(non_nan_values) == 0:  # All NaNs
        return wema
    
    # Calculate initial SMA from the first n non-NaN values
    initial_sum = 0.0
    for val in non_nan_values:
        initial_sum += val
    
    wema_val = initial_sum / len(non_nan_values)
    
    # Start the EMA from the index where we have n non-NaN values
    start_idx = non_nan_indices[-1] if len(non_nan_indices) == n else len(data) - 1
    wema[start_idx] = wema_val

    # Subsequent Wilder's EMA calculation
    for i in range(start_idx + 1, len(data)):
        if np.isnan(data[i]):
            wema[i] = wema[i-1]
        else:
            wema_val = (wema_val * (n - 1) + data[i]) / n
            wema[i] = wema_val
    return wema

@njit
def _wilders_ema_adaptive(data: np.ndarray, n: int) -> np.ndarray:
    """
    Adaptive Wilder's EMA that chooses the best implementation based on data length.
    - Uses _wilders_ema_ta_style for arrays >= window size (faster, matches ta-lib exactly)
    - Uses _wilders_ema_numba_adjusted for shorter arrays (more flexible)
    """
    if len(data) >= n:
        return _wilders_ema_ta_style(data, n)
    else:
        return _wilders_ema_numba_adjusted(data, n)

@njit
def _rma(arr, window):
    """
    Helper function for calculating Wilder's Smoothing (RMA).
    This is equivalent to an EMA with alpha = 1 / window.
    """
    out = np.full_like(arr, np.nan, dtype=np.float64)
    alpha = 1.0 / window
    
    first_valid_idx = -1
    for i in range(len(arr)):
        if not np.isnan(arr[i]):
            first_valid_idx = i
            break
    
    if first_valid_idx == -1: return out
    
    start_idx = first_valid_idx + window
    if start_idx > len(arr): return out

    first_rma = np.mean(arr[first_valid_idx : first_valid_idx + window])
    out[start_idx - 1] = first_rma

    for i in range(start_idx, len(arr)):
        prev_val = out[i-1]
        curr_val = arr[i]
        if np.isnan(prev_val):
            out[i] = np.mean(arr[i-window+1:i+1])
        elif np.isnan(curr_val):
             out[i] = prev_val
        else:
             out[i] = curr_val * alpha + prev_val * (1 - alpha)
    return out

@njit
def _wma(arr, window):
    """Helper function for calculating Weighted Moving Average."""
    out = np.full_like(arr, np.nan, dtype=np.float64)
    weights = np.arange(1, window + 1, dtype=np.float64)
    weight_sum = np.sum(weights)
    for i in range(window - 1, len(arr)):
        window_slice = arr[i-window+1:i+1]
        if not np.any(np.isnan(window_slice)):
            out[i] = np.sum(weights * window_slice) / weight_sum
    return out

@njit
def _ewm_numba(arr, alpha):
    """Exponential weighted moving average with adjust=False (like pandas)."""
    ewm = np.full_like(arr, np.nan)
    if len(arr) == 0:
        return ewm
    
    ewm[0] = arr[0]
    for i in range(1, len(arr)):
        ewm[i] = alpha * arr[i] + (1 - alpha) * ewm[i-1]
    return ewm

@njit
def _true_range_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    tr = np.full_like(high, np.nan)
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    return tr