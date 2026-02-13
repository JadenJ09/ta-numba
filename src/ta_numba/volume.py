# src/ta_numba/volume.py

import numpy as np
from numba import njit

# Import helper functions from the same package
from .helpers import _ema_numba_unadjusted

# ==============================================================================
# Volume Indicator Functions
# ==============================================================================

@njit(fastmath=True)
def money_flow_index_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, n: int = 14) -> np.ndarray:
    tp = (high + low + close) / 3.0
    up_down = np.zeros_like(tp)
    up_down[1:][tp[1:] > tp[:-1]] = 1
    up_down[1:][tp[1:] < tp[:-1]] = -1

    rmf = tp * volume * up_down
    
    positive_mf = np.zeros_like(rmf)
    positive_mf[rmf > 0] = rmf[rmf > 0]
    
    negative_mf = np.zeros_like(rmf)
    negative_mf[rmf < 0] = np.abs(rmf[rmf < 0])
    
    mfi = np.full_like(close, np.nan)
    for i in range(n - 1, len(close)):
        pos_sum = np.sum(positive_mf[i-n+1:i+1])
        neg_sum = np.sum(negative_mf[i-n+1:i+1])
        if neg_sum == 0:
            mfi[i] = 100.0
        else:
            mfr = pos_sum / neg_sum
            mfi[i] = 100.0 - (100.0 / (1.0 + mfr))
    return mfi

@njit(fastmath=True)
def acc_dist_index_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    h_minus_l = high - low
    mfm = np.zeros_like(close)
    for i in range(len(h_minus_l)):
        if h_minus_l[i] != 0:
            mfm[i] = ((close[i] - low[i]) - (high[i] - close[i])) / h_minus_l[i]
    mfv = mfm * volume
    return np.cumsum(mfv)

@njit
def on_balance_volume_numba(close, volume):
    """Corrected OBV to match `ta` library logic exactly."""
    n = len(close)
    obv = np.zeros(n, dtype=np.float64)
    
    # First element: since close.shift(1)[0] is NaN, np.where treats condition as False
    # So the first element of the array passed to cumsum() is +volume[0]
    obv[0] = volume[0]
    
    for i in range(1, n):
        if close[i] < close[i-1]:
            obv[i] = obv[i-1] - volume[i]
        else: # if close >= prev_close, ta library adds volume
            obv[i] = obv[i-1] + volume[i]
    return obv

@njit(fastmath=True)
def chaikin_money_flow_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, n: int = 20) -> np.ndarray:
    h_minus_l = high - low
    mfm = np.zeros_like(close)
    for i in range(len(h_minus_l)):
        if h_minus_l[i] != 0:
            mfm[i] = ((close[i] - low[i]) - (high[i] - close[i])) / h_minus_l[i]
    mfv = mfm * volume
    cmf = np.full_like(close, np.nan)
    for i in range(n - 1, len(close)):
        sum_mfv = np.sum(mfv[i-n+1:i+1])
        sum_vol = np.sum(volume[i-n+1:i+1])
        if sum_vol != 0:
            cmf[i] = sum_mfv / sum_vol
    return cmf

@njit
def force_index_numba(close: np.ndarray, volume: np.ndarray, n: int = 13) -> np.ndarray:
    """Force Index: (close - prev_close) * volume, then EMA."""
    fi1 = np.full_like(close, np.nan)
    # First element is NaN due to (close[0] - close[-1]) being undefined
    for i in range(1, len(close)):
        fi1[i] = (close[i] - close[i-1]) * volume[i]
    
    return _ema_numba_unadjusted(fi1, n)

@njit(fastmath=True)
def ease_of_movement_numba(high, low, volume, n=14):
    """EOM to match `ta` library exactly: distance moved * box height / volume."""
    n = len(high)
    emv_raw = np.full(n, np.nan)
    
    # ta library uses: distance_moved * box_height / volume * 100000000
    # where distance_moved = (high.diff(1) + low.diff(1)) / 2
    # and box_height = high - low
    # NOTE: TA library does NOT apply SMA - it returns raw EMV values
    for i in range(1, n):
        if volume[i] != 0:
            # Distance moved = (high[i] - high[i-1] + low[i] - low[i-1]) / 2
            distance_moved = ((high[i] - high[i-1]) + (low[i] - low[i-1])) / 2.0
            # Box height = high - low
            box_height = high[i] - low[i]
            # Apply formula and scale
            emv_raw[i] = distance_moved * box_height / volume[i] * 100000000
    
    # Return raw EMV values (no SMA applied)
    return emv_raw

@njit(fastmath=True)
def volume_price_trend_numba(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    pct_change = np.zeros_like(close)
    pct_change[1:] = (close[1:] - close[:-1]) / close[:-1]
    vpt_change = volume * pct_change
    return np.cumsum(vpt_change)

@njit
def negative_volume_index_numba(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    nvi = np.full_like(close, np.nan)
    nvi[0] = 1000.0
    pct_change = np.zeros_like(close)
    pct_change[1:] = (close[1:] - close[:-1]) / close[:-1]
    
    for i in range(1, len(close)):
        if volume[i] < volume[i-1]:
            nvi[i] = nvi[i-1] * (1 + pct_change[i])
        else:
            nvi[i] = nvi[i-1]
    return nvi

@njit(fastmath=True)
def volume_weighted_average_price_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, n: int = 14) -> np.ndarray:
    tp = (high + low + close) / 3.0
    tpv = tp * volume
    vwap = np.full_like(close, np.nan)
    for i in range(n - 1, len(close)):
        sum_tpv = np.sum(tpv[i-n+1:i+1])
        sum_vol = np.sum(volume[i-n+1:i+1])
        if sum_vol != 0:
            vwap[i] = sum_tpv / sum_vol
    return vwap

@njit
def volume_weighted_exponential_moving_average_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, n_vwma: int = 14, n_ema: int = 20) -> np.ndarray:
    # --- Step 1: Calculate the VWMA (Moving VWAP) ---
    tp = (high + low + close) / 3.0
    tpv = tp * volume
    
    # Use a distinct variable name for the intermediate VWMA result
    vwma_values = np.full_like(close, np.nan)
    for i in range(n_vwma - 1, len(close)):
        sum_tpv = np.sum(tpv[i - n_vwma + 1:i + 1])
        sum_vol = np.sum(volume[i - n_vwma + 1:i + 1])
        if sum_vol != 0:
            vwma_values[i] = sum_tpv / sum_vol
            
    # --- Step 2: Calculate the EMA of the VWMA series (adjust=True logic) ---
    # Initialize the final output array based on the shape of the intermediate result
    vwema_output = np.full_like(vwma_values, np.nan)
    if len(vwma_values) == 0:
        return vwema_output

    alpha = 2.0 / (n_ema + 1.0)
    
    # Find the first valid data point to start the EMA
    first_valid_idx = -1
    for i in range(len(vwma_values)):
        if not np.isnan(vwma_values[i]):
            first_valid_idx = i
            break
    
    if first_valid_idx == -1: # All NaNs
        return vwema_output

    # Initialize EMA with the first valid VWMA value
    vwema_output[first_valid_idx] = vwma_values[first_valid_idx]
    
    # Subsequent values use the 'adjust=True' formula
    one_minus_alpha = 1.0 - alpha
    
    # For pandas adjust=True, we need to track the weighted sum and total weight
    # Initialize with the first value
    weighted_sum = vwma_values[first_valid_idx]
    weight_sum = 1.0
    
    for i in range(first_valid_idx + 1, len(vwma_values)):
        current_val = vwma_values[i]
        if np.isnan(current_val):
            vwema_output[i] = vwema_output[i-1] # Carry forward previous EMA
        else:
            # Decay the previous weighted sum and weight, then add current
            weighted_sum = one_minus_alpha * weighted_sum + current_val
            weight_sum = one_minus_alpha * weight_sum + 1.0
            
            if weight_sum != 0:
                vwema_output[i] = weighted_sum / weight_sum
            else:
                vwema_output[i] = np.nan
                
    return vwema_output

# ==============================================================================
# Clean Public API Aliases
# ==============================================================================

money_flow_index = money_flow_index_numba
acc_dist_index = acc_dist_index_numba
on_balance_volume = on_balance_volume_numba
chaikin_money_flow = chaikin_money_flow_numba
force_index = force_index_numba
ease_of_movement = ease_of_movement_numba
volume_price_trend = volume_price_trend_numba
negative_volume_index = negative_volume_index_numba
volume_weighted_average_price = volume_weighted_average_price_numba
volume_weighted_exponential_moving_average = volume_weighted_exponential_moving_average_numba


# --- Rust backend dispatch (transparent acceleration) ---
