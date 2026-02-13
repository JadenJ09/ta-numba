# src/ta_numba/trend.py

import numpy as np
from numba import njit
from numpy.lib.stride_tricks import as_strided

# Import helper functions from the same package
from .helpers import (
    _ema_numba_adjusted,
    _ema_numba_unadjusted,
    _sma_numba,
    _true_range_numba,
    _wilders_ema_adaptive,
)

# ==============================================================================
# Trend Indicator Functions
# ==============================================================================


def sma_numba_vectorized(data: np.ndarray, n: int = 20) -> np.ndarray:
    # Create a new array with overlapping windows in each row
    s = data.strides[0]
    shape = (len(data) - n + 1, n)
    strides = (s, s)
    windows = as_strided(data, shape=shape, strides=strides)
    
    # Calculate the mean across each row (each window)
    means = np.mean(windows, axis=1)
    
    # Pad with NaNs to match original array size
    result = np.full_like(data, np.nan)
    result[n-1:] = means
    return result

@njit(fastmath=True)
def sma_numba(data: np.ndarray, n: int = 20, min_periods: int = 1) -> np.ndarray:
    """
    Calculates the Simple Moving Average (SMA).
    This is a wrapper for the helper function.
    Matches ta.trend.SMAIndicator
    """
    sma = np.full_like(data, np.nan)
    for i in range(len(data)):
        # Use expanding window until we have n periods, then use rolling window
        start_idx = max(0, i - n + 1)
        window_size = i - start_idx + 1
        if window_size >= min_periods:
            sma[i] = np.mean(data[start_idx:i+1])
    return sma

# Clean public API aliases
sma_indicator = sma_numba

@njit(fastmath=True)
def ema_numba(data: np.ndarray, n: int = 20, adjusted: bool = True) -> np.ndarray:
    """
    Calculates the Exponential Moving Average (EMA) with adjustable 'adjusted' parameter.
    If adjusted=True, uses adjust=True logic (like pandas.ewm(adjust=True)).
    If adjusted=False, uses adjust=False logic (like pandas.ewm(adjust=False)).
    """
    if adjusted:
        return _ema_numba_adjusted(data, n)
    else:
        return _ema_numba_unadjusted(data, n)

@njit(fastmath=True)
def weighted_moving_average(data: np.ndarray, n: int = 20) -> np.ndarray:
    wma = np.full_like(data, np.nan)
    weights = np.arange(1, n + 1, dtype=np.float64)
    sum_weights = np.sum(weights)
    for i in range(n - 1, len(data)):
        window = data[i-n+1:i+1]
        wma[i] = np.sum(weights * window) / sum_weights
    return wma

@njit(fastmath=True)
def macd_numba(close: np.ndarray, n_fast: int = 12, n_slow: int = 26, n_signal: int = 9, adjusted: bool = False) -> (np.ndarray, np.ndarray, np.ndarray):
    ema_fast = ema_numba(close, n_fast, adjusted=adjusted)
    ema_slow = ema_numba(close, n_slow, adjusted=adjusted)
    macd_line = ema_fast - ema_slow
    signal_line = ema_numba(macd_line, n_signal, adjusted=adjusted)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

@njit
def adx_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14):
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.zeros_like(up_move)
    minus_dm = np.zeros_like(down_move)
    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move[(up_move > down_move) & (up_move > 0)]
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move[(down_move > up_move) & (down_move > 0)]
    
    tr = _true_range_numba(high, low, close)
    
    atr = _wilders_ema_adaptive(tr, n)
    plus_di_smooth = _wilders_ema_adaptive(plus_dm, n)
    minus_di_smooth = _wilders_ema_adaptive(minus_dm, n)
    
    # Pad smoothed values to align with close
    plus_di_smooth_padded = np.full_like(close, np.nan)
    minus_di_smooth_padded = np.full_like(close, np.nan)
    plus_di_smooth_padded[1:] = plus_di_smooth
    minus_di_smooth_padded[1:] = minus_di_smooth
    
    plus_di = 100 * (plus_di_smooth_padded / atr)
    minus_di = 100 * (minus_di_smooth_padded / atr)
    
    dx_val = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di))
    adx_val = _wilders_ema_adaptive(dx_val, n)
    return adx_val, plus_di, minus_di

@njit(fastmath=True)
def vortex_indicator_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> (np.ndarray, np.ndarray):
    tr = _true_range_numba(high, low, close)
    vm_plus = np.zeros_like(high)
    vm_minus = np.zeros_like(high)
    vm_plus[1:] = np.abs(high[1:] - low[:-1])
    vm_minus[1:] = np.abs(low[1:] - high[:-1])
    
    sum_tr = _sma_numba(tr, n) * n
    sum_vm_plus = _sma_numba(vm_plus, n) * n
    sum_vm_minus = _sma_numba(vm_minus, n) * n

    vi_plus = sum_vm_plus / sum_tr
    vi_minus = sum_vm_minus / sum_tr
    return vi_plus, vi_minus

@njit
def trix_numba(close: np.ndarray, n: int = 14) -> np.ndarray:
    """TRIX implementation using triple smoothed EMA and percentage rate of change."""
    # Use the corrected EMA helper function
    ema1 = _ema_numba_adjusted(close, n)
    ema2 = _ema_numba_adjusted(ema1, n)
    ema3 = _ema_numba_adjusted(ema2, n)

    trix_val = np.full_like(close, np.nan)
    
    # Calculate percentage change - only for valid EMA3 values
    for i in range(1, len(ema3)):
        if not np.isnan(ema3[i]) and not np.isnan(ema3[i-1]) and ema3[i-1] != 0:
            trix_val[i] = 100 * (ema3[i] - ema3[i-1]) / ema3[i-1]
    
    return trix_val

@njit
def mass_index_numba(high: np.ndarray, low: np.ndarray, n_ema: int = 9, n_sum: int = 25) -> np.ndarray:
    price_range = high - low
    ema1 = _ema_numba_unadjusted(price_range, n_ema)
    ema2 = _ema_numba_unadjusted(ema1, n_ema)
    ratio = ema1 / ema2
    mi = np.full_like(high, np.nan)
    for i in range(n_sum - 1, len(ratio)):
        if not np.isnan(ratio[i-n_sum+1:i+1]).any():
             mi[i] = np.sum(ratio[i-n_sum+1:i+1])
    return mi

@njit(fastmath=True)
def cci_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 20, c: float = 0.015) -> np.ndarray:
    tp = (high + low + close) / 3.0
    cci_val = np.full_like(close, np.nan)
    for i in range(n - 1, len(close)):
        window = tp[i-n+1:i+1]
        sma_tp = np.mean(window)
        mad = np.mean(np.abs(window - sma_tp))
        if mad == 0:
            cci_val[i] = 0.0
        else:
            cci_val[i] = (tp[i] - sma_tp) / (c * mad)
    return cci_val

@njit(fastmath=True)
def dpo_numba(close: np.ndarray, n: int = 20) -> np.ndarray:
    """DPO: Detrended Price Oscillator to match ta library exactly."""
    dpo_val = np.full_like(close, np.nan)
    displacement = n // 2 + 1
    sma = _sma_numba(close, n)
    
    # DPO[i] = Close[i - displacement] - SMA[i]
    # Output at index i (not i - displacement)
    for i in range(displacement, len(close)):
        if not np.isnan(sma[i]):
            dpo_val[i] = close[i - displacement] - sma[i]
    
    return dpo_val

@njit(fastmath=True)
def kst_numba(close: np.ndarray, r1=10, r2=15, r3=20, r4=30, s1=10, s2=10, s3=10, s4=15, n_sig=9) -> (np.ndarray, np.ndarray):
    """
    Calculates the KST Oscillator without helpers, matching ta library logic.
    """
    # --- Inlined ROC Calculation (Corrected: returns raw ratio) ---
    def _roc_local(data: np.ndarray, n: int) -> np.ndarray:
        roc_val = np.full_like(data, np.nan)
        # Check to prevent index error on small arrays
        if n < len(data):
            # Calculate raw ratio, not percentage
            roc_val[n:] = (data[n:] - data[:-n]) / data[:-n]
        return roc_val

    roc1 = _roc_local(close, r1)
    roc2 = _roc_local(close, r2)
    roc3 = _roc_local(close, r3)
    roc4 = _roc_local(close, r4)

    # --- Inlined SMA Calculation ---
    def _sma_local(data: np.ndarray, n: int) -> np.ndarray:
        sma = np.full_like(data, np.nan)
        for i in range(n - 1, len(data)):
            window_slice = data[i-n+1:i+1]
            if not np.isnan(window_slice).any():
                sma[i] = np.mean(window_slice)
        return sma

    # --- KST Main Logic ---
    rcma1 = _sma_local(roc1, s1)
    rcma2 = _sma_local(roc2, s2)
    rcma3 = _sma_local(roc3, s3)
    rcma4 = _sma_local(roc4, s4)

    # Handle NaNs by setting them to 0 for the weighted sum
    rcma1_clean = np.nan_to_num(rcma1, nan=0.0)
    rcma2_clean = np.nan_to_num(rcma2, nan=0.0)
    rcma3_clean = np.nan_to_num(rcma3, nan=0.0)
    rcma4_clean = np.nan_to_num(rcma4, nan=0.0)

    # Weighted sum of SMAs, then multiply by 100 at the end to match 'ta'
    kst_line = ((rcma1_clean * 1) + (rcma2_clean * 2) + (rcma3_clean * 3) + (rcma4_clean * 4)) * 100.0

    # Restore NaNs where all components were NaN to ensure correctness
    for i in range(len(kst_line)):
        if np.isnan(rcma1[i]) and np.isnan(rcma2[i]) and np.isnan(rcma3[i]) and np.isnan(rcma4[i]):
            kst_line[i] = np.nan

    signal_line = _sma_local(kst_line, n_sig)
    
    return kst_line, signal_line

@njit(fastmath=True)
def ichimoku_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, n1=9, n2=26, n3=52):
    """
    Ichimoku Cloud calculation matching ta library exactly.
    Returns: tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span
    """
    n = len(high)
    tenkan_sen = np.full(n, np.nan)
    kijun_sen = np.full(n, np.nan)
    senkou_span_a = np.full(n, np.nan)
    senkou_span_b = np.full(n, np.nan)
    chikou_span = np.full(n, np.nan)
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    for i in range(n1 - 1, n):
        window_high = high[i-n1+1:i+1]
        window_low = low[i-n1+1:i+1]
        tenkan_sen[i] = (np.max(window_high) + np.min(window_low)) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    for i in range(n2 - 1, n):
        window_high = high[i-n2+1:i+1]
        window_low = low[i-n2+1:i+1]
        kijun_sen[i] = (np.max(window_high) + np.min(window_low)) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    # TA library does NOT shift by default (visual=False)
    for i in range(n):
        if not np.isnan(tenkan_sen[i]) and not np.isnan(kijun_sen[i]):
            senkou_span_a[i] = (tenkan_sen[i] + kijun_sen[i]) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    # TA library does NOT shift by default (visual=False)
    for i in range(n3 - 1, n):
        window_high = high[i-n3+1:i+1]
        window_low = low[i-n3+1:i+1]
        senkou_span_b[i] = (np.max(window_high) + np.min(window_low)) / 2.0
    
    # Chikou Span (Lagging Span): Close shifted back by n2 periods
    for i in range(n):
        if i >= n2:
            chikou_span[i - n2] = close[i]
    
    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span


@njit(fastmath=True)
def parabolic_sar_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, af_start=0.02, af_inc=0.02, af_max=0.2) -> np.ndarray:
    """
    Parabolic SAR implementation matching the TA library exactly.
    
    Based on the original TA library implementation:
    - Initialize with close values (psar = close.copy())
    - Start with up_trend = True
    - Begin calculations from index 2
    - Use proper SAR formula throughout
    """
    sar = close.copy()  # Initialize with close values like TA library
    if len(high) < 3: return sar
    
    # Initialize like TA library
    up_trend = True
    acceleration_factor = af_start
    up_trend_high = high[0]
    down_trend_low = low[0]
    
    # Start calculations from index 2 (like TA library)
    for i in range(2, len(high)):
        reversal = False
        max_high = high[i]
        min_low = low[i]
        
        if up_trend:
            sar[i] = sar[i-1] + acceleration_factor * (up_trend_high - sar[i-1])
            
            if min_low < sar[i]:
                reversal = True
                sar[i] = up_trend_high
                down_trend_low = min_low
                acceleration_factor = af_start
            else:
                if max_high > up_trend_high:
                    up_trend_high = max_high
                    acceleration_factor = min(acceleration_factor + af_inc, af_max)
                
                # Apply SAR constraints for uptrend
                low1 = low[i-1]
                low2 = low[i-2]
                if low2 < sar[i]:
                    sar[i] = low2
                elif low1 < sar[i]:
                    sar[i] = low1
        else:
            sar[i] = sar[i-1] - acceleration_factor * (sar[i-1] - down_trend_low)
            
            if max_high > sar[i]:
                reversal = True
                sar[i] = down_trend_low
                up_trend_high = max_high
                acceleration_factor = af_start
            else:
                if min_low < down_trend_low:
                    down_trend_low = min_low
                    acceleration_factor = min(acceleration_factor + af_inc, af_max)
                
                # Apply SAR constraints for downtrend
                high1 = high[i-1]
                high2 = high[i-2]
                if high2 > sar[i]:
                    sar[i] = high2
                elif high1 > sar[i]:
                    sar[i] = high1
        
        up_trend = up_trend != reversal  # XOR logic
        
    return sar

@njit
def schaff_trend_cycle_numba(close: np.ndarray, n_fast=23, n_slow=50, n_stoch=10, n_smooth=3) -> np.ndarray:
    """
    Calculates the Schaff Trend Cycle (STC).
    Matches ta.trend.STCIndicator
    """
    macd_line, _, _ = macd_numba(close, n_fast, n_slow, 9) # MACD signal window is not used here
    
    # Calculate %K of the MACD line
    stoch_k_of_macd = np.full_like(macd_line, np.nan)
    for i in range(n_stoch - 1, len(macd_line)):
        window = macd_line[i-n_stoch+1:i+1]
        if np.isnan(window).all(): continue
        
        low_macd = np.nanmin(window)
        high_macd = np.nanmax(window)
        
        if high_macd > low_macd:
            stoch_k_of_macd[i] = ((macd_line[i] - low_macd) / (high_macd - low_macd)) * 100
        else:
            stoch_k_of_macd[i] = 0.0
            
    # First smoothing of %K -> %D
    stoch_d = _ema_numba_adjusted(stoch_k_of_macd, n_smooth)
    
    # Second smoothing of %D -> STC
    stoch_d_k = np.full_like(stoch_d, np.nan)
    for i in range(n_stoch - 1, len(stoch_d)):
        window = stoch_d[i-n_stoch+1:i+1]
        if np.isnan(window).all(): continue
        
        low_stoch_d = np.nanmin(window)
        high_stoch_d = np.nanmax(window)
        
        if high_stoch_d > low_stoch_d:
            stoch_d_k[i] = ((stoch_d[i] - low_stoch_d) / (high_stoch_d - low_stoch_d)) * 100
        else:
            stoch_d_k[i] = 0.0
            
    stc = _ema_numba_adjusted(stoch_d_k, n_smooth)
    
    return stc

@njit(fastmath=True)
def aroon_numba(high: np.ndarray, low: np.ndarray, n: int = 25):
    """Aroon Up/Down: (n - periods_since_high/low) / n * 100."""
    aroon_up = np.full_like(high, np.nan)
    aroon_down = np.full_like(high, np.nan)
    
    for i in range(n, len(high)):
        window_high = high[i-n:i+1]  # n+1 elements
        window_low = low[i-n:i+1]    # n+1 elements
        
        # Find the index of the maximum/minimum in the window
        max_idx = np.argmax(window_high)
        min_idx = np.argmin(window_low)
        
        # Calculate periods since high/low (0-based indexing)
        periods_since_high = len(window_high) - 1 - max_idx
        periods_since_low = len(window_low) - 1 - min_idx
        
        # Aroon formula: (n - periods_since) / n * 100
        aroon_up[i] = (n - periods_since_high) / n * 100.0
        aroon_down[i] = (n - periods_since_low) / n * 100.0
    
    return aroon_up, aroon_down


# ==============================================================================
# Clean Public API Aliases
# ==============================================================================

sma = sma_numba
ema = ema_numba
wma = weighted_moving_average
macd = macd_numba
adx = adx_numba
vortex_indicator = vortex_indicator_numba
trix = trix_numba
mass_index = mass_index_numba
cci = cci_numba
dpo = dpo_numba
kst = kst_numba
ichimoku = ichimoku_numba
parabolic_sar = parabolic_sar_numba
schaff_trend_cycle = schaff_trend_cycle_numba
aroon = aroon_numba


# --- Rust backend dispatch (transparent acceleration) ---
