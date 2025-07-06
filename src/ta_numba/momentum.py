# src/ta_numba/momentum.py

import numpy as np
from numba import njit

# Import helper functions from the same package
from .helpers import (
    _sma,
    _sma_numba,
    _ema_numba_adjusted,
    _ema_numba_unadjusted,
    _true_range_numba,
    _ewm_numba
)

# ==============================================================================
# Momentum Indicator Functions
# ==============================================================================

@njit(fastmath=True)
def relative_strength_index_numba(close: np.ndarray, n: int = 14) -> np.ndarray:
    """RSI calculation to match ta library exactly."""
    diff = np.zeros_like(close)
    diff[1:] = close[1:] - close[:-1]
    
    # Split into gains and losses
    up_direction = np.where(diff > 0, diff, 0.0)
    down_direction = np.where(diff < 0, -diff, 0.0)
    
    # Calculate EMA of gains and losses using alpha=1/n (like TA library)
    alpha = 1.0 / n
    emaup = _ewm_numba(up_direction, alpha)
    emadn = _ewm_numba(down_direction, alpha)
    
    # Calculate RSI
    rs = emaup / emadn
    rsi = np.where(emadn == 0, 100.0, 100.0 - (100.0 / (1.0 + rs)))
    
    return rsi

@njit(fastmath=True)
def stochastic_rsi_numba(close: np.ndarray, n: int = 14, k: int = 3, d: int = 3):
    """Stochastic RSI: Apply stochastic oscillator to RSI values."""
    rsi = relative_strength_index_numba(close, n)
    stoch_rsi = np.full_like(rsi, np.nan)
    
    # Apply stochastic oscillator formula to RSI
    # TA library first StochRSI appears at index 26 for window=14
    # This is when we have 14 RSI values to work with (RSI starts at 13)
    start_idx = (n - 1) + (n - 1)  # 2*n-2 = 26 for n=14
    
    for i in range(start_idx, len(rsi)):
        # Get 14 RSI values for the window
        window_start = i - n + 1
        low_rsi = rsi[window_start]
        high_rsi = rsi[window_start]
        
        # Find min and max RSI in the window
        for j in range(window_start, i + 1):
            if not np.isnan(rsi[j]):
                if np.isnan(low_rsi) or rsi[j] < low_rsi:
                    low_rsi = rsi[j]
                if np.isnan(high_rsi) or rsi[j] > high_rsi:
                    high_rsi = rsi[j]
        
        # Calculate StochRSI
        if not np.isnan(rsi[i]) and not np.isnan(low_rsi) and not np.isnan(high_rsi):
            if high_rsi > low_rsi:
                stoch_rsi[i] = (rsi[i] - low_rsi) / (high_rsi - low_rsi)
            else:
                stoch_rsi[i] = 0.0
    
    # Apply smoothing with SMA - TA library uses SMA not EMA
    stoch_k_final = _sma(stoch_rsi, k)
    stoch_d_final = _sma(stoch_k_final, d)
    return stoch_rsi, stoch_k_final, stoch_d_final

@njit
def true_strength_index_numba(close: np.ndarray, r: int = 25, s: int = 13) -> np.ndarray:
    """
    Calculates the True Strength Index (TSI).
    Matches ta.momentum.TSIIndicator
    """
    pc = np.zeros_like(close)
    pc[1:] = close[1:] - close[:-1]
    
    smooth2_pc = _ema_numba_adjusted(_ema_numba_adjusted(pc, r), s)
    smooth2_abspc = _ema_numba_adjusted(_ema_numba_adjusted(np.abs(pc), r), s)
    
    tsi_val = 100 * (smooth2_pc / smooth2_abspc)
    return tsi_val

@njit(fastmath=True)
def ultimate_oscillator_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, n1=7, n2=14, n3=28) -> np.ndarray:
    bp = np.zeros_like(close); bp[1:] = close[1:] - np.minimum(low[1:], close[:-1])
    tr = _true_range_numba(high, low, close)
    sum_bp1 = _sma_numba(bp, window=n1) * n1
    sum_tr1 = _sma_numba(tr, window=n1) * n1
    sum_bp2 = _sma_numba(bp, window=n2) * n2
    sum_tr2 = _sma_numba(tr, window=n2) * n2
    sum_bp3 = _sma_numba(bp, window=n3) * n3
    sum_tr3 = _sma_numba(tr, window=n3) * n3
    avg1, avg2, avg3 = sum_bp1 / sum_tr1, sum_bp2 / sum_tr2, sum_bp3 / sum_tr3
    uo = 100 * ((4 * avg1) + (2 * avg2) + (1 * avg3)) / 7.0
    return uo

@njit(fastmath=True)
def stochastic_oscillator_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14, d: int = 3):
    percent_k = np.full_like(close, np.nan)
    for i in range(n - 1, len(close)):
        window_high, window_low = np.max(high[i-n+1:i+1]), np.min(low[i-n+1:i+1])
        if window_high != window_low:
            percent_k[i] = 100 * (close[i] - window_low) / (window_high - window_low)
        else:
            percent_k[i] = 0.0
    return percent_k, _sma_numba(percent_k, window=d)

@njit(fastmath=True)
def williams_r_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> np.ndarray:
    wr = np.full_like(close, np.nan)
    for i in range(n - 1, len(close)):
        window_high, window_low = np.max(high[i-n+1:i+1]), np.min(low[i-n+1:i+1])
        if window_high != window_low:
            wr[i] = -100 * (window_high - close[i]) / (window_high - window_low)
        else:
            wr[i] = -100.0
    return wr

@njit(fastmath=True)
def awesome_oscillator_numba(high: np.ndarray, low: np.ndarray, n1: int = 5, n2: int = 34) -> np.ndarray:
    midpoint = (high + low) / 2.0
    return _sma_numba(midpoint, window=n1) - _sma_numba(midpoint, window=n2)

@njit(fastmath=True)
def kaufmans_adaptive_moving_average_numba(close: np.ndarray, n: int = 10, n_fast: int = 2, n_slow: int = 30) -> np.ndarray:
    direction = np.abs(close[n:] - close[:-n])
    volatility = np.zeros_like(direction)
    diffs = np.abs(close[1:] - close[:-1])
    for i in range(len(direction)):
        volatility[i] = np.sum(diffs[i:i+n])
    er = direction / volatility
    fast_sc, slow_sc = 2.0 / (n_fast + 1.0), 2.0 / (n_slow + 1.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc)**2
    kama = np.full_like(close, np.nan)
    if len(close) > n:
        kama[n-1] = close[n-1]
        for i in range(n, len(close)):
            kama[i] = kama[i-1] + sc[i-n] * (close[i] - kama[i-1])
    return kama

@njit(fastmath=True)
def rate_of_change_numba(close: np.ndarray, n: int = 12) -> np.ndarray:
    roc = np.full_like(close, np.nan)
    roc[n:] = (close[n:] - close[:-n]) / close[:-n] * 100.0
    return roc

@njit(fastmath=True)
def percentage_price_oscillator_numba(close: np.ndarray, n_fast: int = 12, n_slow: int = 26, n_signal: int = 9):
    ema_fast = _ema_numba_unadjusted(close, n_fast)
    ema_slow = _ema_numba_unadjusted(close, n_slow)
    ppo_line = ((ema_fast - ema_slow) / ema_slow) * 100.0
    signal_line = _ema_numba_unadjusted(ppo_line, n_signal)
    histogram = ppo_line - signal_line
    return ppo_line, signal_line, histogram

@njit(fastmath=True)
def percentage_volume_oscillator_numba(volume: np.ndarray, n_fast: int = 12, n_slow: int = 26, n_signal: int = 9):
    volume_float = volume.astype(np.float64)
    ema_fast, ema_slow = _ema_numba_unadjusted(volume_float, n_fast), _ema_numba_unadjusted(volume_float, n_slow)
    pvo_line = ((ema_fast - ema_slow) / ema_slow) * 100.0
    signal_line = _ema_numba_adjusted(pvo_line, n_signal)
    histogram = pvo_line - signal_line
    return pvo_line, signal_line, histogram


# ==============================================================================
# Clean Public API Aliases
# ==============================================================================

rsi = relative_strength_index_numba
stochrsi = stochastic_rsi_numba
tsi = true_strength_index_numba
ultimate_oscillator = ultimate_oscillator_numba
stoch = stochastic_oscillator_numba
williams_r = williams_r_numba
awesome_oscillator = awesome_oscillator_numba
kama = kaufmans_adaptive_moving_average_numba
roc = rate_of_change_numba
ppo = percentage_price_oscillator_numba
pvo = percentage_volume_oscillator_numba
