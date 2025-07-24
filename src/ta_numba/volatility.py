# src/ta_numba/volatility.py

import numpy as np
from numba import njit

# Import helper functions from the same package
from .helpers import _sma_numba, _true_range_numba, _wilders_ema_adaptive

# ==============================================================================
# Volatility Indicator Functions
# ==============================================================================

@njit
def average_true_range_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> np.ndarray:
    """
    Calculates the Average True Range (ATR).
    Matches ta.volatility.AverageTrueRange
    """
    tr = _true_range_numba(high, low, close)
    return _wilders_ema_adaptive(tr, n)

@njit(fastmath=True)
def bollinger_bands_numba(close: np.ndarray, n: int = 20, k: float = 2.0):
    upper = np.full_like(close, np.nan)
    middle = np.full_like(close, np.nan)
    lower = np.full_like(close, np.nan)
    for i in range(n - 1, len(close)):
        window = close[i-n+1:i+1]
        sma = np.mean(window)
        std_dev = np.std(window)
        middle[i], upper[i], lower[i] = sma, sma + (k * std_dev), sma - (k * std_dev)
    return upper, middle, lower

@njit
def keltner_channel_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, n_ema: int = 20, n_atr: int = 10, k: float = 2.0):
    """
    Keltner Channel implementation matching ta library's original_version=True.
    
    For original_version=True, ta library uses:
    - Middle line: SMA(Typical Price, 20) with min_periods=window (so NaN until window reached)
    - High band: SMA((4*H - 2*L + C)/3, 20) with min_periods=0 (starts immediately)
    - Low band: SMA((-2*H + 4*L + C)/3, 20) with min_periods=0 (starts immediately)
    """
    # Calculate typical price for middle line
    typical_price = (high + low + close) / 3.0
    
    # Middle line: SMA of typical price with min_periods=window (NaN until window reached)
    middle_line = _sma_numba(typical_price, n=n_ema, min_periods=n_ema)
    
    # High band: SMA of (4*H - 2*L + C)/3 with min_periods=0
    high_tp = (4.0 * high - 2.0 * low + close) / 3.0
    high_band = _sma_numba(high_tp, n=n_ema, min_periods=0)
    
    # Low band: SMA of (-2*H + 4*L + C)/3 with min_periods=0  
    low_tp = (-2.0 * high + 4.0 * low + close) / 3.0
    low_band = _sma_numba(low_tp, n=n_ema, min_periods=0)
    
    return high_band, middle_line, low_band

@njit(fastmath=True)
def donchian_channel_numba(high: np.ndarray, low: np.ndarray, n: int = 20):
    upper = np.full_like(high, np.nan)
    middle = np.full_like(high, np.nan)
    lower = np.full_like(high, np.nan)
    for i in range(n - 1, len(high)):
        window_high = high[i-n+1:i+1]
        window_low = low[i-n+1:i+1]
        upper_val, lower_val = np.max(window_high), np.min(window_low)
        upper[i], lower[i], middle[i] = upper_val, lower_val, (upper_val + lower_val) / 2.0
    return upper, middle, lower

@njit(fastmath=True)
def ulcer_index_numba(close: np.ndarray, n: int = 14) -> np.ndarray:
    ui = np.full_like(close, np.nan)
    pct_drawdown_sq = np.zeros_like(close)
    for i in range(1, len(close)):
        max_close = np.max(close[max(0, i-n+1):i+1])
        pct_drawdown_sq[i] = (((close[i] - max_close) / max_close) * 100.0)**2

    for i in range(n - 1, len(close)):
        ui[i] = np.sqrt(np.mean(pct_drawdown_sq[i-n+1:i+1]))
    return ui


# ==============================================================================
# Clean Public API Aliases
# ==============================================================================

average_true_range = average_true_range_numba
bollinger_bands = bollinger_bands_numba
keltner_channel = keltner_channel_numba
donchian_channel = donchian_channel_numba
ulcer_index = ulcer_index_numba
