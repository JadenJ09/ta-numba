"""
Base streaming indicator class following industrial standards.
"""

from abc import ABC, abstractmethod
from collections import deque
from typing import Optional, Union

import numpy as np
from numba import njit


class StreamingIndicator(ABC):
    """
    Base class for all streaming technical indicators.

    Design principles:
    - O(1) per update performance
    - Minimal memory footprint
    - Thread-safe operations
    - Consistent API across all indicators
    """

    def __init__(self, window: int):
        """
        Initialize streaming indicator.

        Args:
            window: Period for the indicator calculation
        """
        self.window = window
        self.buffer = deque(maxlen=window)
        self._current_value = np.nan
        self._is_ready = False
        self._update_count = 0

    @property
    def current_value(self) -> float:
        """Get current indicator value."""
        return self._current_value

    @property
    def is_ready(self) -> bool:
        """Check if indicator has enough data."""
        return self._is_ready

    @property
    def update_count(self) -> int:
        """Get number of updates processed."""
        return self._update_count

    @abstractmethod
    def update(self, value: float) -> float:
        """
        Update indicator with new value.

        Args:
            value: New price/value to process

        Returns:
            Current indicator value (nan if not ready)
        """
        pass

    def reset(self):
        """Reset indicator to initial state."""
        self.buffer.clear()
        self._current_value = np.nan
        self._is_ready = False
        self._update_count = 0

    def get_buffer_array(self) -> np.ndarray:
        """Get current buffer as numpy array."""
        return np.array(self.buffer, dtype=np.float64)


class StreamingIndicatorMultiple(StreamingIndicator):
    """
    Base class for indicators that return multiple values (e.g., MACD, Bollinger Bands).
    """

    def __init__(self, window: int):
        super().__init__(window)
        self._current_values = {}

    @property
    def current_values(self) -> dict:
        """Get all current indicator values."""
        return self._current_values.copy()

    @abstractmethod
    def update(self, value: float) -> dict:
        """
        Update indicator with new value.

        Args:
            value: New price/value to process

        Returns:
            Dictionary of current indicator values
        """
        pass


# Optimized helper functions for streaming calculations
@njit(fastmath=True)
def _streaming_sma(buffer: np.ndarray) -> float:
    """Fast SMA calculation for streaming."""
    return np.mean(buffer)


@njit(fastmath=True)
def _streaming_ema_update(prev_ema: float, new_value: float, alpha: float) -> float:
    """Fast EMA update for streaming."""
    if np.isnan(prev_ema):
        return new_value
    return alpha * new_value + (1 - alpha) * prev_ema


@njit(fastmath=True)
def _streaming_variance(buffer: np.ndarray) -> float:
    """Fast variance calculation for streaming."""
    mean = np.mean(buffer)
    return np.mean((buffer - mean) ** 2)


@njit(fastmath=True)
def _streaming_stddev(buffer: np.ndarray) -> float:
    """Fast standard deviation calculation for streaming."""
    return np.sqrt(_streaming_variance(buffer))


@njit(fastmath=True)
def _streaming_true_range(
    high: float, low: float, close: float, prev_close: float
) -> float:
    """Fast True Range calculation for streaming."""
    if np.isnan(prev_close):
        return high - low

    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)

    return max(tr1, tr2, tr3)


@njit(fastmath=True)
def _streaming_rsi_update(
    prev_avg_gain: float, prev_avg_loss: float, current_change: float, alpha: float
) -> tuple:
    """Fast RSI update for streaming."""
    if current_change > 0:
        current_gain = current_change
        current_loss = 0.0
    else:
        current_gain = 0.0
        current_loss = -current_change

    # Update exponential moving averages
    if np.isnan(prev_avg_gain):
        avg_gain = current_gain
        avg_loss = current_loss
    else:
        avg_gain = alpha * current_gain + (1 - alpha) * prev_avg_gain
        avg_loss = alpha * current_loss + (1 - alpha) * prev_avg_loss

    # Calculate RSI
    if avg_loss == 0:
        rsi = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

    return rsi, avg_gain, avg_loss
