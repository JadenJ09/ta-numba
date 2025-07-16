"""
Streaming volatility indicators for real-time trading systems.
"""

from collections import deque
from typing import Optional, Union

import numpy as np
from numba import njit

from .base import (
    StreamingIndicator,
    StreamingIndicatorMultiple,
    _streaming_ema_update,
    _streaming_sma,
    _streaming_stddev,
    _streaming_true_range,
)


class ATRStreaming(StreamingIndicator):
    """
    Streaming Average True Range (ATR).

    Optimized for real-time performance with O(1) updates.
    Uses Wilder's smoothing method.
    """

    def __init__(self, window: int = 14):
        super().__init__(window)
        self.alpha = 1.0 / window  # Wilder's smoothing factor
        self.prev_close = np.nan

    def update(self, high: float, low: float, close: float) -> float:
        """Update ATR with new HLC values."""
        self._update_count += 1

        # Calculate True Range directly without function call overhead
        if np.isnan(self.prev_close):
            tr = high - low
        else:
            tr1 = high - low
            tr2 = abs(high - self.prev_close)
            tr3 = abs(low - self.prev_close)
            tr = max(tr1, tr2, tr3)

        # Update ATR using exponential smoothing (Wilder's method)
        if np.isnan(self._current_value):
            self._current_value = tr
        else:
            self._current_value = (
                1 - self.alpha
            ) * self._current_value + self.alpha * tr

        # ATR is ready after enough updates
        if self._update_count >= self.window:
            self._is_ready = True

        self.prev_close = close
        return self._current_value


class BBandsStreaming(StreamingIndicatorMultiple):
    """
    Streaming Bollinger Bands.

    Returns: {
        'upper': Upper band,
        'middle': Middle band (SMA),
        'lower': Lower band
    }
    """

    def __init__(self, window: int = 20, std_dev: float = 2.0):
        super().__init__(window)
        self.std_dev = std_dev

        # Initialize current values
        self._current_values = {"upper": np.nan, "middle": np.nan, "lower": np.nan}

    def update(self, value: float) -> dict:
        """Update Bollinger Bands with new value."""
        self._update_count += 1
        self.buffer.append(value)

        # Calculate Bollinger Bands when we have enough data
        if len(self.buffer) >= self.window:
            buffer_array = self.get_buffer_array()

            # Direct calculations without function call overhead
            sma = np.mean(buffer_array)
            std = np.std(buffer_array)

            # Calculate bands
            self._current_values["middle"] = sma
            self._current_values["upper"] = sma + (self.std_dev * std)
            self._current_values["lower"] = sma - (self.std_dev * std)

            self._is_ready = True

        return self._current_values.copy()

    @property
    def current_value(self) -> float:
        """Get current middle band (SMA) value."""
        return self._current_values["middle"]


class KeltnerChannelStreaming(StreamingIndicatorMultiple):
    """
    Streaming Keltner Channel.

    Returns: {
        'upper': Upper channel,
        'middle': Middle line (EMA),
        'lower': Lower channel
    }
    """

    def __init__(self, window: int = 20, atr_period: int = 10, multiplier: float = 2.0):
        super().__init__(window)
        self.multiplier = multiplier

        # Create component indicators
        self.ema = EMAStreaming(window)
        self.atr = ATRStreaming(atr_period)

        # Initialize current values
        self._current_values = {"upper": np.nan, "middle": np.nan, "lower": np.nan}

    def update(self, high: float, low: float, close: float) -> dict:
        """Update Keltner Channel with new HLC values."""
        self._update_count += 1

        # Update component indicators
        ema_value = self.ema.update(close)
        atr_value = self.atr.update(high, low, close)

        # Calculate channels when both indicators are ready
        if self.ema.is_ready and self.atr.is_ready:
            self._current_values["middle"] = ema_value
            self._current_values["upper"] = ema_value + (self.multiplier * atr_value)
            self._current_values["lower"] = ema_value - (self.multiplier * atr_value)

            self._is_ready = True

        return self._current_values.copy()

    @property
    def current_value(self) -> float:
        """Get current middle line (EMA) value."""
        return self._current_values["middle"]


class DonchianChannelStreaming(StreamingIndicatorMultiple):
    """
    Streaming Donchian Channel.

    Returns: {
        'upper': Highest high over period,
        'middle': Average of upper and lower,
        'lower': Lowest low over period
    }
    """

    def __init__(self, window: int = 20):
        super().__init__(window)

        # Buffers for high/low tracking
        self.high_buffer = deque(maxlen=window)
        self.low_buffer = deque(maxlen=window)

        # Initialize current values
        self._current_values = {"upper": np.nan, "middle": np.nan, "lower": np.nan}

    def update(self, high: float, low: float) -> dict:
        """Update Donchian Channel with new HL values."""
        self._update_count += 1

        # Add to buffers
        self.high_buffer.append(high)
        self.low_buffer.append(low)

        # Calculate channels when we have enough data
        if len(self.high_buffer) >= self.window:
            upper = max(self.high_buffer)
            lower = min(self.low_buffer)

            self._current_values["upper"] = upper
            self._current_values["lower"] = lower
            self._current_values["middle"] = (upper + lower) / 2.0

            self._is_ready = True

        return self._current_values.copy()

    @property
    def current_value(self) -> float:
        """Get current middle line value."""
        return self._current_values["middle"]


class StandardDeviationStreaming(StreamingIndicator):
    """
    Streaming Standard Deviation.

    Rolling standard deviation over specified window.
    """

    def __init__(self, window: int = 20):
        super().__init__(window)

    def update(self, value: float) -> float:
        """Update Standard Deviation with new value."""
        self._update_count += 1
        self.buffer.append(value)

        # Calculate standard deviation when we have enough data
        if len(self.buffer) >= self.window:
            # Direct calculation without function call overhead
            self._current_value = np.std(self.get_buffer_array())
            self._is_ready = True

        return self._current_value


class VarianceStreaming(StreamingIndicator):
    """
    Streaming Variance.

    Rolling variance over specified window.
    """

    def __init__(self, window: int = 20):
        super().__init__(window)

    def update(self, value: float) -> float:
        """Update Variance with new value."""
        self._update_count += 1
        self.buffer.append(value)

        # Calculate variance when we have enough data
        if len(self.buffer) >= self.window:
            # Direct calculation without function call overhead
            self._current_value = np.var(self.get_buffer_array())
            self._is_ready = True

        return self._current_value


class RangeStreaming(StreamingIndicator):
    """
    Streaming Range (High - Low).

    Simple price range over specified window.
    """

    def __init__(self, window: int = 20):
        super().__init__(window)
        self.high_buffer = deque(maxlen=window)
        self.low_buffer = deque(maxlen=window)

    def update(self, high: float, low: float) -> float:
        """Update Range with new HL values."""
        self._update_count += 1

        # Add to buffers
        self.high_buffer.append(high)
        self.low_buffer.append(low)

        # Calculate range when we have enough data
        if len(self.high_buffer) >= self.window:
            self._current_value = max(self.high_buffer) - min(self.low_buffer)
            self._is_ready = True

        return self._current_value


class HistoricalVolatilityStreaming(StreamingIndicator):
    """
    Streaming Historical Volatility.

    Calculates volatility based on log returns.
    """

    def __init__(self, window: int = 20, annualize: bool = True):
        super().__init__(window)
        self.annualize = annualize
        self.prev_value = np.nan
        self.returns_buffer = deque(maxlen=window)

    def update(self, value: float) -> float:
        """Update Historical Volatility with new value."""
        self._update_count += 1

        # Calculate log return
        if not np.isnan(self.prev_value) and self.prev_value > 0 and value > 0:
            log_return = np.log(value / self.prev_value)
            self.returns_buffer.append(log_return)

            # Calculate volatility when we have enough data
            if len(self.returns_buffer) >= self.window:
                returns_array = np.array(self.returns_buffer)
                volatility = np.std(returns_array, ddof=1)

                # Annualize if requested (assuming daily data)
                if self.annualize:
                    volatility *= np.sqrt(252)

                self._current_value = volatility
                self._is_ready = True

        self.prev_value = value
        return self._current_value


class UlcerIndexStreaming(StreamingIndicator):
    """
    Streaming Ulcer Index.

    Measures downside volatility.
    """

    def __init__(self, window: int = 14):
        super().__init__(window)
        self.close_buffer = deque(maxlen=window)

    def update(self, value: float) -> float:
        """Update Ulcer Index with new value."""
        self._update_count += 1

        # Store close price
        self.close_buffer.append(value)

        # Calculate Ulcer Index when we have enough data
        if len(self.close_buffer) >= self.window:
            close_array = np.array(self.close_buffer)

            # Calculate percentage drawdowns
            pct_drawdown_sq = np.zeros(len(close_array))

            for i in range(1, len(close_array)):
                max_close = np.max(close_array[: i + 1])
                if max_close > 0:
                    pct_drawdown = ((close_array[i] - max_close) / max_close) * 100.0
                    pct_drawdown_sq[i] = pct_drawdown**2

            # Calculate Ulcer Index
            self._current_value = np.sqrt(np.mean(pct_drawdown_sq))
            self._is_ready = True

        return self._current_value


# Import EMAStreaming here to avoid circular imports
from .trend import EMAStreaming
