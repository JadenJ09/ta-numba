"""
Streaming momentum indicators for real-time trading systems.
"""

from collections import deque
from typing import Optional, Union

import numpy as np
from numba import njit

from .base import StreamingIndicator, StreamingIndicatorMultiple, _streaming_rsi_update


class RSIStreaming(StreamingIndicator):
    """
    Streaming Relative Strength Index (RSI).

    Optimized for real-time performance with O(1) updates.
    Uses exponential smoothing like the original RSI formula.
    """

    def __init__(self, window: int = 14):
        super().__init__(window)
        self.alpha = 1.0 / window
        self.prev_close = np.nan
        self.avg_gain = np.nan
        self.avg_loss = np.nan

    def update(self, value: float) -> float:
        """Update RSI with new value."""
        self._update_count += 1

        if self._update_count == 1:
            # First update - just store the value
            self.prev_close = value
            return self._current_value

        # Calculate price change
        change = value - self.prev_close

        # Direct RSI calculation without function call overhead
        if change > 0:
            current_gain = change
            current_loss = 0.0
        else:
            current_gain = 0.0
            current_loss = -change

        # Update averages
        if np.isnan(self.avg_gain):
            self.avg_gain = current_gain
            self.avg_loss = current_loss
        else:
            self.avg_gain = self.alpha * current_gain + (1 - self.alpha) * self.avg_gain
            self.avg_loss = self.alpha * current_loss + (1 - self.alpha) * self.avg_loss

        # Calculate RSI
        if self.avg_loss == 0:
            self._current_value = 100.0
        else:
            rs = self.avg_gain / self.avg_loss
            self._current_value = 100.0 - (100.0 / (1.0 + rs))

        # RSI is ready after enough updates
        if self._update_count >= self.window:
            self._is_ready = True

        self.prev_close = value
        return self._current_value


class StochasticStreaming(StreamingIndicatorMultiple):
    """
    Streaming Stochastic Oscillator.

    Returns: {
        'percent_k': %K value,
        'percent_d': %D value (smoothed %K)
    }
    """

    def __init__(self, k_period: int = 14, d_period: int = 3):
        super().__init__(k_period)
        self.k_period = k_period
        self.d_period = d_period

        # Buffers for high/low tracking
        self.high_buffer = deque(maxlen=k_period)
        self.low_buffer = deque(maxlen=k_period)

        # SMA for %D calculation
        self.percent_k_buffer = deque(maxlen=d_period)

        # Initialize current values
        self._current_values = {"percent_k": np.nan, "percent_d": np.nan}

    def update(self, high: float, low: float, close: float) -> dict:
        """Update Stochastic with new HLC values."""
        self._update_count += 1

        # Add to buffers
        self.high_buffer.append(high)
        self.low_buffer.append(low)

        # Calculate %K when we have enough data
        if len(self.high_buffer) >= self.k_period:
            highest_high = max(self.high_buffer)
            lowest_low = min(self.low_buffer)

            if highest_high != lowest_low:
                percent_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
            else:
                percent_k = 0.0

            self._current_values["percent_k"] = percent_k
            self.percent_k_buffer.append(percent_k)

            # Calculate %D (smoothed %K)
            if len(self.percent_k_buffer) >= self.d_period:
                percent_d = sum(self.percent_k_buffer) / len(self.percent_k_buffer)
                self._current_values["percent_d"] = percent_d
                self._is_ready = True

        return self._current_values.copy()

    @property
    def current_value(self) -> float:
        """Get current %K value."""
        return self._current_values["percent_k"]


class WilliamsRStreaming(StreamingIndicator):
    """
    Streaming Williams %R.

    Similar to Stochastic but with different scaling.
    """

    def __init__(self, window: int = 14):
        super().__init__(window)
        self.high_buffer = deque(maxlen=window)
        self.low_buffer = deque(maxlen=window)

    def update(self, high: float, low: float, close: float) -> float:
        """Update Williams %R with new HLC values."""
        self._update_count += 1

        # Add to buffers
        self.high_buffer.append(high)
        self.low_buffer.append(low)

        # Calculate Williams %R when we have enough data
        if len(self.high_buffer) >= self.window:
            highest_high = max(self.high_buffer)
            lowest_low = min(self.low_buffer)

            if highest_high != lowest_low:
                self._current_value = (
                    -100 * (highest_high - close) / (highest_high - lowest_low)
                )
            else:
                self._current_value = -100.0

            self._is_ready = True

        return self._current_value


class ROCStreaming(StreamingIndicator):
    """
    Streaming Rate of Change (ROC).

    Calculates percentage change from n periods ago.
    """

    def __init__(self, window: int = 12):
        super().__init__(window)

    def update(self, value: float) -> float:
        """Update ROC with new value."""
        self._update_count += 1
        self.buffer.append(value)

        # Calculate ROC when we have enough data
        if len(self.buffer) >= self.window:
            old_value = self.buffer[0]  # Value from n periods ago
            if old_value != 0:
                self._current_value = (value - old_value) / old_value * 100.0
            else:
                self._current_value = 0.0

            self._is_ready = True

        return self._current_value


class CCIStreaming(StreamingIndicator):
    """
    Streaming Commodity Channel Index (CCI).

    Measures deviation from typical price average.
    """

    def __init__(self, window: int = 20, constant: float = 0.015):
        super().__init__(window)
        self.constant = constant
        self.tp_buffer = deque(maxlen=window)

    def update(self, high: float, low: float, close: float) -> float:
        """Update CCI with new HLC values."""
        self._update_count += 1

        # Calculate typical price
        typical_price = (high + low + close) / 3.0
        self.tp_buffer.append(typical_price)

        # Calculate CCI when we have enough data
        if len(self.tp_buffer) >= self.window:
            tp_array = np.array(self.tp_buffer)
            sma_tp = np.mean(tp_array)
            mad = np.mean(np.abs(tp_array - sma_tp))

            if mad == 0:
                self._current_value = 0.0
            else:
                self._current_value = (typical_price - sma_tp) / (self.constant * mad)

            self._is_ready = True

        return self._current_value


class MomentumStreaming(StreamingIndicator):
    """
    Streaming Momentum indicator.

    Simple price difference from n periods ago.
    """

    def __init__(self, window: int = 10):
        super().__init__(window)

    def update(self, value: float) -> float:
        """Update Momentum with new value."""
        self._update_count += 1
        self.buffer.append(value)

        # Calculate momentum when we have enough data
        if len(self.buffer) >= self.window:
            old_value = self.buffer[0]  # Value from n periods ago
            self._current_value = value - old_value
            self._is_ready = True

        return self._current_value


class UltimateOscillatorStreaming(StreamingIndicatorMultiple):
    """
    Streaming Ultimate Oscillator.

    Uses multiple timeframes (7, 14, 28 periods).
    """

    def __init__(self, period1: int = 7, period2: int = 14, period3: int = 28):
        super().__init__(period3)  # Use longest period as base
        self.period1 = period1
        self.period2 = period2
        self.period3 = period3

        # Buffers for BP and TR
        self.bp_buffer = deque(maxlen=period3)
        self.tr_buffer = deque(maxlen=period3)

        # Previous close for TR calculation
        self.prev_close = np.nan

        # Initialize current values
        self._current_values = {"ultimate_oscillator": np.nan}

    def update(self, high: float, low: float, close: float) -> dict:
        """Update Ultimate Oscillator with new HLC values."""
        self._update_count += 1

        # Calculate buying pressure (BP)
        if not np.isnan(self.prev_close):
            bp = close - min(low, self.prev_close)
            tr = max(
                high - low, abs(high - self.prev_close), abs(low - self.prev_close)
            )
        else:
            bp = close - low
            tr = high - low

        self.bp_buffer.append(bp)
        self.tr_buffer.append(tr)

        # Calculate Ultimate Oscillator when we have enough data
        if len(self.bp_buffer) >= self.period3:
            bp_array = np.array(self.bp_buffer)
            tr_array = np.array(self.tr_buffer)

            # Calculate averages for each period
            avg1 = np.sum(bp_array[-self.period1 :]) / np.sum(tr_array[-self.period1 :])
            avg2 = np.sum(bp_array[-self.period2 :]) / np.sum(tr_array[-self.period2 :])
            avg3 = np.sum(bp_array[-self.period3 :]) / np.sum(tr_array[-self.period3 :])

            # Ultimate Oscillator formula
            uo = 100 * ((4 * avg1) + (2 * avg2) + (1 * avg3)) / 7.0
            self._current_values["ultimate_oscillator"] = uo

            self._is_ready = True

        self.prev_close = close
        return self._current_values.copy()

    @property
    def current_value(self) -> float:
        """Get current Ultimate Oscillator value."""
        return self._current_values["ultimate_oscillator"]


class StochasticRSIStreaming(StreamingIndicatorMultiple):
    """
    Streaming Stochastic RSI.

    Applies stochastic oscillator formula to RSI values.

    Returns: {
        'stochrsi': Raw Stochastic RSI,
        'stochrsi_k': %K (smoothed),
        'stochrsi_d': %D (smoothed %K)
    }
    """

    def __init__(
        self,
        rsi_period: int = 14,
        stoch_period: int = 14,
        k_period: int = 3,
        d_period: int = 3,
    ):
        super().__init__(rsi_period + stoch_period)
        self.rsi_period = rsi_period
        self.stoch_period = stoch_period
        self.k_period = k_period
        self.d_period = d_period

        # RSI streamer
        self.rsi_stream = RSIStreaming(rsi_period)

        # Buffer for RSI values
        self.rsi_buffer = deque(maxlen=stoch_period)

        # SMAs for smoothing
        self.k_sma = SMAStreaming(k_period)
        self.d_sma = SMAStreaming(d_period)

        # Initialize current values
        self._current_values = {
            "stochrsi": np.nan,
            "stochrsi_k": np.nan,
            "stochrsi_d": np.nan,
        }

    def update(self, value: float) -> dict:
        """Update Stochastic RSI with new value."""
        self._update_count += 1

        # Update RSI
        rsi_value = self.rsi_stream.update(value)

        # Store RSI value when ready
        if self.rsi_stream.is_ready:
            self.rsi_buffer.append(rsi_value)

            # Calculate Stochastic RSI when we have enough RSI values
            if len(self.rsi_buffer) >= self.stoch_period:
                rsi_array = np.array(self.rsi_buffer)
                low_rsi = np.min(rsi_array)
                high_rsi = np.max(rsi_array)

                if high_rsi > low_rsi:
                    stoch_rsi = (rsi_value - low_rsi) / (high_rsi - low_rsi)
                else:
                    stoch_rsi = 0.0

                self._current_values["stochrsi"] = stoch_rsi

                # Apply smoothing
                k_value = self.k_sma.update(stoch_rsi)
                d_value = self.d_sma.update(k_value)

                self._current_values["stochrsi_k"] = k_value
                self._current_values["stochrsi_d"] = d_value

                # Ready when all components are ready
                if self.k_sma.is_ready and self.d_sma.is_ready:
                    self._is_ready = True

        return self._current_values.copy()

    @property
    def current_value(self) -> float:
        """Get current Stochastic RSI value."""
        return self._current_values["stochrsi"]


class TSIStreaming(StreamingIndicator):
    """
    Streaming True Strength Index (TSI).

    Double smoothed momentum indicator.
    """

    def __init__(self, first_smooth: int = 25, second_smooth: int = 13):
        super().__init__(first_smooth + second_smooth)
        self.first_smooth = first_smooth
        self.second_smooth = second_smooth

        # Previous close for momentum calculation
        self.prev_close = np.nan

        # EMA streamers for double smoothing
        self.momentum_ema1 = EMAStreaming(first_smooth)
        self.momentum_ema2 = EMAStreaming(second_smooth)
        self.abs_momentum_ema1 = EMAStreaming(first_smooth)
        self.abs_momentum_ema2 = EMAStreaming(second_smooth)

    def update(self, value: float) -> float:
        """Update TSI with new value."""
        self._update_count += 1

        if self._update_count == 1:
            # First update - just store value
            self.prev_close = value
            return self._current_value

        # Calculate momentum
        momentum = value - self.prev_close
        abs_momentum = abs(momentum)

        # Apply double smoothing
        smooth1_momentum = self.momentum_ema1.update(momentum)
        smooth2_momentum = self.momentum_ema2.update(smooth1_momentum)

        smooth1_abs = self.abs_momentum_ema1.update(abs_momentum)
        smooth2_abs = self.abs_momentum_ema2.update(smooth1_abs)

        # Calculate TSI
        if (
            not np.isnan(smooth2_momentum)
            and not np.isnan(smooth2_abs)
            and smooth2_abs != 0
        ):
            self._current_value = 100 * (smooth2_momentum / smooth2_abs)

            # TSI is ready when both double smoothed values are ready
            if (
                self.momentum_ema1.is_ready
                and self.momentum_ema2.is_ready
                and self.abs_momentum_ema1.is_ready
                and self.abs_momentum_ema2.is_ready
            ):
                self._is_ready = True

        self.prev_close = value
        return self._current_value


class AwesomeOscillatorStreaming(StreamingIndicator):
    """
    Streaming Awesome Oscillator.

    Difference between fast and slow SMAs of midpoint price.
    """

    def __init__(self, fast_period: int = 5, slow_period: int = 34):
        super().__init__(slow_period)
        self.fast_period = fast_period
        self.slow_period = slow_period

        # SMA streamers for fast and slow averages
        self.fast_sma = SMAStreaming(fast_period)
        self.slow_sma = SMAStreaming(slow_period)

    def update(self, high: float, low: float) -> float:
        """Update Awesome Oscillator with new HL values."""
        self._update_count += 1

        # Calculate midpoint price
        midpoint = (high + low) / 2.0

        # Update SMAs
        fast_sma = self.fast_sma.update(midpoint)
        slow_sma = self.slow_sma.update(midpoint)

        # Calculate Awesome Oscillator
        if self.fast_sma.is_ready and self.slow_sma.is_ready:
            self._current_value = fast_sma - slow_sma
            self._is_ready = True

        return self._current_value


class KAMAStreaming(StreamingIndicator):
    """
    Streaming Kaufman's Adaptive Moving Average (KAMA).

    Adaptive moving average that adjusts to market volatility.
    """

    def __init__(self, window: int = 10, fast_period: int = 2, slow_period: int = 30):
        super().__init__(window)
        self.fast_period = fast_period
        self.slow_period = slow_period

        # Calculate smoothing constants
        self.fast_sc = 2.0 / (fast_period + 1.0)
        self.slow_sc = 2.0 / (slow_period + 1.0)

        # Buffer for direction calculation
        self.price_buffer = deque(maxlen=window + 1)

        # Previous KAMA value
        self.prev_kama = np.nan

    def update(self, value: float) -> float:
        """Update KAMA with new value."""
        self._update_count += 1

        # Store price in buffer
        self.price_buffer.append(value)

        # Calculate KAMA when we have enough data
        if len(self.price_buffer) >= self.window + 1:
            price_array = np.array(self.price_buffer)

            # Calculate direction and volatility
            direction = abs(price_array[-1] - price_array[0])
            volatility = np.sum(np.abs(np.diff(price_array)))

            # Calculate efficiency ratio
            if volatility > 0:
                er = direction / volatility
            else:
                er = 0.0

            # Calculate smoothing constant
            sc = (er * (self.fast_sc - self.slow_sc) + self.slow_sc) ** 2

            # Update KAMA
            if np.isnan(self.prev_kama):
                self._current_value = value
            else:
                self._current_value = self.prev_kama + sc * (value - self.prev_kama)

            self.prev_kama = self._current_value
            self._is_ready = True

        return self._current_value


class PPOStreaming(StreamingIndicatorMultiple):
    """
    Streaming Percentage Price Oscillator (PPO).

    Returns: {
        'ppo': PPO line,
        'signal': Signal line,
        'histogram': PPO histogram
    }
    """

    def __init__(
        self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
    ):
        super().__init__(slow_period)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

        # EMA streamers
        self.fast_ema = EMAStreaming(fast_period)
        self.slow_ema = EMAStreaming(slow_period)
        self.signal_ema = EMAStreaming(signal_period)

        # Initialize current values
        self._current_values = {"ppo": np.nan, "signal": np.nan, "histogram": np.nan}

    def update(self, value: float) -> dict:
        """Update PPO with new value."""
        self._update_count += 1

        # Update EMAs
        fast_ema = self.fast_ema.update(value)
        slow_ema = self.slow_ema.update(value)

        # Calculate PPO line
        if self.fast_ema.is_ready and self.slow_ema.is_ready and slow_ema != 0:
            ppo_line = ((fast_ema - slow_ema) / slow_ema) * 100.0
            self._current_values["ppo"] = ppo_line

            # Update signal line
            signal_line = self.signal_ema.update(ppo_line)
            self._current_values["signal"] = signal_line

            # Calculate histogram
            if self.signal_ema.is_ready:
                self._current_values["histogram"] = ppo_line - signal_line
                self._is_ready = True

        return self._current_values.copy()

    @property
    def current_value(self) -> float:
        """Get current PPO line value."""
        return self._current_values["ppo"]


class PVOStreaming(StreamingIndicatorMultiple):
    """
    Streaming Percentage Volume Oscillator (PVO).

    Same as PPO but applied to volume.

    Returns: {
        'pvo': PVO line,
        'signal': Signal line,
        'histogram': PVO histogram
    }
    """

    def __init__(
        self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
    ):
        super().__init__(slow_period)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

        # EMA streamers
        self.fast_ema = EMAStreaming(fast_period)
        self.slow_ema = EMAStreaming(slow_period)
        self.signal_ema = EMAStreaming(signal_period)

        # Initialize current values
        self._current_values = {"pvo": np.nan, "signal": np.nan, "histogram": np.nan}

    def update(self, volume: float) -> dict:
        """Update PVO with new volume value."""
        self._update_count += 1

        # Update EMAs
        fast_ema = self.fast_ema.update(volume)
        slow_ema = self.slow_ema.update(volume)

        # Calculate PVO line
        if self.fast_ema.is_ready and self.slow_ema.is_ready and slow_ema != 0:
            pvo_line = ((fast_ema - slow_ema) / slow_ema) * 100.0
            self._current_values["pvo"] = pvo_line

            # Update signal line
            signal_line = self.signal_ema.update(pvo_line)
            self._current_values["signal"] = signal_line

            # Calculate histogram
            if self.signal_ema.is_ready:
                self._current_values["histogram"] = pvo_line - signal_line
                self._is_ready = True

        return self._current_values.copy()

    @property
    def current_value(self) -> float:
        """Get current PVO line value."""
        return self._current_values["pvo"]


# Import SMAStreaming and EMAStreaming here to avoid circular imports
from .trend import EMAStreaming, SMAStreaming
