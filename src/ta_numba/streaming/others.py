"""
Streaming return indicators for real-time trading systems.
"""

from collections import deque
from typing import Optional, Union

import numpy as np
from numba import njit

from .base import StreamingIndicator, StreamingIndicatorMultiple


class DailyReturnStreaming(StreamingIndicator):
    """
    Streaming Daily Return.

    Percentage change from previous close.
    """

    def __init__(self):
        super().__init__(1)  # No fixed window
        self.prev_close = np.nan

    def update(self, close: float) -> float:
        """Update Daily Return with new close value."""
        self._update_count += 1

        if self._update_count == 1:
            # First update - store close
            self.prev_close = close
            return self._current_value

        # Calculate daily return
        if self.prev_close != 0:
            self._current_value = (close - self.prev_close) / self.prev_close * 100.0
            self._is_ready = True

        self.prev_close = close
        return self._current_value


class DailyLogReturnStreaming(StreamingIndicator):
    """
    Streaming Daily Log Return.

    Logarithmic return from previous close.
    """

    def __init__(self):
        super().__init__(1)  # No fixed window
        self.prev_close = np.nan

    def update(self, close: float) -> float:
        """Update Daily Log Return with new close value."""
        self._update_count += 1

        if self._update_count == 1:
            # First update - store close
            self.prev_close = close
            return self._current_value

        # Calculate daily log return
        if self.prev_close > 0 and close > 0:
            self._current_value = np.log(close / self.prev_close) * 100.0
            self._is_ready = True

        self.prev_close = close
        return self._current_value


class CumulativeReturnStreaming(StreamingIndicator):
    """
    Streaming Cumulative Return.

    Total return from initial price.
    """

    def __init__(self):
        super().__init__(1)  # No fixed window
        self.initial_price = np.nan

    def update(self, close: float) -> float:
        """Update Cumulative Return with new close value."""
        self._update_count += 1

        if self._update_count == 1:
            # First update - store initial price
            self.initial_price = close
            self._current_value = 0.0
            self._is_ready = True
        else:
            # Calculate cumulative return
            if self.initial_price != 0:
                self._current_value = ((close / self.initial_price) - 1) * 100.0

        return self._current_value


class CompoundLogReturnStreaming(StreamingIndicator):
    """
    Streaming Compound Log Return.

    Cumulative logarithmic return.
    """

    def __init__(self):
        super().__init__(1)  # No fixed window
        self.cumulative_log_return = 0.0
        self.prev_close = np.nan

    def update(self, close: float) -> float:
        """Update Compound Log Return with new close value."""
        self._update_count += 1

        if self._update_count == 1:
            # First update - store close
            self.prev_close = close
            self._current_value = 0.0
            self._is_ready = True
        else:
            # Calculate log return and add to cumulative
            if self.prev_close > 0 and close > 0:
                log_return = np.log(close / self.prev_close)
                self.cumulative_log_return += log_return

                # Convert to percentage
                self._current_value = (np.exp(self.cumulative_log_return) - 1) * 100.0

        self.prev_close = close
        return self._current_value


class RollingReturnStreaming(StreamingIndicator):
    """
    Streaming Rolling Return.

    Return over a specified window period.
    """

    def __init__(self, window: int = 20):
        super().__init__(window)
        self.close_buffer = deque(maxlen=window)

    def update(self, close: float) -> float:
        """Update Rolling Return with new close value."""
        self._update_count += 1

        # Store close price
        self.close_buffer.append(close)

        # Calculate rolling return when we have enough data
        if len(self.close_buffer) >= self.window:
            start_price = self.close_buffer[0]
            end_price = self.close_buffer[-1]

            if start_price != 0:
                self._current_value = (end_price - start_price) / start_price * 100.0
            else:
                self._current_value = 0.0

            self._is_ready = True

        return self._current_value


class VolatilityStreaming(StreamingIndicator):
    """
    Streaming Volatility.

    Rolling standard deviation of returns.
    """

    def __init__(self, window: int = 20, annualize: bool = True):
        super().__init__(window)
        self.annualize = annualize
        self.prev_close = np.nan
        self.returns_buffer = deque(maxlen=window)

    def update(self, close: float) -> float:
        """Update Volatility with new close value."""
        self._update_count += 1

        if self._update_count == 1:
            # First update - store close
            self.prev_close = close
            return self._current_value

        # Calculate log return
        if self.prev_close > 0 and close > 0:
            log_return = np.log(close / self.prev_close)
            self.returns_buffer.append(log_return)

            # Calculate volatility when we have enough data
            if len(self.returns_buffer) >= self.window:
                returns_array = np.array(self.returns_buffer)
                volatility = np.std(returns_array, ddof=1)

                # Annualize if requested (assuming daily data)
                if self.annualize:
                    volatility *= np.sqrt(252)

                self._current_value = volatility * 100.0  # Convert to percentage
                self._is_ready = True

        self.prev_close = close
        return self._current_value


class SharpeRatioStreaming(StreamingIndicator):
    """
    Streaming Sharpe Ratio.

    Risk-adjusted return metric.
    """

    def __init__(self, window: int = 252, risk_free_rate: float = 0.02, annualization_factor: float = 252.0):
        super().__init__(window)
        self.risk_free_rate = risk_free_rate
        self.annualization_factor = annualization_factor
        self.prev_close = np.nan
        self.returns_buffer = deque(maxlen=window)

    def update(self, close: float) -> float:
        """Update Sharpe Ratio with new close value."""
        self._update_count += 1

        if self._update_count == 1:
            # First update - store close
            self.prev_close = close
            return self._current_value

        # Calculate log return
        if self.prev_close > 0 and close > 0:
            log_return = np.log(close / self.prev_close)
            self.returns_buffer.append(log_return)

            # Calculate Sharpe ratio when we have enough data
            if len(self.returns_buffer) >= self.window:
                returns_array = np.array(self.returns_buffer)

                # Calculate annualized return and volatility
                avg_return = np.mean(returns_array) * self.annualization_factor
                volatility = np.std(returns_array, ddof=1) * np.sqrt(self.annualization_factor)

                # Calculate Sharpe ratio
                if volatility > 0:
                    self._current_value = (
                        avg_return - self.risk_free_rate
                    ) / volatility
                else:
                    self._current_value = 0.0

                self._is_ready = True

        self.prev_close = close
        return self._current_value


class MaxDrawdownStreaming(StreamingIndicator):
    """
    Streaming Maximum Drawdown.

    Largest peak-to-trough decline.
    """

    def __init__(self, window: int = 252):
        super().__init__(window)
        self.close_buffer = deque(maxlen=window)

    def update(self, close: float) -> float:
        """Update Maximum Drawdown with new close value."""
        self._update_count += 1

        # Store close price
        self.close_buffer.append(close)

        # Calculate maximum drawdown when we have enough data
        if len(self.close_buffer) >= 2:
            close_array = np.array(self.close_buffer)

            # Calculate running maximum
            running_max = np.maximum.accumulate(close_array)

            # Calculate drawdowns
            drawdowns = (close_array - running_max) / running_max

            # Find maximum drawdown
            self._current_value = np.min(drawdowns) * 100.0  # Convert to percentage
            self._is_ready = True

        return self._current_value


class RollingZScoreStreaming(StreamingIndicator):
    """
    Streaming Rolling Z-Score.

    (x - rolling_mean) / rolling_std with Welford's online algorithm.
    """

    def __init__(self, window: int = 20):
        super().__init__(window)

    def update(self, value: float) -> float:
        """Update Rolling Z-Score with new value."""
        self._update_count += 1
        self.buffer.append(value)

        if len(self.buffer) >= self.window:
            buffer_array = self.get_buffer_array()
            mean = np.mean(buffer_array)
            std = np.std(buffer_array)

            if std != 0:
                self._current_value = (value - mean) / std
            else:
                self._current_value = 0.0

            self._is_ready = True

        return self._current_value


class LinearRegressionSlopeStreaming(StreamingIndicator):
    """
    Streaming Linear Regression Slope.

    Uses running sums for O(1) updates.
    """

    def __init__(self, window: int = 14):
        super().__init__(window)

        # Precompute x-related constants (x = 0, 1, ..., window-1)
        w = window
        self._sum_x = w * (w - 1) / 2.0
        self._sum_x2 = w * (w - 1) * (2 * w - 1) / 6.0
        self._denom = w * self._sum_x2 - self._sum_x * self._sum_x

    def update(self, value: float) -> float:
        """Update Linear Regression Slope with new value."""
        self._update_count += 1
        self.buffer.append(value)

        if len(self.buffer) >= self.window and self._denom != 0:
            buffer_array = self.get_buffer_array()
            sum_y = np.sum(buffer_array)
            sum_xy = 0.0
            for j in range(self.window):
                sum_xy += j * buffer_array[j]

            self._current_value = (
                self.window * sum_xy - self._sum_x * sum_y
            ) / self._denom
            self._is_ready = True

        return self._current_value


class RollingPercentileStreaming(StreamingIndicator):
    """
    Streaming Rolling Percentile.

    Fraction of values in window <= current value.
    """

    def __init__(self, window: int = 120):
        super().__init__(window)

    def update(self, value: float) -> float:
        """Update Rolling Percentile with new value."""
        self._update_count += 1
        self.buffer.append(value)

        if len(self.buffer) >= self.window:
            buffer_array = self.get_buffer_array()
            count = np.sum(buffer_array <= value)
            self._current_value = count / self.window
            self._is_ready = True

        return self._current_value


class CalmarRatioStreaming(StreamingIndicator):
    """
    Streaming Calmar Ratio.

    Annual return divided by maximum drawdown.
    """

    def __init__(self, window: int = 252):
        super().__init__(window)
        self.close_buffer = deque(maxlen=window)

    def update(self, close: float) -> float:
        """Update Calmar Ratio with new close value."""
        self._update_count += 1

        # Store close price
        self.close_buffer.append(close)

        # Calculate Calmar ratio when we have enough data
        if len(self.close_buffer) >= self.window:
            close_array = np.array(self.close_buffer)

            # Calculate annualized return
            total_return = (close_array[-1] / close_array[0]) - 1
            annual_return = total_return * (252 / len(close_array))

            # Calculate maximum drawdown
            running_max = np.maximum.accumulate(close_array)
            drawdowns = (close_array - running_max) / running_max
            max_drawdown = abs(np.min(drawdowns))

            # Calculate Calmar ratio
            if max_drawdown > 0:
                self._current_value = annual_return / max_drawdown
            else:
                self._current_value = 0.0

            self._is_ready = True

        return self._current_value
