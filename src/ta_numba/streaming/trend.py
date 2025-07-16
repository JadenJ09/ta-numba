"""
Streaming trend indicators for real-time trading systems.
"""

import numpy as np
from numba import njit
from collections import deque
from typing import Optional, Union

from .base import (
    StreamingIndicator, 
    StreamingIndicatorMultiple,
    _streaming_sma,
    _streaming_ema_update
)


class SMAStreaming(StreamingIndicator):
    """
    Streaming Simple Moving Average (SMA).
    
    Optimized for real-time performance with O(1) updates.
    """
    
    def __init__(self, window: int = 20):
        super().__init__(window)
        
    def update(self, value: float) -> float:
        """Update SMA with new value."""
        self._update_count += 1
        self.buffer.append(value)
        
        # Calculate SMA when we have enough data
        if len(self.buffer) >= self.window:
            # Direct calculation without function call overhead
            self._current_value = np.mean(self.get_buffer_array())
            self._is_ready = True
        
        return self._current_value


class EMAStreaming(StreamingIndicator):
    """
    Streaming Exponential Moving Average (EMA).
    
    Optimized for real-time performance with O(1) updates.
    Uses direct calculation without function call overhead.
    """
    
    def __init__(self, window: int = 20):
        super().__init__(window)
        self.alpha = 2.0 / (window + 1.0)
        
    def update(self, value: float) -> float:
        """Update EMA with new value."""
        self._update_count += 1
        
        # Direct EMA calculation - fastest approach
        if np.isnan(self._current_value):
            self._current_value = value
        else:
            self._current_value = self.alpha * value + (1 - self.alpha) * self._current_value
        
        # EMA is ready after first update
        if self._update_count >= 1:
            self._is_ready = True
        
        return self._current_value


class WMAStreaming(StreamingIndicator):
    """
    Streaming Weighted Moving Average (WMA).
    
    Optimized for real-time performance with O(window) updates.
    """
    
    def __init__(self, window: int = 20):
        super().__init__(window)
        self.weights = np.arange(1, window + 1, dtype=np.float64)
        self.sum_weights = np.sum(self.weights)
        
    def update(self, value: float) -> float:
        """Update WMA with new value."""
        self._update_count += 1
        self.buffer.append(value)
        
        # Calculate WMA when we have enough data
        if len(self.buffer) >= self.window:
            buffer_array = self.get_buffer_array()
            # Direct calculation without function call overhead
            self._current_value = np.dot(buffer_array, self.weights) / self.sum_weights
            self._is_ready = True
        
        return self._current_value


class MACDStreaming(StreamingIndicatorMultiple):
    """
    Streaming MACD (Moving Average Convergence Divergence).
    
    Returns: {
        'macd': MACD line,
        'signal': Signal line,
        'histogram': MACD histogram
    }
    """
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        # Use the slower period as the base window
        super().__init__(slow_period)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        
        # Create EMA streamers
        self.ema_fast = EMAStreaming(fast_period)
        self.ema_slow = EMAStreaming(slow_period)
        self.ema_signal = EMAStreaming(signal_period)
        
        # Initialize current values
        self._current_values = {
            'macd': np.nan,
            'signal': np.nan,
            'histogram': np.nan
        }
        
    def update(self, value: float) -> dict:
        """Update MACD with new value."""
        self._update_count += 1
        
        # Update both EMAs
        fast_ema = self.ema_fast.update(value)
        slow_ema = self.ema_slow.update(value)
        
        # Calculate MACD line
        if self.ema_fast.is_ready and self.ema_slow.is_ready:
            macd_line = fast_ema - slow_ema
            self._current_values['macd'] = macd_line
            
            # Update signal line
            signal_line = self.ema_signal.update(macd_line)
            self._current_values['signal'] = signal_line
            
            # Calculate histogram
            if self.ema_signal.is_ready:
                self._current_values['histogram'] = macd_line - signal_line
                self._is_ready = True
        
        return self._current_values.copy()
    
    @property
    def current_value(self) -> float:
        """Get current MACD line value."""
        return self._current_values['macd']


class ADXStreaming(StreamingIndicatorMultiple):
    """
    Streaming Average Directional Index (ADX).
    
    Returns: {
        'adx': ADX value,
        'plus_di': +DI value,
        'minus_di': -DI value
    }
    """
    
    def __init__(self, window: int = 14):
        super().__init__(window)
        self.alpha = 1.0 / window  # Wilder's smoothing
        
        # State variables
        self.prev_high = np.nan
        self.prev_low = np.nan
        self.prev_close = np.nan
        
        # Smoothed values
        self.smoothed_plus_dm = np.nan
        self.smoothed_minus_dm = np.nan
        self.smoothed_tr = np.nan
        self.smoothed_dx = np.nan
        
        # Initialize current values
        self._current_values = {
            'adx': np.nan,
            'plus_di': np.nan,
            'minus_di': np.nan
        }
        
    def update(self, high: float, low: float, close: float) -> dict:
        """Update ADX with new HLC values."""
        self._update_count += 1
        
        if self._update_count == 1:
            # First update - just store values
            self.prev_high = high
            self.prev_low = low
            self.prev_close = close
            return self._current_values.copy()
        
        # Calculate directional movement
        plus_dm = max(high - self.prev_high, 0.0) if high - self.prev_high > self.prev_low - low else 0.0
        minus_dm = max(self.prev_low - low, 0.0) if self.prev_low - low > high - self.prev_high else 0.0
        
        # Calculate true range
        tr = max(
            high - low,
            abs(high - self.prev_close),
            abs(low - self.prev_close)
        )
        
        # Smooth using Wilder's method
        if np.isnan(self.smoothed_plus_dm):
            self.smoothed_plus_dm = plus_dm
            self.smoothed_minus_dm = minus_dm
            self.smoothed_tr = tr
        else:
            self.smoothed_plus_dm = (1 - self.alpha) * self.smoothed_plus_dm + self.alpha * plus_dm
            self.smoothed_minus_dm = (1 - self.alpha) * self.smoothed_minus_dm + self.alpha * minus_dm
            self.smoothed_tr = (1 - self.alpha) * self.smoothed_tr + self.alpha * tr
        
        # Calculate DI values
        if self.smoothed_tr > 0:
            plus_di = 100 * (self.smoothed_plus_dm / self.smoothed_tr)
            minus_di = 100 * (self.smoothed_minus_dm / self.smoothed_tr)
            
            self._current_values['plus_di'] = plus_di
            self._current_values['minus_di'] = minus_di
            
            # Calculate DX
            di_sum = plus_di + minus_di
            if di_sum > 0:
                dx = 100 * abs(plus_di - minus_di) / di_sum
                
                # Smooth DX to get ADX
                if np.isnan(self.smoothed_dx):
                    self.smoothed_dx = dx
                else:
                    self.smoothed_dx = (1 - self.alpha) * self.smoothed_dx + self.alpha * dx
                
                self._current_values['adx'] = self.smoothed_dx
                
                # ADX is ready after enough updates
                if self._update_count >= self.window:
                    self._is_ready = True
        
        # Store current values for next update
        self.prev_high = high
        self.prev_low = low
        self.prev_close = close
        
        return self._current_values.copy()
    
    @property
    def current_value(self) -> float:
        """Get current ADX value."""
        return self._current_values['adx']


class VortexIndicatorStreaming(StreamingIndicatorMultiple):
    """
    Streaming Vortex Indicator.
    
    Returns: {
        'vi_plus': VI+ value,
        'vi_minus': VI- value
    }
    """
    
    def __init__(self, window: int = 14):
        super().__init__(window)
        
        # Buffers for calculation
        self.vm_plus_buffer = deque(maxlen=window)
        self.vm_minus_buffer = deque(maxlen=window)
        self.tr_buffer = deque(maxlen=window)
        
        # Previous values for calculation
        self.prev_high = np.nan
        self.prev_low = np.nan
        self.prev_close = np.nan
        
        # Initialize current values
        self._current_values = {
            'vi_plus': np.nan,
            'vi_minus': np.nan
        }
        
    def update(self, high: float, low: float, close: float) -> dict:
        """Update Vortex Indicator with new HLC values."""
        self._update_count += 1
        
        if self._update_count == 1:
            # First update - just store values
            self.prev_high = high
            self.prev_low = low
            self.prev_close = close
            return self._current_values.copy()
        
        # Calculate vortex movements
        vm_plus = abs(high - self.prev_low)
        vm_minus = abs(low - self.prev_high)
        
        # Calculate true range
        tr = max(
            high - low,
            abs(high - self.prev_close),
            abs(low - self.prev_close)
        )
        
        # Add to buffers
        self.vm_plus_buffer.append(vm_plus)
        self.vm_minus_buffer.append(vm_minus)
        self.tr_buffer.append(tr)
        
        # Calculate VI when we have enough data
        if len(self.vm_plus_buffer) >= self.window:
            sum_vm_plus = sum(self.vm_plus_buffer)
            sum_vm_minus = sum(self.vm_minus_buffer)
            sum_tr = sum(self.tr_buffer)
            
            if sum_tr > 0:
                self._current_values['vi_plus'] = sum_vm_plus / sum_tr
                self._current_values['vi_minus'] = sum_vm_minus / sum_tr
                self._is_ready = True
        
        # Store current values for next update
        self.prev_high = high
        self.prev_low = low
        self.prev_close = close
        
        return self._current_values.copy()
    
    @property
    def current_value(self) -> float:
        """Get current VI+ value."""
        return self._current_values['vi_plus']


class TRIXStreaming(StreamingIndicator):
    """
    Streaming TRIX indicator.
    
    Triple smoothed exponential moving average rate of change.
    """
    
    def __init__(self, window: int = 14):
        super().__init__(window)
        
        # Three EMA streamers for triple smoothing
        self.ema1 = EMAStreaming(window)
        self.ema2 = EMAStreaming(window)
        self.ema3 = EMAStreaming(window)
        
        # Previous EMA3 value for rate of change calculation
        self.prev_ema3 = np.nan
        
    def update(self, value: float) -> float:
        """Update TRIX with new value."""
        self._update_count += 1
        
        # Apply triple smoothing
        ema1_val = self.ema1.update(value)
        ema2_val = self.ema2.update(ema1_val)
        ema3_val = self.ema3.update(ema2_val)
        
        # Calculate rate of change
        if not np.isnan(self.prev_ema3) and not np.isnan(ema3_val) and self.prev_ema3 != 0:
            self._current_value = 100 * (ema3_val - self.prev_ema3) / self.prev_ema3
            
            # TRIX is ready when all three EMAs are ready
            if self.ema1.is_ready and self.ema2.is_ready and self.ema3.is_ready:
                self._is_ready = True
        
        self.prev_ema3 = ema3_val
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


class DPOStreaming(StreamingIndicator):
    """
    Streaming Detrended Price Oscillator (DPO).
    
    Removes trend from price using displaced moving average.
    """
    
    def __init__(self, window: int = 20):
        super().__init__(window)
        self.displacement = window // 2 + 1
        self.sma_stream = SMAStreaming(window)
        
        # Buffer to store prices for displacement
        self.price_buffer = deque(maxlen=window)
        
    def update(self, value: float) -> float:
        """Update DPO with new value."""
        self._update_count += 1
        
        # Store price for displacement calculation
        self.price_buffer.append(value)
        
        # Update SMA
        sma_value = self.sma_stream.update(value)
        
        # Calculate DPO when we have enough data
        if len(self.price_buffer) >= self.displacement and self.sma_stream.is_ready:
            # Get displaced price
            displaced_price = self.price_buffer[len(self.price_buffer) - self.displacement]
            
            # DPO = displaced_price - current_sma
            self._current_value = displaced_price - sma_value
            self._is_ready = True
        
        return self._current_value


class AroonStreaming(StreamingIndicatorMultiple):
    """
    Streaming Aroon indicator.
    
    Returns: {
        'aroon_up': Aroon Up value,
        'aroon_down': Aroon Down value
    }
    """
    
    def __init__(self, window: int = 25):
        super().__init__(window)
        
        # Buffers for high/low tracking
        self.high_buffer = deque(maxlen=window + 1)  # +1 for n+1 periods
        self.low_buffer = deque(maxlen=window + 1)
        
        # Initialize current values
        self._current_values = {
            'aroon_up': np.nan,
            'aroon_down': np.nan
        }
        
    def update(self, high: float, low: float) -> dict:
        """Update Aroon with new HL values."""
        self._update_count += 1
        
        # Add to buffers
        self.high_buffer.append(high)
        self.low_buffer.append(low)
        
        # Calculate Aroon when we have enough data
        if len(self.high_buffer) >= self.window + 1:
            high_list = list(self.high_buffer)
            low_list = list(self.low_buffer)
            
            # Find periods since highest high and lowest low
            max_idx = high_list.index(max(high_list))
            min_idx = low_list.index(min(low_list))
            
            # Calculate periods since (0-based indexing)
            periods_since_high = len(high_list) - 1 - max_idx
            periods_since_low = len(low_list) - 1 - min_idx
            
            # Aroon formula
            self._current_values['aroon_up'] = (self.window - periods_since_high) / self.window * 100.0
            self._current_values['aroon_down'] = (self.window - periods_since_low) / self.window * 100.0
            
            self._is_ready = True
        
        return self._current_values.copy()
    
    @property
    def current_value(self) -> float:
        """Get current Aroon Up value."""
        return self._current_values['aroon_up']


class ParabolicSARStreaming(StreamingIndicator):
    """
    Streaming Parabolic SAR.
    
    Stop and Reverse indicator for trend following.
    """
    
    def __init__(self, af_start: float = 0.02, af_inc: float = 0.02, af_max: float = 0.2):
        super().__init__(1)  # No fixed window
        
        self.af_start = af_start
        self.af_inc = af_inc
        self.af_max = af_max
        
        # SAR state variables
        self.up_trend = True
        self.acceleration_factor = af_start
        self.up_trend_high = np.nan
        self.down_trend_low = np.nan
        self.prev_sar = np.nan
        self.prev_high = np.nan
        self.prev_low = np.nan
        
    def update(self, high: float, low: float, close: float) -> float:
        """Update Parabolic SAR with new HLC values."""
        self._update_count += 1
        
        if self._update_count == 1:
            # Initialize SAR with close price
            self._current_value = close
            self.prev_sar = close
            self.up_trend_high = high
            self.down_trend_low = low
            self.prev_high = high
            self.prev_low = low
            return self._current_value
        
        if self._update_count == 2:
            # Second update - just store values
            self.prev_high = high
            self.prev_low = low
            self._current_value = close
            return self._current_value
        
        # Calculate SAR (starting from 3rd update)
        reversal = False
        
        if self.up_trend:
            # Calculate SAR for uptrend
            self._current_value = self.prev_sar + self.acceleration_factor * (self.up_trend_high - self.prev_sar)
            
            # Check for reversal
            if low < self._current_value:
                reversal = True
                self._current_value = self.up_trend_high
                self.down_trend_low = low
                self.acceleration_factor = self.af_start
            else:
                # Update extreme point and acceleration factor
                if high > self.up_trend_high:
                    self.up_trend_high = high
                    self.acceleration_factor = min(self.acceleration_factor + self.af_inc, self.af_max)
                
                # Apply SAR constraints for uptrend
                if self.prev_low < self._current_value:
                    self._current_value = self.prev_low
                elif len(self.buffer) >= 2 and self.buffer[-2] < self._current_value:
                    self._current_value = self.buffer[-2]
        else:
            # Calculate SAR for downtrend
            self._current_value = self.prev_sar - self.acceleration_factor * (self.prev_sar - self.down_trend_low)
            
            # Check for reversal
            if high > self._current_value:
                reversal = True
                self._current_value = self.down_trend_low
                self.up_trend_high = high
                self.acceleration_factor = self.af_start
            else:
                # Update extreme point and acceleration factor
                if low < self.down_trend_low:
                    self.down_trend_low = low
                    self.acceleration_factor = min(self.acceleration_factor + self.af_inc, self.af_max)
                
                # Apply SAR constraints for downtrend
                if self.prev_high > self._current_value:
                    self._current_value = self.prev_high
                elif len(self.buffer) >= 2 and self.buffer[-2] > self._current_value:
                    self._current_value = self.buffer[-2]
        
        # Update trend direction
        self.up_trend = self.up_trend != reversal  # XOR logic
        
        # Store values for next update
        self.prev_sar = self._current_value
        self.prev_high = high
        self.prev_low = low
        
        # Store in buffer for constraints
        self.buffer.append(low if self.up_trend else high)
        
        # SAR is ready after 3rd update
        if self._update_count >= 3:
            self._is_ready = True
        
        return self._current_value


# Helper functions for streaming calculations
@njit(fastmath=True)
def _streaming_wma(buffer: np.ndarray, weights: np.ndarray, sum_weights: float) -> float:
    """Fast WMA calculation for streaming."""
    return np.sum(buffer * weights) / sum_weights