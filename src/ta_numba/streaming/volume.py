"""
Streaming volume indicators for real-time trading systems.
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


class MoneyFlowIndexStreaming(StreamingIndicator):
    """
    Streaming Money Flow Index (MFI).
    
    Volume-weighted RSI.
    """
    
    def __init__(self, window: int = 14):
        super().__init__(window)
        
        # Buffers for calculation
        self.positive_mf_buffer = deque(maxlen=window)
        self.negative_mf_buffer = deque(maxlen=window)
        
        # Previous typical price
        self.prev_tp = np.nan
        
    def update(self, high: float, low: float, close: float, volume: float) -> float:
        """Update MFI with new HLCV values."""
        self._update_count += 1
        
        # Calculate typical price
        typical_price = (high + low + close) / 3.0
        
        # Calculate raw money flow
        rmf = typical_price * volume
        
        # Determine direction and calculate money flow
        if np.isnan(self.prev_tp):
            # First update - neutral
            positive_mf = 0.0
            negative_mf = 0.0
        elif typical_price > self.prev_tp:
            positive_mf = rmf
            negative_mf = 0.0
        elif typical_price < self.prev_tp:
            positive_mf = 0.0
            negative_mf = rmf
        else:
            positive_mf = 0.0
            negative_mf = 0.0
        
        # Add to buffers
        self.positive_mf_buffer.append(positive_mf)
        self.negative_mf_buffer.append(negative_mf)
        
        # Calculate MFI when we have enough data
        if len(self.positive_mf_buffer) >= self.window:
            pos_sum = sum(self.positive_mf_buffer)
            neg_sum = sum(self.negative_mf_buffer)
            
            if neg_sum == 0:
                self._current_value = 100.0
            else:
                mfr = pos_sum / neg_sum
                self._current_value = 100.0 - (100.0 / (1.0 + mfr))
            
            self._is_ready = True
        
        self.prev_tp = typical_price
        return self._current_value


class AccDistIndexStreaming(StreamingIndicator):
    """
    Streaming Accumulation/Distribution Index.
    
    Cumulative volume-weighted price momentum.
    """
    
    def __init__(self):
        super().__init__(1)  # No fixed window
        self.ad_line = 0.0
        
    def update(self, high: float, low: float, close: float, volume: float) -> float:
        """Update A/D Index with new HLCV values."""
        self._update_count += 1
        
        # Calculate money flow multiplier
        if high != low:
            mfm = ((close - low) - (high - close)) / (high - low)
        else:
            mfm = 0.0
        
        # Calculate money flow volume
        mfv = mfm * volume
        
        # Update cumulative A/D line
        self.ad_line += mfv
        self._current_value = self.ad_line
        
        # A/D is ready after first update
        if self._update_count >= 1:
            self._is_ready = True
        
        return self._current_value


class OnBalanceVolumeStreaming(StreamingIndicator):
    """
    Streaming On-Balance Volume (OBV).
    
    Cumulative volume based on price direction.
    """
    
    def __init__(self):
        super().__init__(1)  # No fixed window
        self.obv_line = 0.0
        self.prev_close = np.nan
        
    def update(self, close: float, volume: float) -> float:
        """Update OBV with new close/volume values."""
        self._update_count += 1
        
        if self._update_count == 1:
            # First update - initialize with volume
            self.obv_line = volume
            self._current_value = self.obv_line
            self._is_ready = True
        else:
            # Update based on price direction
            if close > self.prev_close:
                self.obv_line += volume
            elif close < self.prev_close:
                self.obv_line -= volume
            # If close == prev_close, no change to OBV
            
            self._current_value = self.obv_line
        
        self.prev_close = close
        return self._current_value


class ChaikinMoneyFlowStreaming(StreamingIndicator):
    """
    Streaming Chaikin Money Flow (CMF).
    
    Volume-weighted average of money flow multiplier.
    """
    
    def __init__(self, window: int = 20):
        super().__init__(window)
        
        # Buffers for calculation
        self.mfv_buffer = deque(maxlen=window)
        self.volume_buffer = deque(maxlen=window)
        
    def update(self, high: float, low: float, close: float, volume: float) -> float:
        """Update CMF with new HLCV values."""
        self._update_count += 1
        
        # Calculate money flow multiplier
        if high != low:
            mfm = ((close - low) - (high - close)) / (high - low)
        else:
            mfm = 0.0
        
        # Calculate money flow volume
        mfv = mfm * volume
        
        # Add to buffers
        self.mfv_buffer.append(mfv)
        self.volume_buffer.append(volume)
        
        # Calculate CMF when we have enough data
        if len(self.mfv_buffer) >= self.window:
            sum_mfv = sum(self.mfv_buffer)
            sum_volume = sum(self.volume_buffer)
            
            if sum_volume != 0:
                self._current_value = sum_mfv / sum_volume
            else:
                self._current_value = 0.0
            
            self._is_ready = True
        
        return self._current_value


class ForceIndexStreaming(StreamingIndicator):
    """
    Streaming Force Index.
    
    EMA of (close - prev_close) * volume.
    """
    
    def __init__(self, window: int = 13):
        super().__init__(window)
        self.alpha = 2.0 / (window + 1.0)
        self.prev_close = np.nan
        
    def update(self, close: float, volume: float) -> float:
        """Update Force Index with new close/volume values."""
        self._update_count += 1
        
        if self._update_count == 1:
            # First update - store close
            self.prev_close = close
            return self._current_value
        
        # Calculate raw force index
        force_value = (close - self.prev_close) * volume
        
        # Apply EMA smoothing directly without function call overhead
        if np.isnan(self._current_value):
            self._current_value = force_value
        else:
            self._current_value = self.alpha * force_value + (1 - self.alpha) * self._current_value
        
        # Force Index is ready after enough updates
        if self._update_count >= self.window:
            self._is_ready = True
        
        self.prev_close = close
        return self._current_value


class EaseOfMovementStreaming(StreamingIndicator):
    """
    Streaming Ease of Movement (EMV).
    
    Measures volume-weighted price movement.
    """
    
    def __init__(self, window: int = 14):
        super().__init__(window)
        
        # Previous values for calculation
        self.prev_high = np.nan
        self.prev_low = np.nan
        
    def update(self, high: float, low: float, volume: float) -> float:
        """Update EMV with new HLV values."""
        self._update_count += 1
        
        if self._update_count == 1:
            # First update - store values
            self.prev_high = high
            self.prev_low = low
            return self._current_value
        
        # Calculate ease of movement
        if volume != 0:
            # Distance moved
            distance_moved = ((high - self.prev_high) + (low - self.prev_low)) / 2.0
            
            # Box height
            box_height = high - low
            
            # EMV formula with scaling
            emv_value = distance_moved * box_height / volume * 100000000
            self._current_value = emv_value
            
            # EMV is ready after second update
            if self._update_count >= 2:
                self._is_ready = True
        
        self.prev_high = high
        self.prev_low = low
        return self._current_value


class VolumePriceTrendStreaming(StreamingIndicator):
    """
    Streaming Volume Price Trend (VPT).
    
    Cumulative volume * price change percentage.
    """
    
    def __init__(self):
        super().__init__(1)  # No fixed window
        self.vpt_line = 0.0
        self.prev_close = np.nan
        
    def update(self, close: float, volume: float) -> float:
        """Update VPT with new close/volume values."""
        self._update_count += 1
        
        if self._update_count == 1:
            # First update - initialize
            self.vpt_line = 0.0
            self._current_value = self.vpt_line
            self._is_ready = True
        else:
            # Calculate percentage change
            if self.prev_close != 0:
                pct_change = (close - self.prev_close) / self.prev_close
                vpt_change = volume * pct_change
                self.vpt_line += vpt_change
            
            self._current_value = self.vpt_line
        
        self.prev_close = close
        return self._current_value


class NegativeVolumeIndexStreaming(StreamingIndicator):
    """
    Streaming Negative Volume Index (NVI).
    
    Cumulative price change on days with decreasing volume.
    """
    
    def __init__(self):
        super().__init__(1)  # No fixed window
        self.nvi_line = 1000.0  # Start at 1000
        self.prev_close = np.nan
        self.prev_volume = np.nan
        
    def update(self, close: float, volume: float) -> float:
        """Update NVI with new close/volume values."""
        self._update_count += 1
        
        if self._update_count == 1:
            # First update - initialize
            self.nvi_line = 1000.0
            self._current_value = self.nvi_line
            self._is_ready = True
        else:
            # Update NVI only on decreasing volume
            if volume < self.prev_volume and self.prev_close != 0:
                pct_change = (close - self.prev_close) / self.prev_close
                self.nvi_line *= (1 + pct_change)
            
            self._current_value = self.nvi_line
        
        self.prev_close = close
        self.prev_volume = volume
        return self._current_value


class VWAPStreaming(StreamingIndicator):
    """
    Streaming Volume Weighted Average Price (VWAP).
    
    Volume-weighted average of typical price.
    """
    
    def __init__(self, window: int = 14):
        super().__init__(window)
        
        # Buffers for calculation
        self.tpv_buffer = deque(maxlen=window)
        self.volume_buffer = deque(maxlen=window)
        
    def update(self, high: float, low: float, close: float, volume: float) -> float:
        """Update VWAP with new HLCV values."""
        self._update_count += 1
        
        # Calculate typical price
        typical_price = (high + low + close) / 3.0
        
        # Calculate typical price * volume
        tpv = typical_price * volume
        
        # Add to buffers
        self.tpv_buffer.append(tpv)
        self.volume_buffer.append(volume)
        
        # Calculate VWAP when we have enough data
        if len(self.tpv_buffer) >= self.window:
            sum_tpv = sum(self.tpv_buffer)
            sum_volume = sum(self.volume_buffer)
            
            if sum_volume != 0:
                self._current_value = sum_tpv / sum_volume
            else:
                self._current_value = 0.0
            
            self._is_ready = True
        
        return self._current_value


class VWEMAStreaming(StreamingIndicator):
    """
    Streaming Volume Weighted Exponential Moving Average (VWEMA).
    
    EMA of Volume Weighted Moving Average.
    """
    
    def __init__(self, vwma_period: int = 14, ema_period: int = 20):
        super().__init__(vwma_period)
        self.vwma_period = vwma_period
        self.ema_period = ema_period
        
        # VWAP streamer for intermediate calculation
        self.vwap_stream = VWAPStreaming(vwma_period)
        
        # EMA streamer for final result
        self.ema_stream = EMAStreaming(ema_period)
        
    def update(self, high: float, low: float, close: float, volume: float) -> float:
        """Update VWEMA with new HLCV values."""
        self._update_count += 1
        
        # Update VWAP
        vwap_value = self.vwap_stream.update(high, low, close, volume)
        
        # Update EMA of VWAP
        if self.vwap_stream.is_ready:
            self._current_value = self.ema_stream.update(vwap_value)
            
            # VWEMA is ready when EMA is ready
            if self.ema_stream.is_ready:
                self._is_ready = True
        
        return self._current_value


# Import required streamers
from .trend import EMAStreaming