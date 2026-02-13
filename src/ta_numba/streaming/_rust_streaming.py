"""Rust streaming class wrappers to match Python API (dict returns, properties)."""

import math
from .._backend import _rs

# ============================================================================
# TREND INDICATORS (11 classes)
# ============================================================================

class SMAStreaming:
    """Simple Moving Average - Streaming"""
    def __init__(self, window=20):
        self._inner = _rs.SMAStreaming(window)
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = window

    def update(self, value):
        self._update_count += 1
        result = self._inner.update(value)
        self._current_value = result
        self._is_ready = not math.isnan(result)
        return {"sma": result}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


class EMAStreaming:
    """Exponential Moving Average - Streaming"""
    def __init__(self, window=20):
        self._inner = _rs.EMAStreaming(window)
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = window

    def update(self, value):
        self._update_count += 1
        result = self._inner.update(value)
        self._current_value = result
        self._is_ready = not math.isnan(result)
        return {"ema": result}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


class WMAStreaming:
    """Weighted Moving Average - Streaming"""
    def __init__(self, window=20):
        self._inner = _rs.WMAStreaming(window)
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = window

    def update(self, value):
        self._update_count += 1
        result = self._inner.update(value)
        self._current_value = result
        self._is_ready = not math.isnan(result)
        return {"wma": result}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


class MACDStreaming:
    """MACD - Streaming"""
    def __init__(self, fast_period=12, slow_period=26, signal_period=9):
        self._inner = _rs.MACDStreaming(fast_period, slow_period, signal_period)
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = slow_period

    def update(self, value):
        self._update_count += 1
        macd, signal, hist = self._inner.update(value)
        self._current_value = macd
        self._is_ready = not math.isnan(macd)
        return {"macd": macd, "signal": signal, "histogram": hist}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


class ADXStreaming:
    """Average Directional Index - Streaming"""
    def __init__(self, window=14):
        self._inner = _rs.ADXStreaming(window)
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = window

    def update(self, high, low, close):
        self._update_count += 1
        adx, plus_di, minus_di = self._inner.update(high, low, close)
        self._current_value = adx
        self._is_ready = not math.isnan(adx)
        return {"adx": adx, "plus_di": plus_di, "minus_di": minus_di}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


class CCIStreaming:
    """Commodity Channel Index - Streaming"""
    def __init__(self, window=20, constant=0.015):
        self._inner = _rs.CCIStreaming(window, constant)
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = window

    def update(self, high, low, close):
        self._update_count += 1
        result = self._inner.update(high, low, close)
        self._current_value = result
        self._is_ready = not math.isnan(result)
        return {"cci": result}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


class DPOStreaming:
    """Detrended Price Oscillator - Streaming"""
    def __init__(self, window=20):
        self._inner = _rs.DPOStreaming(window)
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = window

    def update(self, value):
        self._update_count += 1
        result = self._inner.update(value)
        self._current_value = result
        self._is_ready = not math.isnan(result)
        return {"dpo": result}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


class VortexIndicatorStreaming:
    """Vortex Indicator - Streaming"""
    def __init__(self, window=14):
        self._inner = _rs.VortexIndicatorStreaming(window)
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = window

    def update(self, high, low, close):
        self._update_count += 1
        vi_plus, vi_minus = self._inner.update(high, low, close)
        self._current_value = vi_plus
        self._is_ready = not math.isnan(vi_plus)
        return {"vi_plus": vi_plus, "vi_minus": vi_minus}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


class TRIXStreaming:
    """TRIX - Streaming"""
    def __init__(self, window=14):
        self._inner = _rs.TRIXStreaming(window)
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = window

    def update(self, value):
        self._update_count += 1
        result = self._inner.update(value)
        self._current_value = result
        self._is_ready = not math.isnan(result)
        return {"trix": result}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


class AroonStreaming:
    """Aroon Indicator - Streaming"""
    def __init__(self, window=25):
        self._inner = _rs.AroonStreaming(window)
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = window

    def update(self, high, low):
        self._update_count += 1
        aroon_up, aroon_down = self._inner.update(high, low)
        self._current_value = aroon_up
        self._is_ready = not math.isnan(aroon_up)
        return {"aroon_up": aroon_up, "aroon_down": aroon_down}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


class ParabolicSARStreaming:
    """Parabolic SAR - Streaming"""
    def __init__(self, af_start=0.02, af_inc=0.02, af_max=0.2):
        self._inner = _rs.ParabolicSARStreaming(af_start, af_inc, af_max)
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = 1

    def update(self, high, low, close):
        self._update_count += 1
        result = self._inner.update(high, low, close)
        self._current_value = result
        self._is_ready = not math.isnan(result)
        return {"psar": result}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


# ============================================================================
# MOMENTUM INDICATORS (12 classes)
# ============================================================================

class RSIStreaming:
    """Relative Strength Index - Streaming"""
    def __init__(self, window=14):
        self._inner = _rs.RSIStreaming(window)
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = window

    def update(self, value):
        self._update_count += 1
        result = self._inner.update(value)
        self._current_value = result
        self._is_ready = not math.isnan(result)
        return {"rsi": result}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


class StochasticStreaming:
    """Stochastic Oscillator - Streaming"""
    def __init__(self, k_period=14, d_period=3):
        self._inner = _rs.StochasticStreaming(k_period, d_period)
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = k_period

    def update(self, high, low, close):
        self._update_count += 1
        k, d = self._inner.update(high, low, close)
        self._current_value = k
        self._is_ready = not math.isnan(k)
        return {"percent_k": k, "percent_d": d}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


class WilliamsRStreaming:
    """Williams %R - Streaming"""
    def __init__(self, window=14):
        self._inner = _rs.WilliamsRStreaming(window)
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = window

    def update(self, high, low, close):
        self._update_count += 1
        result = self._inner.update(high, low, close)
        self._current_value = result
        self._is_ready = not math.isnan(result)
        return {"williams_r": result}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


class ROCStreaming:
    """Rate of Change - Streaming"""
    def __init__(self, window=12):
        self._inner = _rs.ROCStreaming(window)
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = window

    def update(self, value):
        self._update_count += 1
        result = self._inner.update(value)
        self._current_value = result
        self._is_ready = not math.isnan(result)
        return {"roc": result}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


class PPOStreaming:
    """Percentage Price Oscillator - Streaming"""
    def __init__(self, fast_period=12, slow_period=26, signal_period=9):
        self._inner = _rs.PPOStreaming(fast_period, slow_period, signal_period)
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = slow_period

    def update(self, value):
        self._update_count += 1
        ppo, signal, hist = self._inner.update(value)
        self._current_value = ppo
        self._is_ready = not math.isnan(ppo)
        return {"ppo": ppo, "signal": signal, "histogram": hist}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


class PVOStreaming:
    """Percentage Volume Oscillator - Streaming"""
    def __init__(self, fast_period=12, slow_period=26, signal_period=9):
        self._inner = _rs.PVOStreaming(fast_period, slow_period, signal_period)
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = slow_period

    def update(self, value):
        self._update_count += 1
        pvo, signal, hist = self._inner.update(value)
        self._current_value = pvo
        self._is_ready = not math.isnan(pvo)
        return {"pvo": pvo, "signal": signal, "histogram": hist}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


class UltimateOscillatorStreaming:
    """Ultimate Oscillator - Streaming"""
    def __init__(self, period1=7, period2=14, period3=28):
        self._inner = _rs.UltimateOscillatorStreaming(period1, period2, period3)
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = period3

    def update(self, high, low, close):
        self._update_count += 1
        result = self._inner.update(high, low, close)
        self._current_value = result
        self._is_ready = not math.isnan(result)
        return {"uo": result}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


class StochasticRSIStreaming:
    """Stochastic RSI - Streaming"""
    def __init__(self, rsi_period=14, stoch_period=14, k_period=3, d_period=3):
        self._inner = _rs.StochasticRSIStreaming(rsi_period, stoch_period, k_period, d_period)
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = rsi_period

    def update(self, value):
        self._update_count += 1
        stochrsi, k, d = self._inner.update(value)
        self._current_value = stochrsi
        self._is_ready = not math.isnan(stochrsi)
        return {"stochrsi": stochrsi, "k": k, "d": d}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


class TSIStreaming:
    """True Strength Index - Streaming"""
    def __init__(self, first_smooth=25, second_smooth=13):
        self._inner = _rs.TSIStreaming(first_smooth, second_smooth)
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = first_smooth

    def update(self, value):
        self._update_count += 1
        result = self._inner.update(value)
        self._current_value = result
        self._is_ready = not math.isnan(result)
        return {"tsi": result}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


class AwesomeOscillatorStreaming:
    """Awesome Oscillator - Streaming"""
    def __init__(self, fast_period=5, slow_period=34):
        self._inner = _rs.AwesomeOscillatorStreaming(fast_period, slow_period)
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = slow_period

    def update(self, high, low):
        self._update_count += 1
        result = self._inner.update(high, low)
        self._current_value = result
        self._is_ready = not math.isnan(result)
        return {"ao": result}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


class KAMAStreaming:
    """Kaufman's Adaptive Moving Average - Streaming"""
    def __init__(self, window=10, fast_period=2, slow_period=30):
        self._inner = _rs.KAMAStreaming(window, fast_period, slow_period)
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = window

    def update(self, value):
        self._update_count += 1
        result = self._inner.update(value)
        self._current_value = result
        self._is_ready = not math.isnan(result)
        return {"kama": result}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


class MomentumStreaming:
    """Momentum - Streaming"""
    def __init__(self, window=10):
        self._inner = _rs.MomentumStreaming(window)
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = window

    def update(self, value):
        self._update_count += 1
        result = self._inner.update(value)
        self._current_value = result
        self._is_ready = not math.isnan(result)
        return {"momentum": result}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


# ============================================================================
# VOLATILITY INDICATORS (5 classes)
# ============================================================================

class ATRStreaming:
    """Average True Range - Streaming"""
    def __init__(self, window=14):
        self._inner = _rs.ATRStreaming(window)
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = window

    def update(self, high, low, close):
        self._update_count += 1
        result = self._inner.update(high, low, close)
        self._current_value = result
        self._is_ready = not math.isnan(result)
        return {"atr": result}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


class BBandsStreaming:
    """Bollinger Bands - Streaming"""
    def __init__(self, window=20, std_dev=2.0):
        self._inner = _rs.BBandsStreaming(window, std_dev)
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = window

    def update(self, value):
        self._update_count += 1
        upper, middle, lower = self._inner.update(value)
        self._current_value = upper
        self._is_ready = not math.isnan(upper)
        return {"upper": upper, "middle": middle, "lower": lower}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


class KeltnerChannelStreaming:
    """Keltner Channel - Streaming"""
    def __init__(self, window=20, atr_period=10, multiplier=2.0):
        self._inner = _rs.KeltnerChannelStreaming(window, atr_period, multiplier)
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = window

    def update(self, high, low, close):
        self._update_count += 1
        upper, middle, lower = self._inner.update(high, low, close)
        self._current_value = upper
        self._is_ready = not math.isnan(upper)
        return {"upper": upper, "middle": middle, "lower": lower}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


class DonchianChannelStreaming:
    """Donchian Channel - Streaming"""
    def __init__(self, window=20):
        self._inner = _rs.DonchianChannelStreaming(window)
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = window

    def update(self, high, low):
        self._update_count += 1
        upper, middle, lower = self._inner.update(high, low)
        self._current_value = upper
        self._is_ready = not math.isnan(upper)
        return {"upper": upper, "middle": middle, "lower": lower}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


class UlcerIndexStreaming:
    """Ulcer Index - Streaming"""
    def __init__(self, window=14):
        self._inner = _rs.UlcerIndexStreaming(window)
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = window

    def update(self, value):
        self._update_count += 1
        result = self._inner.update(value)
        self._current_value = result
        self._is_ready = not math.isnan(result)
        return {"ui": result}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


class StandardDeviationStreaming:
    """Rolling Standard Deviation - Streaming"""
    def __init__(self, window=20):
        self._inner = _rs.StandardDeviationStreaming(window)
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = window

    def update(self, value):
        self._update_count += 1
        result = self._inner.update(value)
        self._current_value = result
        self._is_ready = not math.isnan(result)
        return {"std": result}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


class VarianceStreaming:
    """Rolling Variance - Streaming"""
    def __init__(self, window=20):
        self._inner = _rs.VarianceStreaming(window)
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = window

    def update(self, value):
        self._update_count += 1
        result = self._inner.update(value)
        self._current_value = result
        self._is_ready = not math.isnan(result)
        return {"variance": result}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


class RangeStreaming:
    """Rolling Range (High - Low) - Streaming"""
    def __init__(self, window=20):
        self._inner = _rs.RangeStreaming(window)
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = window

    def update(self, high, low):
        self._update_count += 1
        result = self._inner.update(high, low)
        self._current_value = result
        self._is_ready = not math.isnan(result)
        return {"range": result}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


class HistoricalVolatilityStreaming:
    """Historical Volatility (annualized rolling std of log returns) - Streaming"""
    def __init__(self, window=20, annualize=True):
        self._inner = _rs.HistoricalVolatilityStreaming(window, annualize)
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = window

    def update(self, value):
        self._update_count += 1
        result = self._inner.update(value)
        self._current_value = result
        self._is_ready = not math.isnan(result)
        return {"hvol": result}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


# ============================================================================
# VOLUME INDICATORS (10 classes)
# ============================================================================

class MoneyFlowIndexStreaming:
    """Money Flow Index - Streaming"""
    def __init__(self, window=14):
        self._inner = _rs.MoneyFlowIndexStreaming(window)
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = window

    def update(self, high, low, close, volume):
        self._update_count += 1
        result = self._inner.update(high, low, close, volume)
        self._current_value = result
        self._is_ready = not math.isnan(result)
        return {"mfi": result}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


class AccDistIndexStreaming:
    """Accumulation/Distribution Index - Streaming"""
    def __init__(self):
        self._inner = _rs.AccDistIndexStreaming()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = 1

    def update(self, high, low, close, volume):
        self._update_count += 1
        result = self._inner.update(high, low, close, volume)
        self._current_value = result
        self._is_ready = not math.isnan(result)
        return {"ad": result}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


class OnBalanceVolumeStreaming:
    """On-Balance Volume - Streaming"""
    def __init__(self):
        self._inner = _rs.OnBalanceVolumeStreaming()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = 1

    def update(self, close, volume):
        self._update_count += 1
        result = self._inner.update(close, volume)
        self._current_value = result
        self._is_ready = not math.isnan(result)
        return {"obv": result}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


class ChaikinMoneyFlowStreaming:
    """Chaikin Money Flow - Streaming"""
    def __init__(self, window=20):
        self._inner = _rs.ChaikinMoneyFlowStreaming(window)
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = window

    def update(self, high, low, close, volume):
        self._update_count += 1
        result = self._inner.update(high, low, close, volume)
        self._current_value = result
        self._is_ready = not math.isnan(result)
        return {"cmf": result}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


class ForceIndexStreaming:
    """Force Index - Streaming"""
    def __init__(self, window=13):
        self._inner = _rs.ForceIndexStreaming(window)
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = window

    def update(self, close, volume):
        self._update_count += 1
        result = self._inner.update(close, volume)
        self._current_value = result
        self._is_ready = not math.isnan(result)
        return {"fi": result}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


class EaseOfMovementStreaming:
    """Ease of Movement - Streaming"""
    def __init__(self, window=14):
        self._inner = _rs.EaseOfMovementStreaming(window)
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = window

    def update(self, high, low, volume):
        self._update_count += 1
        result = self._inner.update(high, low, volume)
        self._current_value = result
        self._is_ready = not math.isnan(result)
        return {"eom": result}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


class VolumePriceTrendStreaming:
    """Volume Price Trend - Streaming"""
    def __init__(self):
        self._inner = _rs.VolumePriceTrendStreaming()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = 1

    def update(self, close, volume):
        self._update_count += 1
        result = self._inner.update(close, volume)
        self._current_value = result
        self._is_ready = not math.isnan(result)
        return {"vpt": result}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


class NegativeVolumeIndexStreaming:
    """Negative Volume Index - Streaming"""
    def __init__(self):
        self._inner = _rs.NegativeVolumeIndexStreaming()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = 1

    def update(self, close, volume):
        self._update_count += 1
        result = self._inner.update(close, volume)
        self._current_value = result
        self._is_ready = not math.isnan(result)
        return {"nvi": result}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


class VWAPStreaming:
    """Volume Weighted Average Price - Streaming"""
    def __init__(self, window=14):
        self._inner = _rs.VWAPStreaming(window)
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = window

    def update(self, high, low, close, volume):
        self._update_count += 1
        result = self._inner.update(high, low, close, volume)
        self._current_value = result
        self._is_ready = not math.isnan(result)
        return {"vwap": result}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


class VWEMAStreaming:
    """Volume Weighted Exponential Moving Average - Streaming"""
    def __init__(self, vwma_period=14, ema_period=20):
        self._inner = _rs.VWEMAStreaming(vwma_period, ema_period)
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = vwma_period

    def update(self, high, low, close, volume):
        self._update_count += 1
        result = self._inner.update(high, low, close, volume)
        self._current_value = result
        self._is_ready = not math.isnan(result)
        return {"vwema": result}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


class VolumeRatioStreaming:
    """Volume Ratio (volume / SMA(volume)) - Streaming"""
    def __init__(self, window=50):
        self._inner = _rs.VolumeRatioStreaming(window)
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = window

    def update(self, volume):
        self._update_count += 1
        result = self._inner.update(volume)
        self._current_value = result
        self._is_ready = not math.isnan(result)
        return {"volume_ratio": result}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


# ============================================================================
# OTHER INDICATORS (8 classes)
# ============================================================================

class DailyReturnStreaming:
    """Daily Return - Streaming"""
    def __init__(self):
        self._inner = _rs.DailyReturnStreaming()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = 1

    def update(self, value):
        self._update_count += 1
        result = self._inner.update(value)
        self._current_value = result
        self._is_ready = not math.isnan(result)
        return {"dr": result}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


class DailyLogReturnStreaming:
    """Daily Log Return - Streaming"""
    def __init__(self):
        self._inner = _rs.DailyLogReturnStreaming()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = 1

    def update(self, value):
        self._update_count += 1
        result = self._inner.update(value)
        self._current_value = result
        self._is_ready = not math.isnan(result)
        return {"dlr": result}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


class CumulativeReturnStreaming:
    """Cumulative Return - Streaming"""
    def __init__(self):
        self._inner = _rs.CumulativeReturnStreaming()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = 1

    def update(self, value):
        self._update_count += 1
        result = self._inner.update(value)
        self._current_value = result
        self._is_ready = not math.isnan(result)
        return {"cr": result}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


class CompoundLogReturnStreaming:
    """Compound Log Return - Streaming"""
    def __init__(self):
        self._inner = _rs.CompoundLogReturnStreaming()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = 1

    def update(self, value):
        self._update_count += 1
        result = self._inner.update(value)
        self._current_value = result
        self._is_ready = True
        return {"clr": result}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


class RollingReturnStreaming:
    """Rolling Return - Streaming"""
    def __init__(self, window=20):
        self._inner = _rs.RollingReturnStreaming(window)
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = window

    def update(self, value):
        self._update_count += 1
        result = self._inner.update(value)
        self._current_value = result
        self._is_ready = not math.isnan(result)
        return {"rr": result}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


class MaxDrawdownStreaming:
    """Maximum Drawdown - Streaming"""
    def __init__(self):
        self._inner = _rs.MaxDrawdownStreaming()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = 1

    def update(self, value):
        self._update_count += 1
        result = self._inner.update(value)
        self._current_value = result
        self._is_ready = not math.isnan(result)
        return {"mdd": result}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


class SharpeRatioStreaming:
    """Sharpe Ratio - Streaming"""
    def __init__(self, window=252, risk_free_rate=0.0, annualization_factor=252.0):
        self._inner = _rs.SharpeRatioStreaming(window, risk_free_rate, annualization_factor)
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = window

    def update(self, value):
        self._update_count += 1
        result = self._inner.update(value)
        self._current_value = result
        self._is_ready = not math.isnan(result)
        return {"sharpe": result}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


class CalmarRatioStreaming:
    """Calmar Ratio - Streaming"""
    def __init__(self, window=252):
        self._inner = _rs.CalmarRatioStreaming(window)
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = window

    def update(self, value):
        self._update_count += 1
        result = self._inner.update(value)
        self._current_value = result
        self._is_ready = not math.isnan(result)
        return {"calmar": result}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


class RollingZScoreStreaming:
    """Rolling Z-Score - Streaming"""
    def __init__(self, window=20):
        self._inner = _rs.RollingZScoreStreaming(window)
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = window

    def update(self, value):
        self._update_count += 1
        result = self._inner.update(value)
        self._current_value = result
        self._is_ready = not math.isnan(result)
        return {"zscore": result}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


class LinearRegressionSlopeStreaming:
    """Linear Regression Slope - Streaming"""
    def __init__(self, window=14):
        self._inner = _rs.LinearRegressionSlopeStreaming(window)
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = window

    def update(self, value):
        self._update_count += 1
        result = self._inner.update(value)
        self._current_value = result
        self._is_ready = not math.isnan(result)
        return {"slope": result}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0


class RollingPercentileStreaming:
    """Rolling Percentile - Streaming"""
    def __init__(self, window=120):
        self._inner = _rs.RollingPercentileStreaming(window)
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
        self._window = window

    def update(self, value):
        self._update_count += 1
        result = self._inner.update(value)
        self._current_value = result
        self._is_ready = not math.isnan(result)
        return {"percentile": result}

    @property
    def current_value(self):
        return self._current_value

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def update_count(self):
        return self._update_count

    @property
    def window(self):
        return self._window

    def reset(self):
        self._inner.reset()
        self._current_value = float('nan')
        self._is_ready = False
        self._update_count = 0
