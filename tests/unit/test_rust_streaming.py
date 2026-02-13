"""Test API compatibility of Rust-backed streaming indicators."""
import math
import numpy as np
import pytest

pytest.importorskip("ta_numba._ta_numba_rs")

from ta_numba import streaming


class TestStreamingAPI:
    """Verify all streaming classes have correct API surface."""

    def test_sma_api(self):
        sma = streaming.SMAStreaming(20)
        assert hasattr(sma, 'update')
        assert hasattr(sma, 'current_value')
        assert hasattr(sma, 'is_ready')
        assert hasattr(sma, 'update_count')
        assert hasattr(sma, 'reset')

    def test_sma_returns_dict(self):
        sma = streaming.SMAStreaming(3)
        result = sma.update(100.0)
        assert isinstance(result, dict)
        assert "sma" in result

    def test_sma_properties_update(self):
        sma = streaming.SMAStreaming(3)
        assert sma.update_count == 0
        assert not sma.is_ready
        sma.update(1.0)
        sma.update(2.0)
        sma.update(3.0)
        assert sma.update_count == 3
        assert sma.is_ready
        assert abs(sma.current_value - 2.0) < 1e-10

    def test_sma_reset(self):
        sma = streaming.SMAStreaming(3)
        sma.update(1.0)
        sma.update(2.0)
        sma.reset()
        assert sma.update_count == 0
        assert not sma.is_ready

    def test_macd_returns_dict(self):
        macd = streaming.MACDStreaming(12, 26, 9)
        result = macd.update(100.0)
        assert isinstance(result, dict)
        assert "macd" in result
        assert "signal" in result
        assert "histogram" in result

    def test_rsi_returns_dict(self):
        rsi = streaming.RSIStreaming(14)
        result = rsi.update(100.0)
        assert isinstance(result, dict)
        assert "rsi" in result

    def test_bollinger_bands_returns_dict(self):
        bb = streaming.BBandsStreaming(20, 2.0)
        result = bb.update(100.0)
        assert isinstance(result, dict)
        assert "upper" in result
        assert "middle" in result
        assert "lower" in result

    def test_adx_returns_dict(self):
        adx = streaming.ADXStreaming(14)
        result = adx.update(105.0, 95.0, 100.0)
        assert isinstance(result, dict)
        assert "adx" in result
        assert "plus_di" in result
        assert "minus_di" in result

    def test_stochastic_returns_dict(self):
        stoch = streaming.StochasticStreaming(14, 3)
        result = stoch.update(105.0, 95.0, 100.0)
        assert isinstance(result, dict)
        assert "percent_k" in result
        assert "percent_d" in result

    def test_obv_returns_dict(self):
        obv = streaming.OnBalanceVolumeStreaming()
        result = obv.update(100.0, 1000.0)
        assert isinstance(result, dict)
        assert "obv" in result

    def test_mfi_returns_dict(self):
        mfi = streaming.MoneyFlowIndexStreaming(14)
        result = mfi.update(105.0, 95.0, 100.0, 1000.0)
        assert isinstance(result, dict)
        assert "mfi" in result

    def test_atr_returns_dict(self):
        atr = streaming.ATRStreaming(14)
        result = atr.update(105.0, 95.0, 100.0)
        assert isinstance(result, dict)
        assert "atr" in result


class TestStreamingAliases:
    """Verify short aliases work."""

    def test_short_aliases_exist(self):
        assert hasattr(streaming, 'SMA')
        assert hasattr(streaming, 'EMA')
        assert hasattr(streaming, 'RSI')
        assert hasattr(streaming, 'MACD')
        assert hasattr(streaming, 'ATR')
        assert hasattr(streaming, 'VWAP')

    def test_short_alias_is_same_class(self):
        assert streaming.SMA is streaming.SMAStreaming
        assert streaming.RSI is streaming.RSIStreaming
        assert streaming.MACD is streaming.MACDStreaming


class TestStreamingFunctionality:
    """Test actual computation correctness of streaming indicators."""

    def test_sma_computation(self):
        sma = streaming.SMAStreaming(3)
        sma.update(10.0)
        sma.update(20.0)
        result = sma.update(30.0)
        assert abs(result["sma"] - 20.0) < 1e-10

    def test_ema_computation(self):
        ema = streaming.EMAStreaming(3)
        ema.update(10.0)
        result = ema.update(20.0)
        # EMA should be between 10 and 20
        assert 10.0 < result["ema"] < 20.0

    def test_rsi_range(self):
        rsi = streaming.RSIStreaming(14)
        for i in range(50):
            result = rsi.update(100.0 + i)
        val = result["rsi"]
        if not math.isnan(val):
            assert 0 <= val <= 100

    def test_daily_return_computation(self):
        dr = streaming.DailyReturnStreaming()
        dr.update(100.0)
        result = dr.update(105.0)
        assert abs(result["dr"] - 5.0) < 1e-10  # 5% return
