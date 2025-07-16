"""
Comprehensive tests for streaming indicators.
"""

from typing import Any, Dict, List

import numpy as np
import pytest

from ta_numba.momentum import rsi, stoch, willr

# Import streaming indicators
from ta_numba.streaming import (
    ADX,
    ATR,
    EMA,
    MACD,
    RSI,
    SMA,
    BBands,
    Stochastic,
    WilliamsR,
)

# Import bulk functions for comparison
from ta_numba.trend import adx, ema, macd, sma
from ta_numba.volatility import atr, bbands


class TestStreamingIndicators:
    """Test suite for streaming indicators."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample OHLC data for testing."""
        np.random.seed(42)
        n = 100

        # Generate realistic price data
        base_price = 100.0
        returns = np.random.normal(0, 0.02, n)
        prices = [base_price]

        for i in range(1, n):
            new_price = prices[-1] * (1 + returns[i])
            prices.append(max(new_price, 0.01))  # Ensure positive prices

        # Generate OHLC from prices
        close = np.array(prices)
        high = close * np.random.uniform(1.0, 1.05, n)
        low = close * np.random.uniform(0.95, 1.0, n)
        open_price = close * np.random.uniform(0.98, 1.02, n)

        return {"open": open_price, "high": high, "low": low, "close": close}

    def test_sma_streaming_accuracy(self, sample_data):
        """Test SMA streaming accuracy against bulk function."""
        close = sample_data["close"]
        window = 20

        # Calculate using bulk function
        bulk_sma = sma(close, window)

        # Calculate using streaming
        sma_stream = SMA(window)
        streaming_results = []

        for price in close:
            result = sma_stream.update(price)
            streaming_results.append(result)

        streaming_sma = np.array(streaming_results)

        # Compare results (ignore NaN values)
        valid_idx = ~np.isnan(bulk_sma)
        np.testing.assert_allclose(
            streaming_sma[valid_idx],
            bulk_sma[valid_idx],
            rtol=1e-10,
            err_msg="SMA streaming results don't match bulk calculation",
        )

    def test_ema_streaming_accuracy(self, sample_data):
        """Test EMA streaming accuracy against bulk function."""
        close = sample_data["close"]
        window = 20

        # Calculate using bulk function
        bulk_ema = ema(close, window)

        # Calculate using streaming
        ema_stream = EMA(window)
        streaming_results = []

        for price in close:
            result = ema_stream.update(price)
            streaming_results.append(result)

        streaming_ema = np.array(streaming_results)

        # Compare results (ignore NaN values)
        valid_idx = ~np.isnan(bulk_ema)
        np.testing.assert_allclose(
            streaming_ema[valid_idx],
            bulk_ema[valid_idx],
            rtol=1e-10,
            err_msg="EMA streaming results don't match bulk calculation",
        )

    def test_rsi_streaming_accuracy(self, sample_data):
        """Test RSI streaming accuracy against bulk function."""
        close = sample_data["close"]
        window = 14

        # Calculate using bulk function
        bulk_rsi = rsi(close, window)

        # Calculate using streaming
        rsi_stream = RSI(window)
        streaming_results = []

        for price in close:
            result = rsi_stream.update(price)
            streaming_results.append(result)

        streaming_rsi = np.array(streaming_results)

        # Compare results (ignore NaN values)
        valid_idx = ~np.isnan(bulk_rsi)
        np.testing.assert_allclose(
            streaming_rsi[valid_idx],
            bulk_rsi[valid_idx],
            rtol=1e-8,  # Slightly more tolerance for RSI
            err_msg="RSI streaming results don't match bulk calculation",
        )

    def test_atr_streaming_accuracy(self, sample_data):
        """Test ATR streaming accuracy against bulk function."""
        high = sample_data["high"]
        low = sample_data["low"]
        close = sample_data["close"]
        window = 14

        # Calculate using bulk function
        bulk_atr = atr(high, low, close, window)

        # Calculate using streaming
        atr_stream = ATR(window)
        streaming_results = []

        for h, l, c in zip(high, low, close):
            result = atr_stream.update(h, l, c)
            streaming_results.append(result)

        streaming_atr = np.array(streaming_results)

        # Compare results (ignore NaN values)
        valid_idx = ~np.isnan(bulk_atr)
        np.testing.assert_allclose(
            streaming_atr[valid_idx],
            bulk_atr[valid_idx],
            rtol=1e-8,
            err_msg="ATR streaming results don't match bulk calculation",
        )

    def test_streaming_indicator_properties(self, sample_data):
        """Test streaming indicator properties and state management."""
        close = sample_data["close"]
        window = 20

        sma_stream = SMA(window)

        # Test initial state
        assert sma_stream.current_value != sma_stream.current_value  # NaN check
        assert not sma_stream.is_ready
        assert sma_stream.update_count == 0

        # Test during warm-up period
        for i in range(window - 1):
            result = sma_stream.update(close[i])
            assert result != result  # Should be NaN
            assert not sma_stream.is_ready
            assert sma_stream.update_count == i + 1

        # Test after warm-up
        result = sma_stream.update(close[window - 1])
        assert not (result != result)  # Should not be NaN
        assert sma_stream.is_ready
        assert sma_stream.update_count == window

        # Test reset
        sma_stream.reset()
        assert sma_stream.current_value != sma_stream.current_value  # NaN check
        assert not sma_stream.is_ready
        assert sma_stream.update_count == 0

    def test_macd_streaming_multiple_values(self, sample_data):
        """Test MACD streaming with multiple return values."""
        close = sample_data["close"]
        fast_period = 12
        slow_period = 26
        signal_period = 9

        # Calculate using bulk function
        bulk_macd, bulk_signal, bulk_hist = macd(
            close, fast_period, slow_period, signal_period
        )

        # Calculate using streaming
        macd_stream = MACD(fast_period, slow_period, signal_period)
        streaming_results = []

        for price in close:
            result = macd_stream.update(price)
            streaming_results.append(result)

        # Extract individual components
        streaming_macd = np.array([r["macd"] for r in streaming_results])
        streaming_signal = np.array([r["signal"] for r in streaming_results])
        streaming_hist = np.array([r["histogram"] for r in streaming_results])

        # Compare MACD line
        valid_idx = ~np.isnan(bulk_macd)
        np.testing.assert_allclose(
            streaming_macd[valid_idx],
            bulk_macd[valid_idx],
            rtol=1e-10,
            err_msg="MACD line streaming results don't match bulk calculation",
        )

        # Compare signal line
        valid_idx = ~np.isnan(bulk_signal)
        np.testing.assert_allclose(
            streaming_signal[valid_idx],
            bulk_signal[valid_idx],
            rtol=1e-10,
            err_msg="MACD signal streaming results don't match bulk calculation",
        )

    def test_bollinger_bands_streaming(self, sample_data):
        """Test Bollinger Bands streaming functionality."""
        close = sample_data["close"]
        window = 20
        std_dev = 2.0

        # Calculate using bulk function
        bulk_upper, bulk_middle, bulk_lower = bbands(close, window, std_dev)

        # Calculate using streaming
        bb_stream = BBands(window, std_dev)
        streaming_results = []

        for price in close:
            result = bb_stream.update(price)
            streaming_results.append(result)

        # Extract individual components
        streaming_upper = np.array([r["upper"] for r in streaming_results])
        streaming_middle = np.array([r["middle"] for r in streaming_results])
        streaming_lower = np.array([r["lower"] for r in streaming_results])

        # Compare results
        valid_idx = ~np.isnan(bulk_upper)
        np.testing.assert_allclose(
            streaming_upper[valid_idx],
            bulk_upper[valid_idx],
            rtol=1e-10,
            err_msg="Bollinger Bands upper streaming results don't match bulk calculation",
        )

        np.testing.assert_allclose(
            streaming_middle[valid_idx],
            bulk_middle[valid_idx],
            rtol=1e-10,
            err_msg="Bollinger Bands middle streaming results don't match bulk calculation",
        )

        np.testing.assert_allclose(
            streaming_lower[valid_idx],
            bulk_lower[valid_idx],
            rtol=1e-10,
            err_msg="Bollinger Bands lower streaming results don't match bulk calculation",
        )

    def test_streaming_performance_characteristics(self, sample_data):
        """Test that streaming indicators maintain O(1) performance."""
        close = sample_data["close"]
        window = 20

        sma_stream = SMA(window)

        # Test that buffer size is maintained
        for i, price in enumerate(close):
            sma_stream.update(price)

            # Buffer should never exceed window size
            assert len(sma_stream.buffer) <= window

            # After warm-up, buffer should be exactly window size
            if i >= window - 1:
                assert len(sma_stream.buffer) == window

    def test_streaming_edge_cases(self, sample_data):
        """Test streaming indicators with edge cases."""
        window = 20

        # Test with single value
        sma_stream = SMA(window)
        result = sma_stream.update(100.0)
        assert result != result  # Should be NaN for single value

        # Test with zero values
        sma_stream = SMA(1)  # Window of 1
        result = sma_stream.update(0.0)
        assert result == 0.0

        # Test with negative values
        result = sma_stream.update(-100.0)
        assert result == -100.0

    def test_streaming_thread_safety_properties(self, sample_data):
        """Test that streaming indicators maintain state correctly."""
        close = sample_data["close"]
        window = 20

        # Create two independent instances
        sma_stream1 = SMA(window)
        sma_stream2 = SMA(window)

        # Update them with same data
        for price in close:
            result1 = sma_stream1.update(price)
            result2 = sma_stream2.update(price)

            # Results should be identical
            if not np.isnan(result1) and not np.isnan(result2):
                assert abs(result1 - result2) < 1e-10

        # Final states should be identical
        assert sma_stream1.current_value == sma_stream2.current_value
        assert sma_stream1.is_ready == sma_stream2.is_ready
        assert sma_stream1.update_count == sma_stream2.update_count


if __name__ == "__main__":
    # Run tests
    test_instance = TestStreamingIndicators()

    # Generate sample data
    np.random.seed(42)
    sample_data = test_instance.sample_data()

    # Run individual tests
    print("🧪 Running streaming indicator tests...")

    print("✅ Testing SMA streaming accuracy...")
    test_instance.test_sma_streaming_accuracy(sample_data)

    print("✅ Testing EMA streaming accuracy...")
    test_instance.test_ema_streaming_accuracy(sample_data)

    print("✅ Testing RSI streaming accuracy...")
    test_instance.test_rsi_streaming_accuracy(sample_data)

    print("✅ Testing ATR streaming accuracy...")
    test_instance.test_atr_streaming_accuracy(sample_data)

    print("✅ Testing indicator properties...")
    test_instance.test_streaming_indicator_properties(sample_data)

    print("✅ Testing MACD multiple values...")
    test_instance.test_macd_streaming_multiple_values(sample_data)

    print("✅ Testing Bollinger Bands streaming...")
    test_instance.test_bollinger_bands_streaming(sample_data)

    print("✅ Testing performance characteristics...")
    test_instance.test_streaming_performance_characteristics(sample_data)

    print("✅ Testing edge cases...")
    test_instance.test_streaming_edge_cases(sample_data)

    print("✅ Testing thread safety properties...")
    test_instance.test_streaming_thread_safety_properties(sample_data)

    print("\n🎉 All streaming indicator tests passed!")
