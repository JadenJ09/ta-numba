"""
Tests for strategy functionality (bulk and streaming).
"""

import numpy as np
import pytest

import ta_numba


class TestBulkStrategy:
    """Test bulk strategy functionality."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        n = 100
        self.close = np.random.randn(n).cumsum() + 100
        self.high = self.close + np.random.rand(n) * 2
        self.low = self.close - np.random.rand(n) * 2
        self.volume = np.random.randint(1000, 10000, n)
    
    def test_momentum_strategy(self):
        """Test momentum indicators strategy."""
        results = ta_numba.bulk.strategy("momentum", close=self.close)
        
        # Check that we get momentum indicators
        assert 'rsi' in results
        assert 'roc' in results
        assert 'kama' in results
        
        # Check output shapes
        assert len(results['rsi']) == len(self.close)
        assert len(results['roc']) == len(self.close)
        
        # Check for multi-output indicators
        assert 'stoch_rsi_stoch_rsi' in results or 'stoch_rsi' in results
        assert 'ppo_ppo' in results or 'ppo' in results
    
    def test_trend_strategy(self):
        """Test trend indicators strategy."""
        results = ta_numba.bulk.strategy("trend", high=self.high, low=self.low, close=self.close)
        
        # Check that we get trend indicators
        assert 'sma' in results
        assert 'ema' in results
        assert 'wma' in results
        assert 'schaff_trend_cycle' in results
        
        # Check for multi-output indicators
        assert 'macd_macd' in results or 'macd' in results
        assert 'adx_adx' in results or 'adx' in results
    
    def test_volatility_strategy(self):
        """Test volatility indicators strategy."""
        results = ta_numba.bulk.strategy("volatility", high=self.high, low=self.low, close=self.close)
        
        # Check that we get volatility indicators
        assert 'ulcer_index' in results
        assert 'atr' in results
        
        # Check for multi-output indicators (Bollinger Bands, Donchian Channel)
        assert 'bollinger_upper' in results or 'bollinger' in results
        assert 'donchian_upper' in results or 'donchian' in results
    
    def test_volume_strategy(self):
        """Test volume indicators strategy."""
        results = ta_numba.bulk.strategy("volume", high=self.high, low=self.low, 
                               close=self.close, volume=self.volume)
        
        # Check that we get volume indicators
        assert 'obv' in results
        assert 'mfi' in results
        assert 'vwap' in results
    
    def test_others_strategy(self):
        """Test others (returns) indicators strategy."""
        results = ta_numba.bulk.strategy("others", close=self.close)
        
        # Check that we get return indicators
        assert 'daily_return' in results
        assert 'daily_log_return' in results
        assert 'cumulative_return' in results
    
    def test_all_strategy(self):
        """Test all indicators strategy."""
        results = ta_numba.bulk.strategy("all", high=self.high, low=self.low, 
                               close=self.close, volume=self.volume)
        
        # Should contain indicators from all categories
        # Check a few from each category
        assert 'rsi' in results  # momentum
        assert 'sma' in results  # trend
        assert 'atr' in results  # volatility  
        assert 'obv' in results  # volume
        assert 'daily_return' in results  # others
        
        # Should have many indicators
        assert len(results) > 20
    
    def test_custom_parameters(self):
        """Test custom parameters override."""
        results = ta_numba.bulk.strategy("momentum", close=self.close, n=21)
        
        # RSI should be calculated with period 21 instead of default 14
        # We can't easily test this directly, but we can ensure it runs
        assert 'rsi' in results
        assert not np.all(np.isnan(results['rsi']))
    
    def test_missing_data_handling(self):
        """Test handling of missing required data."""
        # Test with only close data for volume strategy
        results = ta_numba.bulk.strategy("volume", close=self.close)
        
        # Should skip indicators that require volume data
        # Should still work if there are indicators that only need close
        # Volume strategy mostly needs volume, so results might be limited
        assert isinstance(results, dict)
    
    def test_invalid_strategy(self):
        """Test invalid strategy name."""
        with pytest.raises(ValueError):
            ta_numba.bulk.strategy("invalid_strategy", close=self.close)


class TestStreamingStrategy:
    """Test streaming strategy functionality."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.prices = np.random.randn(50).cumsum() + 100
        self.highs = self.prices + np.random.rand(50) * 2
        self.lows = self.prices - np.random.rand(50) * 2
        self.volumes = np.random.randint(1000, 10000, 50)
    
    def test_create_momentum_strategy(self):
        """Test creating momentum streaming strategy."""
        strategy_manager = ta_numba.stream.create_strategy("momentum")
        
        # Check that indicators were created
        indicator_names = strategy_manager.get_indicator_names()
        assert 'rsi' in indicator_names
        assert 'roc' in indicator_names
        assert 'kama' in indicator_names
    
    def test_create_trend_strategy(self):
        """Test creating trend streaming strategy."""
        strategy_manager = ta_numba.stream.create_strategy("trend")
        
        # Check that indicators were created
        indicator_names = strategy_manager.get_indicator_names()
        assert 'sma' in indicator_names
        assert 'ema' in indicator_names
        assert 'wma' in indicator_names
    
    def test_create_all_strategy(self):
        """Test creating all indicators streaming strategy."""
        strategy_manager = ta_numba.stream.create_strategy("all")
        
        # Should have many indicators
        indicator_names = strategy_manager.get_indicator_names()
        assert len(indicator_names) > 15
        
        # Should have indicators from different categories
        assert any('rsi' in name or 'momentum' in name for name in indicator_names)
        assert any('sma' in name or 'trend' in name for name in indicator_names)
    
    def test_streaming_updates(self):
        """Test updating streaming strategy with price data."""
        strategy_manager = ta_numba.stream.create_strategy("momentum")
        
        # Update with some prices
        results_history = []
        for i, price in enumerate(self.prices[:20]):
            results = strategy_manager.update(close=price)
            results_history.append(results)
            
            # Check that results is a dictionary
            assert isinstance(results, dict)
            
            # After sufficient updates, some indicators should be ready
            if i > 15:  # After enough data points
                ready_status = strategy_manager.get_ready_status()
                # At least some indicators should be ready
                assert any(ready_status.values())
    
    def test_get_current_values(self):
        """Test getting current values without updating."""
        strategy_manager = ta_numba.stream.create_strategy("trend", n=10)

        # Update with some data
        for price in self.prices[:15]:
            strategy_manager.update(close=price)
        
        # Get current values
        current_values = strategy_manager.get_current_values()
        assert isinstance(current_values, dict)
        assert len(current_values) > 0
    
    def test_ready_status(self):
        """Test getting ready status of indicators."""
        strategy_manager = ta_numba.stream.create_strategy("momentum", n=5)
        
        # Initially, indicators should not be ready
        ready_status = strategy_manager.get_ready_status()
        assert all(status == False for status in ready_status.values())
        
        # After enough updates, some should be ready
        for price in self.prices[:10]:
            strategy_manager.update(close=price)
        
        ready_status = strategy_manager.get_ready_status()
        # Some indicators should now be ready
        assert any(ready_status.values())
    
    def test_reset_all(self):
        """Test resetting all indicators."""
        strategy_manager = ta_numba.stream.create_strategy("trend", n=5)
        
        # Update with data
        for price in self.prices[:10]:
            strategy_manager.update(close=price)
        
        # Some indicators should be ready
        ready_status = strategy_manager.get_ready_status()
        assert any(ready_status.values())
        
        # Reset all
        strategy_manager.reset_all()
        
        # Should not be ready anymore
        ready_status = strategy_manager.get_ready_status()
        assert all(status == False for status in ready_status.values())
    
    def test_custom_parameters(self):
        """Test custom parameters in streaming strategy."""
        # Create strategy with custom window
        strategy_manager = ta_numba.stream.create_strategy("momentum", n=21)
        
        # Should create indicators with custom window
        rsi_indicator = strategy_manager.get_indicator('rsi')
        if rsi_indicator:
            # Check if window parameter was applied (if accessible)
            assert hasattr(rsi_indicator, 'window')
    
    def test_hlcv_updates(self):
        """Test updates with high, low, close, volume data."""
        strategy_manager = ta_numba.stream.create_strategy("volatility")
        
        # Update with HLCV data
        for i in range(20):
            results = strategy_manager.update(
                high=self.highs[i],
                low=self.lows[i], 
                close=self.prices[i],
                volume=self.volumes[i]
            )
            
            assert isinstance(results, dict)
    
    def test_invalid_strategy_name(self):
        """Test invalid strategy name."""
        with pytest.raises(ValueError):
            ta_numba.stream.create_strategy("invalid_strategy")
    
    def test_Strategy_class_alias(self):
        """Test that Strategy class alias works."""
        # Test that we can use ta_numba.stream.Strategy directly
        strategy_manager = ta_numba.stream.Strategy("momentum")
        assert isinstance(strategy_manager, ta_numba.stream.Strategy)
        
        # Should work the same as create_strategy
        indicator_names = strategy_manager.get_indicator_names()
        assert 'rsi' in indicator_names


class TestStrategyIntegration:
    """Test integration between bulk and streaming strategies."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.close = np.random.randn(100).cumsum() + 100
    
    def test_bulk_vs_streaming_consistency(self):
        """Test that bulk and streaming give similar results for basic indicators."""
        # Get bulk results
        bulk_results = ta_numba.bulk.strategy("momentum", close=self.close, n=14)
        
        # Get streaming results
        strategy_manager = ta_numba.stream.create_strategy("momentum", n=14)
        streaming_results = []
        
        for price in self.close:
            result = strategy_manager.update(close=price)
            streaming_results.append(result)
        
        # Compare RSI values (if available)
        if 'rsi' in bulk_results and len(streaming_results) > 20:
            bulk_rsi = bulk_results['rsi']
            streaming_rsi = [r.get('rsi', np.nan) for r in streaming_results]
            
            # Filter out NaN values and compare
            valid_indices = ~np.isnan(bulk_rsi) & ~np.isnan(streaming_rsi)
            if np.any(valid_indices):
                bulk_valid = bulk_rsi[valid_indices]
                streaming_valid = np.array(streaming_rsi)[valid_indices]
                
                # Should be reasonably close (allowing for small numerical differences)
                # For indicators ready after the warmup period
                if len(bulk_valid) > 20:
                    np.testing.assert_allclose(
                        bulk_valid[-20:], streaming_valid[-20:], 
                        rtol=1e-3, atol=1e-3
                    )