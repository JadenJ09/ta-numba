"""
Examples demonstrating the strategy functionality in ta-numba.

This module shows how to use the new strategy features to calculate 
multiple indicators at once in both bulk and streaming modes.
"""

import numpy as np
import ta_numba.bulk as bulk
import ta_numba.stream as stream
import ta_numba.warmup

# Warm up for optimal performance
ta_numba.warmup.warmup_all()

def generate_sample_data(n=1000):
    """Generate sample OHLCV data for examples."""
    np.random.seed(42)
    
    # Generate close prices with random walk
    close = np.random.randn(n).cumsum() + 100
    
    # Generate high/low with realistic spreads
    high = close + np.random.rand(n) * 3
    low = close - np.random.rand(n) * 3
    
    # Generate volume
    volume = np.random.randint(1000, 50000, n)
    
    return high, low, close, volume


def bulk_strategy_examples():
    """Demonstrate bulk strategy functionality."""
    print("=" * 60)
    print("BULK STRATEGY EXAMPLES")
    print("=" * 60)
    
    # Generate sample data
    high, low, close, volume = generate_sample_data(1000)
    
    # Example 1: Calculate all momentum indicators
    print("\n1. Momentum Indicators Strategy")
    print("-" * 40)
    momentum_results = bulk.strategy("momentum", close=close)
    print(f"Calculated {len(momentum_results)} momentum indicators:")
    for name, values in momentum_results.items():
        non_nan_count = np.sum(~np.isnan(values))
        print(f"  {name}: {non_nan_count} valid values")
    
    # Example 2: Calculate all trend indicators
    print("\n2. Trend Indicators Strategy")
    print("-" * 40)
    trend_results = bulk.strategy("trend", high=high, low=low, close=close)
    print(f"Calculated {len(trend_results)} trend indicators:")
    for name, values in trend_results.items():
        non_nan_count = np.sum(~np.isnan(values))
        print(f"  {name}: {non_nan_count} valid values")
    
    # Example 3: Calculate all volatility indicators
    print("\n3. Volatility Indicators Strategy")
    print("-" * 40)
    volatility_results = bulk.strategy("volatility", high=high, low=low, close=close)
    print(f"Calculated {len(volatility_results)} volatility indicators:")
    for name, values in volatility_results.items():
        non_nan_count = np.sum(~np.isnan(values))
        print(f"  {name}: {non_nan_count} valid values")
    
    # Example 4: Calculate all volume indicators
    print("\n4. Volume Indicators Strategy")
    print("-" * 40)
    volume_results = bulk.strategy("volume", high=high, low=low, close=close, volume=volume)
    print(f"Calculated {len(volume_results)} volume indicators:")
    for name, values in volume_results.items():
        non_nan_count = np.sum(~np.isnan(values))
        print(f"  {name}: {non_nan_count} valid values")
    
    # Example 5: Calculate all indicators at once
    print("\n5. All Indicators Strategy")
    print("-" * 40)
    all_results = bulk.strategy("all", high=high, low=low, close=close, volume=volume)
    print(f"Calculated {len(all_results)} total indicators!")
    
    # Group by category for display
    categories = {
        'Trend': [name for name in all_results.keys() if any(trend in name.lower() for trend in ['sma', 'ema', 'wma', 'macd', 'adx', 'trix', 'dpo', 'aroon', 'parabolic', 'vortex', 'cci', 'schaff'])],
        'Momentum': [name for name in all_results.keys() if any(mom in name.lower() for mom in ['rsi', 'stoch', 'williams', 'roc', 'kama', 'tsi', 'awesome', 'ultimate', 'ppo'])],
        'Volatility': [name for name in all_results.keys() if any(vol in name.lower() for vol in ['atr', 'bollinger', 'keltner', 'donchian', 'ulcer', 'std', 'variance'])],
        'Volume': [name for name in all_results.keys() if any(vol in name.lower() for vol in ['obv', 'mfi', 'adi', 'cmf', 'force', 'emv', 'vpt', 'nvi', 'vwap', 'vwema'])],
        'Others': [name for name in all_results.keys() if any(other in name.lower() for other in ['return', 'cumulative', 'compound'])]
    }
    
    for category, indicators in categories.items():
        if indicators:
            print(f"  {category}: {len(indicators)} indicators")
    
    # Example 6: Custom parameters
    print("\n6. Custom Parameters Example")
    print("-" * 40)
    custom_results = bulk.strategy("momentum", close=close, n=21)  # Use 21-period instead of default
    print(f"Momentum indicators with custom period (21): {len(custom_results)} indicators")
    
    # Show some actual values
    if 'rsi' in custom_results:
        rsi_values = custom_results['rsi']
        valid_rsi = rsi_values[~np.isnan(rsi_values)]
        if len(valid_rsi) > 0:
            print(f"  RSI (21-period) - Last 5 values: {valid_rsi[-5:]}")


def streaming_strategy_examples():
    """Demonstrate streaming strategy functionality."""
    print("\n\n" + "=" * 60)
    print("STREAMING STRATEGY EXAMPLES")
    print("=" * 60)
    
    # Generate sample data
    high, low, close, volume = generate_sample_data(100)
    
    # Example 1: Create momentum streaming strategy
    print("\n1. Momentum Streaming Strategy")
    print("-" * 40)
    momentum_strategy = stream.create_strategy("momentum")
    print(f"Created momentum strategy with {len(momentum_strategy.get_indicator_names())} indicators:")
    print(f"  Indicators: {', '.join(momentum_strategy.get_indicator_names())}")
    
    # Update with first 20 prices and show progress
    print("\n  Updating with price data...")
    for i in range(20):
        results = momentum_strategy.update(close=close[i])
        
        if i in [5, 10, 15, 19]:  # Show progress at intervals
            ready_count = sum(momentum_strategy.get_ready_status().values())
            print(f"    After {i+1} updates: {ready_count} indicators ready")
    
    # Show final results
    final_results = momentum_strategy.get_current_values()
    ready_status = momentum_strategy.get_ready_status()
    print(f"\n  Final state: {sum(ready_status.values())} indicators ready")
    for name, is_ready in ready_status.items():
        if is_ready:
            value = final_results.get(name, np.nan)
            if not np.isnan(value):
                print(f"    {name}: {value:.4f}")
    
    # Example 2: All indicators streaming strategy
    print("\n2. All Indicators Streaming Strategy")
    print("-" * 40)
    all_strategy = stream.create_strategy("all")
    print(f"Created all-indicators strategy with {len(all_strategy.get_indicator_names())} indicators")
    
    # Update with some data
    for i in range(25):
        all_strategy.update(
            high=high[i], 
            low=low[i], 
            close=close[i], 
            volume=volume[i]
        )
    
    # Show ready indicators
    ready_status = all_strategy.get_ready_status()
    ready_indicators = [name for name, is_ready in ready_status.items() if is_ready]
    print(f"  After 25 updates: {len(ready_indicators)} indicators ready")
    print(f"  Ready indicators: {', '.join(ready_indicators[:10])}...")  # Show first 10
    
    # Example 3: Real-time simulation
    print("\n3. Real-time Trading Simulation")
    print("-" * 40)
    
    # Create a focused strategy for trading
    trading_strategy = stream.create_strategy("trend", window=10)  # Shorter window for faster response
    
    print("  Simulating real-time price feed...")
    signals = []
    
    for i, price in enumerate(close[:50]):
        results = trading_strategy.update(close=price)
        
        # Simple trading logic example
        if trading_strategy.get_ready_status().get('sma', False) and trading_strategy.get_ready_status().get('ema', False):
            sma_value = results.get('sma', np.nan)
            ema_value = results.get('ema', np.nan)
            
            if not np.isnan(sma_value) and not np.isnan(ema_value):
                if ema_value > sma_value:
                    signal = "BUY"
                elif ema_value < sma_value:
                    signal = "SELL"
                else:
                    signal = "HOLD"
                
                signals.append((i, price, signal, ema_value, sma_value))
    
    # Show some trading signals
    print(f"  Generated {len(signals)} trading signals")
    print(f"  Sample signals (tick, price, signal, EMA, SMA):")
    for signal in signals[-5:]:  # Show last 5 signals
        tick, price, sig, ema, sma = signal
        print(f"    Tick {tick}: Price={price:.2f}, Signal={sig}, EMA={ema:.2f}, SMA={sma:.2f}")
    
    # Example 4: Strategy with custom parameters
    print("\n4. Custom Parameters Streaming Strategy")
    print("-" * 40)
    custom_strategy = stream.create_strategy("volatility", window=15, std_dev=2.5)
    print("  Created volatility strategy with custom parameters (window=15, std_dev=2.5)")
    
    # Update and show results
    for price in close[:30]:
        custom_strategy.update(close=price)
    
    ready_indicators = [name for name, is_ready in custom_strategy.get_ready_status().items() if is_ready]
    print(f"  Ready indicators: {', '.join(ready_indicators)}")


def performance_comparison():
    """Compare bulk vs streaming performance."""
    print("\n\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    
    import time
    
    # Generate larger dataset
    high, low, close, volume = generate_sample_data(10000)
    
    # Bulk processing timing
    print("\n1. Bulk Processing Performance")
    print("-" * 40)
    start_time = time.time()
    bulk_results = bulk.strategy("momentum", close=close)
    bulk_time = time.time() - start_time
    print(f"  Calculated {len(bulk_results)} momentum indicators")
    print(f"  Time taken: {bulk_time:.4f} seconds")
    print(f"  Speed: {len(close)/bulk_time:.0f} data points per second")
    
    # Streaming processing timing
    print("\n2. Streaming Processing Performance")
    print("-" * 40)
    streaming_strategy = stream.create_strategy("momentum")
    
    start_time = time.time()
    for price in close:
        streaming_strategy.update(close=price)
    streaming_time = time.time() - start_time
    
    print(f"  Updated {len(streaming_strategy.get_indicator_names())} indicators")
    print(f"  Time taken: {streaming_time:.4f} seconds")
    print(f"  Speed: {len(close)/streaming_time:.0f} updates per second")
    
    print(f"\n  Bulk vs Streaming speed ratio: {streaming_time/bulk_time:.2f}x")
    print("  Note: Streaming maintains constant memory and provides O(1) updates")


def advanced_usage_examples():
    """Show advanced usage patterns."""
    print("\n\n" + "=" * 60)
    print("ADVANCED USAGE EXAMPLES")
    print("=" * 60)
    
    high, low, close, volume = generate_sample_data(200)
    
    # Example 1: Multiple strategy managers
    print("\n1. Multiple Strategy Managers")
    print("-" * 40)
    short_term = stream.create_strategy("trend", window=5)
    medium_term = stream.create_strategy("trend", window=20)
    long_term = stream.create_strategy("trend", window=50)
    
    print("  Created 3 strategy managers with different timeframes")
    
    # Update all with same data
    for i, price in enumerate(close[:100]):
        short_results = short_term.update(close=price)
        medium_results = medium_term.update(close=price)
        long_results = long_term.update(close=price)
        
        # Show crossover signals
        if i > 50:  # After sufficient warmup
            short_sma = short_results.get('sma', np.nan)
            medium_sma = medium_results.get('sma', np.nan)
            long_sma = long_results.get('sma', np.nan)
            
            if not any(np.isnan([short_sma, medium_sma, long_sma])):
                if short_sma > medium_sma > long_sma:
                    print(f"    Tick {i}: Strong BULLISH signal (Price: {price:.2f})")
                elif short_sma < medium_sma < long_sma:
                    print(f"    Tick {i}: Strong BEARISH signal (Price: {price:.2f})")
    
    # Example 2: Strategy reset and reuse
    print("\n2. Strategy Reset and Reuse")
    print("-" * 40)
    reusable_strategy = stream.create_strategy("momentum", window=10)
    
    # First run
    for price in close[:30]:
        reusable_strategy.update(close=price)
    
    first_run_ready = sum(reusable_strategy.get_ready_status().values())
    print(f"  First run: {first_run_ready} indicators ready")
    
    # Reset and second run
    reusable_strategy.reset_all()
    after_reset_ready = sum(reusable_strategy.get_ready_status().values())
    print(f"  After reset: {after_reset_ready} indicators ready")
    
    # Second run with different data
    for price in close[100:130]:
        reusable_strategy.update(close=price)
    
    second_run_ready = sum(reusable_strategy.get_ready_status().values())
    print(f"  Second run: {second_run_ready} indicators ready")
    
    # Example 3: Accessing individual indicators
    print("\n3. Accessing Individual Indicators")
    print("-" * 40)
    mixed_strategy = stream.create_strategy("all")
    
    # Update with some data
    for price in close[:50]:
        mixed_strategy.update(close=price)
    
    # Access specific indicators
    rsi_indicator = mixed_strategy.get_indicator('rsi')
    sma_indicator = mixed_strategy.get_indicator('sma')
    
    print(f"  Accessed individual indicators:")
    if rsi_indicator:
        print(f"    RSI indicator: {type(rsi_indicator).__name__}")
        print(f"    RSI current value: {getattr(rsi_indicator, 'current_value', 'N/A')}")
    if sma_indicator:
        print(f"    SMA indicator: {type(sma_indicator).__name__}")
        print(f"    SMA current value: {getattr(sma_indicator, 'current_value', 'N/A')}")


def main():
    """Run all examples."""
    print("TA-NUMBA STRATEGY USAGE EXAMPLES")
    print("This demonstrates the new strategy functionality for calculating")
    print("multiple indicators at once in both bulk and streaming modes.")
    
    # Run all example sections
    bulk_strategy_examples()
    streaming_strategy_examples()
    performance_comparison()
    advanced_usage_examples()
    
    print("\n\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("✅ Bulk strategies: Calculate multiple indicators efficiently on historical data")
    print("✅ Streaming strategies: Real-time updates with O(1) performance")
    print("✅ Flexible API: Support for all indicator categories")
    print("✅ Custom parameters: Override defaults easily")
    print("✅ Production ready: Reset, reuse, and manage multiple strategies")
    print("\nSee the code examples above for detailed usage patterns!")


if __name__ == "__main__":
    main()