"""
JIT Pre-compilation Module for ta-numba
Provides efficient warm-up strategies for all JIT-compiled functions.
"""

import numpy as np
import time
import threading
import warnings
from typing import Dict, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor
import os

# Suppress numba warnings during warmup
warnings.filterwarnings('ignore', category=UserWarning)

class JITWarmupManager:
    """Manages JIT compilation warm-up for ta-numba functions."""
    
    def __init__(self):
        self.warmed_functions: Dict[str, bool] = {}
        self.warmup_times: Dict[str, float] = {}
        self.lock = threading.Lock()
        self._warmup_data = None
        
    def _generate_warmup_data(self, size: int = 100) -> Dict[str, np.ndarray]:
        """Generate test data for warm-up."""
        if self._warmup_data is not None:
            return self._warmup_data
            
        np.random.seed(42)  # Consistent data for reproducible warm-up
        
        # Generate realistic market data
        base_price = 100.0
        prices = [base_price]
        for i in range(1, size):
            change = np.random.normal(0.0001, 0.015)
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 0.01))
        
        close = np.array(prices)
        high = close * np.random.uniform(1.000, 1.015, size)
        low = close * np.random.uniform(0.985, 1.000, size)
        open_price = close * np.random.uniform(0.995, 1.005, size)
        volume = np.random.randint(1000, 10000, size).astype(np.float64)
        
        self._warmup_data = {
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }
        
        return self._warmup_data
    
    def warmup_function(self, func: Callable, func_name: str, 
                       args: tuple, kwargs: dict = None) -> float:
        """Warm up a single function."""
        if kwargs is None:
            kwargs = {}
            
        if func_name in self.warmed_functions:
            return self.warmup_times.get(func_name, 0.0)
        
        try:
            start_time = time.perf_counter()
            _ = func(*args, **kwargs)
            warmup_time = time.perf_counter() - start_time
            
            with self.lock:
                self.warmed_functions[func_name] = True
                self.warmup_times[func_name] = warmup_time
            
            return warmup_time
            
        except Exception as e:
            # Silent failure - function might not be available
            return 0.0
    
    def warmup_trend_indicators(self, data: Dict[str, np.ndarray], 
                              show_progress: bool = True) -> Dict[str, float]:
        """Warm up trend indicators."""
        results = {}
        
        # Import trend functions
        try:
            from ta_numba.trend import (
                sma_numba, ema_numba, weighted_moving_average, macd_numba,
                adx_numba, vortex_indicator_numba, trix_numba, cci_numba,
                dpo_numba, aroon_numba, parabolic_sar_numba
            )
            
            trend_functions = [
                (sma_numba, "sma_numba", (data['close'], 20)),
                (ema_numba, "ema_numba", (data['close'], 20)),
                (weighted_moving_average, "weighted_moving_average", (data['close'], 20)),
                (macd_numba, "macd_numba", (data['close'], 12, 26, 9)),
                (adx_numba, "adx_numba", (data['high'], data['low'], data['close'], 14)),
                (vortex_indicator_numba, "vortex_indicator_numba", (data['high'], data['low'], data['close'], 14)),
                (trix_numba, "trix_numba", (data['close'], 14)),
                (cci_numba, "cci_numba", (data['high'], data['low'], data['close'], 20)),
                (dpo_numba, "dpo_numba", (data['close'], 20)),
                (aroon_numba, "aroon_numba", (data['high'], data['low'], 14)),
                (parabolic_sar_numba, "parabolic_sar_numba", (data['high'], data['low'], 0.02, 0.2)),
            ]
            
            for func, name, args in trend_functions:
                warmup_time = self.warmup_function(func, name, args)
                results[name] = warmup_time
                if show_progress and warmup_time > 0:
                    print(f"  ✅ {name}: {warmup_time:.3f}s")
                    
        except ImportError:
            pass
            
        return results
    
    def warmup_momentum_indicators(self, data: Dict[str, np.ndarray], 
                                 show_progress: bool = True) -> Dict[str, float]:
        """Warm up momentum indicators."""
        results = {}
        
        try:
            from ta_numba.momentum import (
                relative_strength_index_numba, stochastic_oscillator_numba,
                williams_r_numba, rate_of_change_numba, ultimate_oscillator_numba,
                stochastic_rsi_numba, true_strength_index_numba, awesome_oscillator_numba,
                kaufmans_adaptive_moving_average_numba, percentage_price_oscillator_numba
            )
            
            momentum_functions = [
                (relative_strength_index_numba, "relative_strength_index_numba", (data['close'], 14)),
                (stochastic_oscillator_numba, "stochastic_oscillator_numba", (data['high'], data['low'], data['close'], 14)),
                (williams_r_numba, "williams_r_numba", (data['high'], data['low'], data['close'], 14)),
                (rate_of_change_numba, "rate_of_change_numba", (data['close'], 10)),
                (ultimate_oscillator_numba, "ultimate_oscillator_numba", (data['high'], data['low'], data['close'], 7, 14, 28)),
                (stochastic_rsi_numba, "stochastic_rsi_numba", (data['close'], 14)),
                (true_strength_index_numba, "true_strength_index_numba", (data['close'], 25, 13)),
                (awesome_oscillator_numba, "awesome_oscillator_numba", (data['high'], data['low'])),
                (kaufmans_adaptive_moving_average_numba, "kaufmans_adaptive_moving_average_numba", (data['close'], 10)),
                (percentage_price_oscillator_numba, "percentage_price_oscillator_numba", (data['close'], 12, 26)),
            ]
            
            for func, name, args in momentum_functions:
                warmup_time = self.warmup_function(func, name, args)
                results[name] = warmup_time
                if show_progress and warmup_time > 0:
                    print(f"  ✅ {name}: {warmup_time:.3f}s")
                    
        except ImportError:
            pass
            
        return results
    
    def warmup_volatility_indicators(self, data: Dict[str, np.ndarray], 
                                   show_progress: bool = True) -> Dict[str, float]:
        """Warm up volatility indicators."""
        results = {}
        
        try:
            from ta_numba.volatility import (
                average_true_range_numba, bollinger_bands_numba, keltner_channel_numba,
                donchian_channel_numba, ulcer_index_numba
            )
            
            volatility_functions = [
                (average_true_range_numba, "average_true_range_numba", (data['high'], data['low'], data['close'], 14)),
                (bollinger_bands_numba, "bollinger_bands_numba", (data['close'], 20, 2.0)),
                (keltner_channel_numba, "keltner_channel_numba", (data['high'], data['low'], data['close'], 20)),
                (donchian_channel_numba, "donchian_channel_numba", (data['high'], data['low'], 20)),
                (ulcer_index_numba, "ulcer_index_numba", (data['close'], 14)),
            ]
            
            for func, name, args in volatility_functions:
                warmup_time = self.warmup_function(func, name, args)
                results[name] = warmup_time
                if show_progress and warmup_time > 0:
                    print(f"  ✅ {name}: {warmup_time:.3f}s")
                    
        except ImportError:
            pass
            
        return results
    
    def warmup_volume_indicators(self, data: Dict[str, np.ndarray], 
                               show_progress: bool = True) -> Dict[str, float]:
        """Warm up volume indicators."""
        results = {}
        
        try:
            from ta_numba.volume import (
                money_flow_index_numba, acc_dist_index_numba, on_balance_volume_numba,
                chaikin_money_flow_numba, force_index_numba, ease_of_movement_numba,
                volume_price_trend_numba, negative_volume_index_numba,
                volume_weighted_average_price_numba, volume_weighted_exponential_moving_average_numba
            )
            
            volume_functions = [
                (money_flow_index_numba, "money_flow_index_numba", (data['high'], data['low'], data['close'], data['volume'], 14)),
                (acc_dist_index_numba, "acc_dist_index_numba", (data['high'], data['low'], data['close'], data['volume'])),
                (on_balance_volume_numba, "on_balance_volume_numba", (data['close'], data['volume'])),
                (chaikin_money_flow_numba, "chaikin_money_flow_numba", (data['high'], data['low'], data['close'], data['volume'], 20)),
                (force_index_numba, "force_index_numba", (data['close'], data['volume'], 13)),
                (ease_of_movement_numba, "ease_of_movement_numba", (data['high'], data['low'], data['volume'], 14)),
                (volume_price_trend_numba, "volume_price_trend_numba", (data['close'], data['volume'])),
                (negative_volume_index_numba, "negative_volume_index_numba", (data['close'], data['volume'])),
                (volume_weighted_average_price_numba, "volume_weighted_average_price_numba", (data['high'], data['low'], data['close'], data['volume'])),
                (volume_weighted_exponential_moving_average_numba, "volume_weighted_exponential_moving_average_numba", (data['close'], data['volume'], 20)),
            ]
            
            for func, name, args in volume_functions:
                warmup_time = self.warmup_function(func, name, args)
                results[name] = warmup_time
                if show_progress and warmup_time > 0:
                    print(f"  ✅ {name}: {warmup_time:.3f}s")
                    
        except ImportError:
            pass
            
        return results
    
    def warmup_others_indicators(self, data: Dict[str, np.ndarray], 
                               show_progress: bool = True) -> Dict[str, float]:
        """Warm up others indicators."""
        results = {}
        
        try:
            from ta_numba.others import (
                daily_return_numba, daily_log_return_numba, cumulative_return_numba,
                compound_log_return_numba
            )
            
            others_functions = [
                (daily_return_numba, "daily_return_numba", (data['close'],)),
                (daily_log_return_numba, "daily_log_return_numba", (data['close'],)),
                (cumulative_return_numba, "cumulative_return_numba", (data['close'],)),
                (compound_log_return_numba, "compound_log_return_numba", (data['close'],)),
            ]
            
            for func, name, args in others_functions:
                warmup_time = self.warmup_function(func, name, args)
                results[name] = warmup_time
                if show_progress and warmup_time > 0:
                    print(f"  ✅ {name}: {warmup_time:.3f}s")
                    
        except ImportError:
            pass
            
        return results
    
    def warmup_all(self, show_progress: bool = True, parallel: bool = False) -> Dict[str, float]:
        """Warm up all indicators."""
        if show_progress:
            print("🔥 Warming up ta-numba JIT functions...")
        
        start_time = time.perf_counter()
        data = self._generate_warmup_data()
        
        all_results = {}
        
        if parallel:
            # Parallel warmup
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                futures.append(executor.submit(self.warmup_trend_indicators, data, show_progress))
                futures.append(executor.submit(self.warmup_momentum_indicators, data, show_progress))
                futures.append(executor.submit(self.warmup_volatility_indicators, data, show_progress))
                futures.append(executor.submit(self.warmup_volume_indicators, data, show_progress))
                futures.append(executor.submit(self.warmup_others_indicators, data, show_progress))
                
                for future in futures:
                    try:
                        results = future.result()
                        all_results.update(results)
                    except Exception:
                        pass
        else:
            # Sequential warmup
            all_results.update(self.warmup_trend_indicators(data, show_progress))
            all_results.update(self.warmup_momentum_indicators(data, show_progress))
            all_results.update(self.warmup_volatility_indicators(data, show_progress))
            all_results.update(self.warmup_volume_indicators(data, show_progress))
            all_results.update(self.warmup_others_indicators(data, show_progress))
        
        total_time = time.perf_counter() - start_time
        
        if show_progress:
            successful_warmups = sum(1 for t in all_results.values() if t > 0)
            print(f"✅ Warmed up {successful_warmups} functions in {total_time:.3f}s")
        
        return all_results
    
    def get_status(self) -> Dict[str, any]:
        """Get warmup status."""
        return {
            'warmed_functions': len(self.warmed_functions),
            'total_warmup_time': sum(self.warmup_times.values()),
            'functions': list(self.warmed_functions.keys()),
            'timings': dict(self.warmup_times)
        }


# Global warmup manager instance
_warmup_manager = JITWarmupManager()

# Public API functions
def warmup_all(show_progress: bool = True, parallel: bool = False) -> Dict[str, float]:
    """
    Warm up all ta-numba JIT functions.
    
    Args:
        show_progress: Show progress messages
        parallel: Use parallel warmup (faster but uses more CPU)
    
    Returns:
        Dictionary with warmup times for each function
    """
    return _warmup_manager.warmup_all(show_progress, parallel)

def warmup_essential(show_progress: bool = True) -> Dict[str, float]:
    """
    Warm up essential indicators only (SMA, EMA, RSI, ATR, BBands).
    
    Args:
        show_progress: Show progress messages
    
    Returns:
        Dictionary with warmup times for each function
    """
    if show_progress:
        print("🔥 Warming up essential ta-numba functions...")
    
    data = _warmup_manager._generate_warmup_data()
    results = {}
    
    # Essential functions
    try:
        from ta_numba.trend import sma_numba, ema_numba
        from ta_numba.momentum import relative_strength_index_numba
        from ta_numba.volatility import average_true_range_numba, bollinger_bands_numba
        
        essential_functions = [
            (sma_numba, "sma_numba", (data['close'], 20)),
            (ema_numba, "ema_numba", (data['close'], 20)),
            (relative_strength_index_numba, "relative_strength_index_numba", (data['close'], 14)),
            (average_true_range_numba, "average_true_range_numba", (data['high'], data['low'], data['close'], 14)),
            (bollinger_bands_numba, "bollinger_bands_numba", (data['close'], 20, 2.0)),
        ]
        
        for func, name, args in essential_functions:
            warmup_time = _warmup_manager.warmup_function(func, name, args)
            results[name] = warmup_time
            if show_progress and warmup_time > 0:
                print(f"  ✅ {name}: {warmup_time:.3f}s")
                
    except ImportError:
        pass
    
    return results

def warmup_background() -> threading.Thread:
    """
    Start background warmup in a separate thread.
    
    Returns:
        Thread object for monitoring
    """
    def background_warmup():
        warmup_all(show_progress=False, parallel=True)
    
    thread = threading.Thread(target=background_warmup, daemon=True)
    thread.start()
    return thread

def get_warmup_status() -> Dict[str, any]:
    """Get current warmup status."""
    return _warmup_manager.get_status()

def is_warmed_up() -> bool:
    """Check if any functions have been warmed up."""
    return len(_warmup_manager.warmed_functions) > 0

# Configuration
class WarmupConfig:
    """Configuration for automatic warmup behavior."""
    AUTO_WARMUP = True  # Auto-warmup on import
    WARMUP_ESSENTIAL_ONLY = True  # Only warmup essential functions
    SHOW_PROGRESS = False  # Show progress during auto-warmup
    PARALLEL_WARMUP = False  # Use parallel warmup

def auto_warmup():
    """Automatically warm up functions based on configuration."""
    if WarmupConfig.AUTO_WARMUP:
        if WarmupConfig.WARMUP_ESSENTIAL_ONLY:
            warmup_essential(WarmupConfig.SHOW_PROGRESS)
        else:
            warmup_all(WarmupConfig.SHOW_PROGRESS, WarmupConfig.PARALLEL_WARMUP)

# Auto-warmup on import (can be disabled)
if os.environ.get('TA_NUMBA_NO_AUTO_WARMUP') != '1':
    auto_warmup()