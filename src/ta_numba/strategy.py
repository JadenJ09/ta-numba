"""
Strategy module for calculating multiple indicators at once.

This module provides convenient functions to calculate all indicators or groups of indicators
in both bulk and streaming modes, similar to pandas-ta's strategy functionality.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Any
import warnings

# Import all indicator modules
from . import trend, momentum, volatility, volume, others


class IndicatorStrategy:
    """
    Main class for calculating multiple technical indicators at once.
    
    This class provides methods to calculate groups of indicators (all, trend, momentum, 
    volatility, volume) in bulk mode, returning results as a dictionary.
    """
    
    # Define indicator groups with their function names and input requirements
    INDICATOR_GROUPS = {
        'trend': {
            # Single input (close only)
            'close_only': [
                ('sma', 'sma_numba', {'n': 20}),
                ('ema', 'ema_numba', {'n': 20}), 
                ('wma', 'weighted_moving_average', {'n': 20}),
                ('trix', 'trix_numba', {'n': 14}),
                ('dpo', 'dpo_numba', {'n': 20}),
                ('schaff_trend_cycle', 'schaff_trend_cycle_numba', {'n_fast': 23, 'n_slow': 50, 'n_stoch': 10, 'n_smooth': 3}),
            ],
            # High, Low, Close inputs
            'hlc': [
                ('adx', 'adx_numba', {'n': 14}, ['adx', 'plus_di', 'minus_di']),
                ('vortex', 'vortex_indicator_numba', {'n': 14}, ['vi_plus', 'vi_minus']),
                ('cci', 'cci_numba', {'n': 20}),
                ('parabolic_sar', 'parabolic_sar_numba', {}),
            ],
            # High, Low only
            'hl': [
                ('mass_index', 'mass_index_numba', {'n_ema': 9, 'n_sum': 25}),
                ('aroon', 'aroon_numba', {'n': 25}, ['aroon_up', 'aroon_down']),
            ],
            # Multi-output indicators
            'multi_output': [
                ('macd', 'macd_numba', {'n_fast': 12, 'n_slow': 26, 'n_signal': 9}, ['macd', 'signal', 'histogram']),
                ('kst', 'kst_numba', {}, ['kst', 'signal']),
                ('ichimoku', 'ichimoku_numba', {'n1': 9, 'n2': 26, 'n3': 52}, 
                 ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span']),
            ]
        },
        
        'momentum': {
            'close_only': [
                ('rsi', 'relative_strength_index_numba', {'n': 14}),
                ('roc', 'rate_of_change_numba', {'n': 12}),
                ('kama', 'kaufmans_adaptive_moving_average_numba', {'n': 10}),
            ],
            'hlc': [
                ('stoch', 'stochastic_oscillator_numba', {'n': 14, 'd': 3}, ['percent_k', 'percent_d']),
                ('williams_r', 'williams_r_numba', {'n': 14}),
                ('ultimate_osc', 'ultimate_oscillator_numba', {'n1': 7, 'n2': 14, 'n3': 28}),
            ],
            'hl': [
                ('awesome_osc', 'awesome_oscillator_numba', {'n1': 5, 'n2': 34}),
            ],
            'multi_output': [
                ('stoch_rsi', 'stochastic_rsi_numba', {'n': 14, 'k': 3, 'd': 3}, 
                 ['stoch_rsi', 'stoch_k', 'stoch_d']),
                ('tsi', 'true_strength_index_numba', {'r': 25, 's': 13}),
                ('ppo', 'percentage_price_oscillator_numba', {'n_fast': 12, 'n_slow': 26, 'n_signal': 9},
                 ['ppo', 'signal', 'histogram']),
            ],
            'volume_only': [
                ('pvo', 'percentage_volume_oscillator_numba', {'n_fast': 12, 'n_slow': 26, 'n_signal': 9},
                 ['pvo', 'signal', 'histogram']),
            ]
        },
        
        'volatility': {
            'close_only': [
                ('ulcer_index', 'ulcer_index_numba', {'n': 14}),
            ],
            'hlc': [
                ('atr', 'average_true_range_numba', {'n': 14}),
                ('keltner', 'keltner_channel_numba', {'n_ema': 20, 'n_atr': 10, 'k': 2.0},
                 ['high_band', 'middle_line', 'low_band']),
            ],
            # High, Low only
            'hl': [
                ('donchian', 'donchian_channel_numba', {'n': 20}, ['upper', 'middle', 'lower']),
            ],
            'multi_output': [
                ('bollinger', 'bollinger_bands_numba', {'n': 20, 'k': 2.0}, ['upper', 'middle', 'lower']),
            ]
        },
        
        'volume': {
            'cv': [  # Close, Volume
                ('obv', 'on_balance_volume_numba', {}),
                ('force_index', 'force_index_numba', {'n': 13}),
                ('vpt', 'volume_price_trend_numba', {}),
                ('nvi', 'negative_volume_index_numba', {}),
            ],
            'hlcv': [  # High, Low, Close, Volume
                ('mfi', 'money_flow_index_numba', {'n': 14}),
                ('adi', 'acc_dist_index_numba', {}),
                ('cmf', 'chaikin_money_flow_numba', {'n': 20}),
                ('vwap', 'volume_weighted_average_price_numba', {'n': 14}),
                ('vwema', 'volume_weighted_exponential_moving_average_numba', {'n_vwma': 14, 'n_ema': 20}),
            ],
            'hlv': [  # High, Low, Volume
                ('emv', 'ease_of_movement_numba', {'n': 14}),
            ]
        },
        
        'others': {
            'close_only': [
                ('daily_return', 'daily_return_numba', {}),
                ('daily_log_return', 'daily_log_return_numba', {}),
                ('cumulative_return', 'cumulative_return_numba', {}),
                ('compound_log_return', 'compound_log_return_numba', {}),
            ]
        }
    }
    
    def __init__(self):
        """Initialize the strategy calculator."""
        self.modules = {
            'trend': trend,
            'momentum': momentum, 
            'volatility': volatility,
            'volume': volume,
            'others': others
        }
    
    def calculate_group(self, 
                       group: str, 
                       high: Optional[np.ndarray] = None,
                       low: Optional[np.ndarray] = None, 
                       close: Optional[np.ndarray] = None,
                       volume: Optional[np.ndarray] = None,
                       **kwargs) -> Dict[str, np.ndarray]:
        """
        Calculate all indicators in a specific group.
        
        Parameters
        ----------
        group : str
            Indicator group ('trend', 'momentum', 'volatility', 'volume', 'others')
        high : np.ndarray, optional
            High prices
        low : np.ndarray, optional  
            Low prices
        close : np.ndarray, optional
            Close prices
        volume : np.ndarray, optional
            Volume data
        **kwargs
            Additional parameters to override defaults
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary of calculated indicators
        """
        if group not in self.INDICATOR_GROUPS:
            raise ValueError(f"Unknown group: {group}. Available: {list(self.INDICATOR_GROUPS.keys())}")
        
        results = {}
        group_config = self.INDICATOR_GROUPS[group]
        module = self.modules[group]
        
        for input_type, indicators in group_config.items():
            for indicator_config in indicators:
                try:
                    name = indicator_config[0]
                    func_name = indicator_config[1] 
                    default_params = indicator_config[2]
                    output_names = indicator_config[3] if len(indicator_config) > 3 else [name]
                    
                    # Override defaults with user parameters
                    params = {**default_params, **kwargs}
                    
                    # Get the function
                    func = getattr(module, func_name)
                    
                    # Call with appropriate inputs
                    if input_type == 'close_only':
                        if close is None:
                            continue
                        result = func(close, **params)
                    elif input_type == 'hl':
                        if high is None or low is None:
                            continue
                        result = func(high, low, **params)
                    elif input_type == 'hlc':
                        if high is None or low is None or close is None:
                            continue
                        result = func(high, low, close, **params)
                    elif input_type == 'cv':
                        if close is None or volume is None:
                            continue
                        result = func(close, volume, **params)
                    elif input_type == 'hlcv':
                        if high is None or low is None or close is None or volume is None:
                            continue
                        result = func(high, low, close, volume, **params)
                    elif input_type == 'hlv':
                        if high is None or low is None or volume is None:
                            continue
                        result = func(high, low, volume, **params)
                    elif input_type == 'volume_only':
                        if volume is None:
                            continue
                        result = func(volume, **params)
                    elif input_type == 'multi_output':
                        # Handle indicators with multiple outputs
                        if name in ['macd', 'kst', 'ppo', 'pvo']:
                            if close is None:
                                continue
                            if name == 'pvo' and volume is None:
                                continue
                            input_data = volume if name == 'pvo' else close
                            result = func(input_data, **params)
                        elif name == 'stoch_rsi':
                            if close is None:
                                continue
                            result = func(close, **params)
                        elif name in ['stoch', 'williams_r', 'ultimate_osc']:
                            if high is None or low is None or close is None:
                                continue
                            result = func(high, low, close, **params)
                        elif name == 'ichimoku':
                            if high is None or low is None or close is None:
                                continue
                            result = func(high, low, close, **params)
                        elif name == 'bollinger':
                            if close is None:
                                continue
                            result = func(close, **params)
                        else:
                            continue
                    else:
                        continue
                    
                    # Store results
                    if isinstance(result, tuple):
                        for i, output_name in enumerate(output_names):
                            if i < len(result):
                                results[f"{name}_{output_name}"] = result[i]
                    else:
                        results[name] = result
                        
                except Exception as e:
                    warnings.warn(f"Failed to calculate {name}: {str(e)}")
                    continue
        
        return results
    
    def calculate_all(self, 
                     high: Optional[np.ndarray] = None,
                     low: Optional[np.ndarray] = None,
                     close: Optional[np.ndarray] = None, 
                     volume: Optional[np.ndarray] = None,
                     **kwargs) -> Dict[str, np.ndarray]:
        """
        Calculate all available indicators.
        
        Parameters
        ----------
        high : np.ndarray, optional
            High prices
        low : np.ndarray, optional
            Low prices  
        close : np.ndarray, optional
            Close prices
        volume : np.ndarray, optional
            Volume data
        **kwargs
            Additional parameters to override defaults
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary of all calculated indicators
        """
        all_results = {}
        
        for group in self.INDICATOR_GROUPS.keys():
            group_results = self.calculate_group(group, high, low, close, volume, **kwargs)
            all_results.update(group_results)
            
        return all_results


# Create global strategy instance
_strategy = IndicatorStrategy()


def strategy(strategy_name: str,
            high: Optional[np.ndarray] = None,
            low: Optional[np.ndarray] = None, 
            close: Optional[np.ndarray] = None,
            volume: Optional[np.ndarray] = None,
            **kwargs) -> Dict[str, np.ndarray]:
    """
    Calculate multiple indicators using a strategy.
    
    Parameters
    ----------
    strategy_name : str
        Strategy name: 'all', 'trend', 'momentum', 'volatility', 'volume', 'others'
    high : np.ndarray, optional
        High prices
    low : np.ndarray, optional
        Low prices
    close : np.ndarray, optional  
        Close prices
    volume : np.ndarray, optional
        Volume data
    **kwargs
        Additional parameters to override defaults
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary of calculated indicators
        
    Examples
    --------
    >>> import numpy as np
    >>> import ta_numba.bulk as bulk
    >>> 
    >>> # Generate sample data
    >>> close = np.random.randn(100).cumsum() + 100
    >>> high = close + np.random.rand(100) * 2
    >>> low = close - np.random.rand(100) * 2
    >>> volume = np.random.randint(1000, 10000, 100)
    >>> 
    >>> # Calculate all indicators
    >>> all_indicators = bulk.strategy("all", high, low, close, volume)
    >>> 
    >>> # Calculate only momentum indicators
    >>> momentum_indicators = bulk.strategy("momentum", close=close)
    >>> 
    >>> # Calculate trend indicators with custom parameters
    >>> trend_indicators = bulk.strategy("trend", high, low, close, n=10)
    """
    if strategy_name == "all":
        return _strategy.calculate_all(high, low, close, volume, **kwargs)
    else:
        return _strategy.calculate_group(strategy_name, high, low, close, volume, **kwargs)