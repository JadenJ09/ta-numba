"""
Streaming strategy module for managing multiple streaming indicators at once.

This module provides classes to manage collections of streaming indicators,
allowing for efficient real-time updates of multiple indicators simultaneously.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Any
import warnings

# Import all streaming indicator classes
from .streaming import (
    # Trend indicators
    SMAStreaming, EMAStreaming, WMAStreaming, MACDStreaming, ADXStreaming,
    VortexIndicatorStreaming, TRIXStreaming, CCIStreaming, DPOStreaming,
    AroonStreaming, ParabolicSARStreaming,
    
    # Momentum indicators  
    RSIStreaming, StochasticStreaming, WilliamsRStreaming, ROCStreaming,
    UltimateOscillatorStreaming, StochasticRSIStreaming, TSIStreaming,
    AwesomeOscillatorStreaming, KAMAStreaming, PPOStreaming,
    
    # Volatility indicators
    ATRStreaming, BBandsStreaming, KeltnerChannelStreaming, DonchianChannelStreaming,
    StandardDeviationStreaming, VarianceStreaming, RangeStreaming,
    HistoricalVolatilityStreaming, UlcerIndexStreaming,
    
    # Volume indicators
    MoneyFlowIndexStreaming, AccDistIndexStreaming, OnBalanceVolumeStreaming,
    ChaikinMoneyFlowStreaming, ForceIndexStreaming, EaseOfMovementStreaming,
    VolumePriceTrendStreaming, NegativeVolumeIndexStreaming, VWAPStreaming,
    VWEMAStreaming,
    
    # Others indicators
    DailyReturnStreaming, DailyLogReturnStreaming, CumulativeReturnStreaming,
    CompoundLogReturnStreaming, RollingReturnStreaming, VolatilityStreaming,
    SharpeRatioStreaming, MaxDrawdownStreaming, CalmarRatioStreaming
)


class StreamingStrategyManager:
    """
    Manager class for multiple streaming indicators.
    
    This class allows you to create and manage collections of streaming indicators,
    providing methods to update all indicators at once and retrieve their current values.
    """
    
    # Define streaming indicator groups with their classes and default parameters
    STREAMING_GROUPS = {
        'trend': {
            # Single input (close only)
            'close_only': [
                ('sma', SMAStreaming, {'window': 20}),
                ('ema', EMAStreaming, {'window': 20}),
                ('wma', WMAStreaming, {'window': 20}),
                ('trix', TRIXStreaming, {'window': 14}),
                ('dpo', DPOStreaming, {'window': 20}),
            ],
            # High, Low, Close inputs (but updated with close for streaming)
            'hlc_as_close': [
                ('cci', CCIStreaming, {'window': 20}),
            ],
            # Multi-output indicators  
            'multi_output': [
                ('macd', MACDStreaming, {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}),
                ('adx', ADXStreaming, {'window': 14}),
                ('vortex', VortexIndicatorStreaming, {'window': 14}),
                ('aroon', AroonStreaming, {'window': 25}),
                ('parabolic_sar', ParabolicSARStreaming, {'af_start': 0.02, 'af_inc': 0.02, 'af_max': 0.2}),
            ]
        },
        
        'momentum': {
            'close_only': [
                ('rsi', RSIStreaming, {'window': 14}),
                ('roc', ROCStreaming, {'window': 12}),
                ('kama', KAMAStreaming, {'window': 10, 'fast_period': 2, 'slow_period': 30}),
                ('tsi', TSIStreaming, {'first_smooth': 25, 'second_smooth': 13}),
                ('ppo', PPOStreaming, {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}),
            ],
            'hlc_as_close': [
                ('stoch', StochasticStreaming, {'k_period': 14, 'd_period': 3}),
                ('williams_r', WilliamsRStreaming, {'window': 14}),
                ('ultimate_osc', UltimateOscillatorStreaming, {'period1': 7, 'period2': 14, 'period3': 28}),
                ('awesome_osc', AwesomeOscillatorStreaming, {'fast_period': 5, 'slow_period': 34}),
            ],
            'multi_output': [
                ('stoch_rsi', StochasticRSIStreaming, {'rsi_period': 14, 'stoch_period': 14, 'k_period': 3, 'd_period': 3}),
            ]
        },
        
        'volatility': {
            'close_only': [
                ('ulcer_index', UlcerIndexStreaming, {'window': 14}),
                ('std_dev', StandardDeviationStreaming, {'window': 20}),
                ('variance', VarianceStreaming, {'window': 20}),
                ('hist_vol', HistoricalVolatilityStreaming, {'window': 20}),
            ],
            'hlc_as_close': [
                ('atr', ATRStreaming, {'window': 14}),
                ('true_range', RangeStreaming, {'window': 20}),
            ],
            'multi_output': [
                ('bollinger', BBandsStreaming, {'window': 20, 'std_dev': 2.0}),
                ('keltner', KeltnerChannelStreaming, {'window': 20, 'atr_period': 10, 'multiplier': 2.0}),
                ('donchian', DonchianChannelStreaming, {'window': 20}),
            ]
        },
        
        'volume': {
            'cv': [  # Close, Volume (but streaming uses single update)
                ('obv', OnBalanceVolumeStreaming, {}),
                ('force_index', ForceIndexStreaming, {'window': 13}),
                ('vpt', VolumePriceTrendStreaming, {}),
                ('nvi', NegativeVolumeIndexStreaming, {}),
            ],
            'hlcv_as_close': [  # High, Low, Close, Volume (simplified to close for streaming)
                ('mfi', MoneyFlowIndexStreaming, {'window': 14}),
                ('adi', AccDistIndexStreaming, {}),
                ('cmf', ChaikinMoneyFlowStreaming, {'window': 20}),
                ('vwap', VWAPStreaming, {'window': 14}),
                ('vwema', VWEMAStreaming, {'vwma_period': 14, 'ema_period': 20}),
                ('emv', EaseOfMovementStreaming, {'window': 14}),
            ]
        },
        
        'others': {
            'close_only': [
                ('daily_return', DailyReturnStreaming, {}),
                ('daily_log_return', DailyLogReturnStreaming, {}),
                ('cumulative_return', CumulativeReturnStreaming, {}),
                ('compound_log_return', CompoundLogReturnStreaming, {}),
                ('rolling_return', RollingReturnStreaming, {'window': 20}),
                ('volatility', VolatilityStreaming, {'window': 20}),
                ('sharpe_ratio', SharpeRatioStreaming, {'window': 252, 'risk_free_rate': 0.02}),
                ('max_drawdown', MaxDrawdownStreaming, {'window': 252}),
                ('calmar_ratio', CalmarRatioStreaming, {'window': 252}),
            ]
        }
    }
    
    def __init__(self, strategy_name: str, **kwargs):
        """
        Initialize streaming strategy manager.
        
        Parameters
        ----------
        strategy_name : str
            Strategy name: 'all', 'trend', 'momentum', 'volatility', 'volume', 'others'
        **kwargs
            Parameters to override defaults for indicator creation
        """
        self.strategy_name = strategy_name
        self.indicators = {}
        self.kwargs = kwargs
        
        # Create indicators based on strategy
        if strategy_name == "all":
            self._create_all_indicators()
        elif strategy_name in self.STREAMING_GROUPS:
            self._create_group_indicators(strategy_name)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}. Available: 'all', {list(self.STREAMING_GROUPS.keys())}")
    
    def _create_group_indicators(self, group: str):
        """Create indicators for a specific group."""
        group_config = self.STREAMING_GROUPS[group]
        
        for input_type, indicators in group_config.items():
            for indicator_config in indicators:
                try:
                    name = indicator_config[0]
                    indicator_class = indicator_config[1]
                    default_params = indicator_config[2]
                    
                    # Override defaults with user parameters, but only pass compatible parameters
                    params = {**default_params}
                    
                    # Add compatible user parameters (avoid passing incompatible parameters)
                    for key, value in self.kwargs.items():
                        if key in default_params:
                            params[key] = value
                        elif key in ('window', 'n'):
                            # Translate 'n' or 'window' to the indicator's window parameter
                            if 'window' in default_params:
                                params['window'] = value
                            elif 'window' in str(indicator_class.__init__.__code__.co_varnames):
                                params['window'] = value
                    
                    # Create indicator instance
                    self.indicators[name] = indicator_class(**params)
                    
                except Exception as e:
                    # Silently skip indicators that can't be created instead of warning
                    continue
    
    def _create_all_indicators(self):
        """Create all available streaming indicators."""
        for group in self.STREAMING_GROUPS.keys():
            self._create_group_indicators(group)
    
    def update(self, 
               high: Optional[float] = None,
               low: Optional[float] = None,
               close: Optional[float] = None,
               volume: Optional[float] = None) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Update all indicators with new market data.
        
        Parameters
        ----------
        high : float, optional
            High price
        low : float, optional
            Low price
        close : float, optional
            Close price
        volume : float, optional
            Volume
            
        Returns
        -------
        Dict[str, Union[float, Dict[str, float]]]
            Dictionary of current indicator values
        """
        results = {}
        
        for name, indicator in self.indicators.items():
            try:
                # Determine what inputs the indicator needs based on its update method
                # For simplicity, most streaming indicators use close price as primary input
                # Some use additional inputs but we'll use close as the main driver
                
                if hasattr(indicator, 'update'):
                    if name in ['obv', 'force_index', 'vpt', 'nvi'] and close is not None and volume is not None:
                        # Volume-based indicators that need both close and volume
                        if hasattr(indicator.update, '__code__') and indicator.update.__code__.co_argcount > 2:
                            result = indicator.update(close, volume)
                        else:
                            result = indicator.update(close)  # Fallback to close only
                    elif name in ['mfi', 'adi', 'cmf', 'vwap', 'vwema', 'emv'] and all(x is not None for x in [high, low, close, volume]):
                        # HLCV indicators - use close as primary but may need additional logic
                        if hasattr(indicator.update, '__code__') and indicator.update.__code__.co_argcount > 2:
                            try:
                                result = indicator.update(high, low, close, volume)
                            except:
                                result = indicator.update(close)  # Fallback
                        else:
                            result = indicator.update(close)
                    elif name in ['atr', 'true_range'] and all(x is not None for x in [high, low, close]):
                        # HLC indicators - use close but may need high/low for True Range
                        if hasattr(indicator.update, '__code__') and indicator.update.__code__.co_argcount > 2:
                            try:
                                result = indicator.update(high, low, close)
                            except:
                                result = indicator.update(close)  # Fallback
                        else:
                            result = indicator.update(close)
                    elif close is not None:
                        # Most indicators use close price
                        result = indicator.update(close)
                    else:
                        continue  # Skip if no close price
                    
                    # Store result
                    if isinstance(result, dict):
                        results[name] = result
                    elif not np.isnan(result):
                        results[name] = result
                    else:
                        results[name] = np.nan
                        
            except Exception as e:
                # Silently skip failed updates instead of warning
                results[name] = np.nan
                continue
        
        return results
    
    def get_current_values(self) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Get current values of all indicators without updating.
        
        Returns
        -------
        Dict[str, Union[float, Dict[str, float]]]
            Dictionary of current indicator values
        """
        results = {}
        
        for name, indicator in self.indicators.items():
            try:
                if hasattr(indicator, 'current_value'):
                    results[name] = indicator.current_value
                elif hasattr(indicator, 'current_values'):
                    results[name] = indicator.current_values
                else:
                    results[name] = np.nan
            except:
                results[name] = np.nan
        
        return results
    
    def get_ready_status(self) -> Dict[str, bool]:
        """
        Get ready status of all indicators.
        
        Returns
        -------
        Dict[str, bool]
            Dictionary indicating which indicators are ready
        """
        status = {}
        
        for name, indicator in self.indicators.items():
            try:
                status[name] = getattr(indicator, 'is_ready', False)
            except:
                status[name] = False
        
        return status
    
    def reset_all(self):
        """Reset all indicators to their initial state."""
        for indicator in self.indicators.values():
            if hasattr(indicator, 'reset'):
                indicator.reset()
    
    def get_indicator_names(self) -> List[str]:
        """Get list of all indicator names."""
        return list(self.indicators.keys())
    
    def get_indicator(self, name: str):
        """Get a specific indicator by name."""
        return self.indicators.get(name)


def create_streaming_strategy(strategy_name: str, **kwargs) -> StreamingStrategyManager:
    """
    Create a streaming strategy manager.
    
    Parameters
    ---------- 
    strategy_name : str
        Strategy name: 'all', 'trend', 'momentum', 'volatility', 'volume', 'others'
    **kwargs
        Parameters to override defaults for indicator creation
        
    Returns
    -------
    StreamingStrategyManager
        Configured strategy manager instance
        
    Examples
    --------
    >>> import ta_numba.stream as stream
    >>> 
    >>> # Create all indicators
    >>> all_strategy = stream.create_streaming_strategy("all")
    >>> 
    >>> # Create only momentum indicators with custom RSI period
    >>> momentum_strategy = stream.create_streaming_strategy("momentum", window=21)
    >>> 
    >>> # Update with new price data
    >>> results = momentum_strategy.update(close=100.5)
    >>> print(f"RSI: {results.get('rsi', 'Not ready')}")
    >>> 
    >>> # Get ready status
    >>> ready_status = momentum_strategy.get_ready_status()
    >>> print(f"Ready indicators: {[k for k, v in ready_status.items() if v]}")
    """
    return StreamingStrategyManager(strategy_name, **kwargs)