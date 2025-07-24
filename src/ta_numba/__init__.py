# src/ta_numba/__init__.py

"""
ta-numba
A high-performance technical analysis library for financial data, accelerated with Numba.

Features:
- Bulk processing: High-performance batch calculations with JIT compilation
- Streaming: Real-time O(1) indicators for live trading
- 1-to-1 compatibility with the popular 'ta' library
- Dramatic speed improvements (100x to 8000x+ on iterative indicators)

Usage:
    # Bulk processing (batch calculations)
    import ta_numba.bulk as ta_bulk
    sma_values = ta_bulk.trend.sma(prices, window=20)

    # Streaming (real-time updates)
    import ta_numba.stream as ta_stream
    sma = ta_stream.SMA(window=20)
    current_sma = sma.update(new_price)

    # JIT warmup for faster startup
    import ta_numba.warmup
    ta_numba.warmup.warmup_all()
"""

# Import warmup functionality
# Import streaming module
# Import bulk processing modules (renamed for clarity)
from . import momentum as _momentum_bulk
from . import others as _others_bulk
from . import strategy as _strategy
from . import streaming as _streaming
from . import streaming_strategy as _streaming_strategy
from . import trend as _trend_bulk
from . import volatility as _volatility_bulk
from . import volume as _volume_bulk
from . import warmup


# Create convenient namespace aliases
class BulkNamespace:
    """Namespace for bulk processing indicators"""

    volume = _volume_bulk
    volatility = _volatility_bulk
    trend = _trend_bulk
    momentum = _momentum_bulk
    others = _others_bulk
    
    # Add strategy function to bulk namespace as static method
    @staticmethod
    def strategy(strategy_name, high=None, low=None, close=None, volume=None, **kwargs):
        """Calculate multiple indicators using a strategy."""
        return _strategy.strategy(strategy_name, high, low, close, volume, **kwargs)


class StreamingNamespace:
    """Enhanced namespace for streaming indicators with strategy support"""
    
    def __init__(self, streaming_module):
        # Copy all attributes from the streaming module
        for attr_name in dir(streaming_module):
            if not attr_name.startswith('_'):
                setattr(self, attr_name, getattr(streaming_module, attr_name))
    
    # Add streaming strategy functionality as static methods
    @staticmethod 
    def create_strategy(strategy_name, **kwargs):
        """Create a streaming strategy manager."""
        return _streaming_strategy.create_streaming_strategy(strategy_name, **kwargs)
    
    # Make Strategy class directly accessible
    Strategy = _streaming_strategy.StreamingStrategyManager


# Create bulk processing namespace
bulk = BulkNamespace()

# Create enhanced streaming namespace
stream = StreamingNamespace(_streaming)


__version__ = "0.2.3"

__all__ = [
    "bulk",  # Bulk processing namespace
    "stream",  # Streaming indicators namespace
    "warmup",  # JIT warmup functions
    # Legacy compatibility (deprecated in future versions)
    "volume",
    "volatility",
    "trend",
    "momentum",
    "others",
    "streaming",
]

# Legacy compatibility - will be deprecated in future versions
volume = _volume_bulk
volatility = _volatility_bulk
trend = _trend_bulk
momentum = _momentum_bulk
others = _others_bulk
streaming = _streaming
