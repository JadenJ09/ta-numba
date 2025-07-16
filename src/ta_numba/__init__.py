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
from . import streaming as _streaming
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


# Create bulk processing namespace
bulk = BulkNamespace()

# Create streaming namespace alias
stream = _streaming


__version__ = "0.2.0"

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
