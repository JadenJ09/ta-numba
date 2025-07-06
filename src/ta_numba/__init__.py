# src/ta_numba/__init__.py

"""
ta-numba
A high-performance technical analysis library for financial data, accelerated with Numba.
This library provides a Numba-accelerated, 1-to-1 compatible replacement for 
many of the indicators in the popular 'ta' library.
"""

# Import the modules to create the package structure
from . import volume
from . import volatility
from . import trend
from . import momentum
from . import others

# You can also import specific functions to the top-level namespace if desired
# For example, to allow `from ta_numba import sma_numba` instead of `from ta_numba.trend import sma_numba`


__version__ = "0.1.0"

__all__ = [
    'volume',
    'volatility',
    'trend',
    'momentum',
    'others',
]
