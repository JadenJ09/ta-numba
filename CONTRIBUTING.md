# Contributing to ta-numba

Thank you for your interest in contributing to ta-numba! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Development Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/JadenJ09/ta-numba.git
   cd ta-numba
   ```

2. **Set up development environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e .[dev]
   ```

3. **Run tests**
   ```bash
   pytest tests/
   ```

## 🎯 Types of Contributions

### 🐛 Bug Reports

- Use GitHub issues with the "bug" label
- Include minimal reproducible example
- Specify Python version, ta-numba version, and OS
- Provide expected vs actual behavior

### 📈 New Indicators

- Follow existing indicator patterns in `src/ta_numba/`
- Implement both bulk and streaming versions
- Include comprehensive tests
- Document mathematical formulas
- Ensure 1-to-1 compatibility with reference implementations

### ⚡ Performance Optimizations

- Benchmark before and after changes
- Focus on Numba-friendly optimizations
- Maintain numerical accuracy
- Test across different data sizes

### 📚 Documentation

- Update README.md for new features
- Add docstrings following NumPy style
- Include usage examples
- Update mathematical documentation

## 📋 Development Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use meaningful variable names
- Add type hints where appropriate
- Keep functions focused and modular

### Numba Best Practices

- Use `@numba.njit` for maximum performance
- Avoid Python objects in jitted functions
- Use NumPy arrays instead of lists
- Minimize memory allocations in loops

### Testing Requirements

- Write tests for both bulk and streaming versions
- Test edge cases (empty arrays, single values)
- Verify numerical accuracy against reference implementations
- Include performance regression tests

### Mathematical Accuracy

- Follow established financial formulas
- Handle edge cases (division by zero, NaN values)
- Document any deviations from standard formulas
- Verify against multiple reference sources

## 🔄 Indicator Implementation Guide

### Bulk Indicator Template

```python
import numba
import numpy as np

@numba.njit
def your_indicator_numba(close: np.ndarray, window: int = 14) -> np.ndarray:
    """
    Your Indicator calculation using Numba JIT compilation.

    Parameters
    ----------
    close : np.ndarray
        Array of closing prices
    window : int, default 14
        Period for calculation

    Returns
    -------
    np.ndarray
        Calculated indicator values
    """
    n = len(close)
    result = np.full(n, np.nan)

    # Your implementation here

    return result

def your_indicator(close, window=14):
    """
    Your Indicator - user-friendly wrapper.

    Parameters
    ----------
    close : array-like
        Closing prices
    window : int, default 14
        Period for calculation

    Returns
    -------
    np.ndarray
        Calculated indicator values
    """
    close_array = np.asarray(close, dtype=np.float64)
    return your_indicator_numba(close_array, window)
```

### Streaming Indicator Template

```python
from .base import StreamingIndicator
import numba

@numba.experimental.jitclass([
    ('window', numba.int64),
    ('buffer', numba.float64[:]),
    ('index', numba.int64),
    ('count', numba.int64),
    ('current_value', numba.float64),
    ('is_ready', numba.boolean),
])
class YourIndicatorStreaming(StreamingIndicator):
    """Streaming Your Indicator implementation."""

    def __init__(self, window=14):
        self.window = window
        self.buffer = np.zeros(window)
        self.index = 0
        self.count = 0
        self.current_value = np.nan
        self.is_ready = False

    def update(self, value):
        """Update indicator with new value."""
        # Your streaming implementation here

        return self.current_value

    def reset(self):
        """Reset indicator state."""
        self.index = 0
        self.count = 0
        self.current_value = np.nan
        self.is_ready = False
        self.buffer.fill(0.0)
```

## 🧪 Testing Guidelines

### Test Structure

```python
import pytest
import numpy as np
from ta_numba.trend import your_indicator
from ta_numba.streaming import YourIndicatorStreaming

class TestYourIndicator:
    def test_basic_calculation(self):
        """Test basic indicator calculation."""
        close = np.array([1, 2, 3, 4, 5])
        result = your_indicator(close, window=3)
        # Add assertions

    def test_streaming_accuracy(self):
        """Test streaming vs bulk accuracy."""
        close = np.random.randn(100).cumsum() + 100

        # Bulk calculation
        bulk_result = your_indicator(close, window=10)

        # Streaming calculation
        stream = YourIndicatorStreaming(window=10)
        streaming_results = []
        for price in close:
            streaming_results.append(stream.update(price))

        # Compare results
        np.testing.assert_allclose(streaming_results, bulk_result, rtol=1e-10)

    def test_edge_cases(self):
        """Test edge cases."""
        # Empty array
        assert len(your_indicator(np.array([]))) == 0

        # Single value
        result = your_indicator(np.array([100]))
        assert len(result) == 1

        # NaN handling
        close_with_nan = np.array([1, 2, np.nan, 4, 5])
        result = your_indicator(close_with_nan)
        # Verify NaN handling
```

## 📝 Documentation Standards

### Function Docstrings

Use NumPy style docstrings:

```python
def indicator_function(close, window=14, param=0.5):
    """
    Brief description of the indicator.

    Longer description explaining the indicator's purpose,
    mathematical background, and typical usage.

    Parameters
    ----------
    close : array-like
        Array of closing prices
    window : int, default 14
        Period for the calculation
    param : float, default 0.5
        Additional parameter description

    Returns
    -------
    np.ndarray
        Array of indicator values with same length as input

    Notes
    -----
    Mathematical formula and implementation details.

    Examples
    --------
    >>> close = np.array([1, 2, 3, 4, 5])
    >>> result = indicator_function(close, window=3)
    """
```

## 🚀 Pull Request Process

1. **Fork the repository** and create a feature branch
2. **Implement your changes** following the guidelines above
3. **Add comprehensive tests** for your changes
4. **Update documentation** as needed
5. **Run the test suite** to ensure all tests pass
6. **Submit a pull request** with a clear description

### Pull Request Checklist

- [ ] Tests pass (`pytest tests/`)
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (for significant changes)
- [ ] Mathematical accuracy verified
- [ ] Performance benchmarked (for optimizations)

## 🤝 Community Guidelines

- Be respectful and constructive in discussions
- Help others learn and grow
- Focus on improving the library for everyone
- Ask questions if you're unsure about anything

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: jaden.b.jeong@gmail.com for sensitive matters

Thank you for contributing to ta-numba! 🚀
