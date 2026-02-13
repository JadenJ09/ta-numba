# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-02-12

### Added
- **Rust/PyO3 streaming backend**: Accelerates 45 streaming indicator classes with transparent Numba fallback
- `ta_numba.get_backend()` function to check active backend ("rust" or "numba")
- `ta_numba.is_rust_available()` function to check Rust extension availability
- `TA_NUMBA_DISABLE_RUST=1` environment variable to force Numba backend
- Pre-built wheels for Linux (x86_64, aarch64), macOS (arm64, x86_64), and Windows (x64)
- maturin-based build system for mixed Python/Rust packages

### Performance
- **Streaming indicators**: 2.6x average speedup over Numba (up to 13.3x for complex indicators)
  - Best: UlcerIndex 13.3x, StochasticRSI 9.2x, AwesomeOscillator 8.1x, BollingerBands 7.2x
  - Rust excels on indicators with state machines, rolling windows, multi-stage computations
- **Bulk indicators**: Numba JIT remains optimal for vectorized array operations
  - At production batch sizes (100K+ points), Numba's direct NumPy memory access outperforms Rust/PyO3 FFI
- Zero API changes — Rust backend is a transparent drop-in

### Changed
- Build system switched from setuptools to maturin (backwards-compatible for pure-Python installs)
- Numba JIT remains the automatic fallback on platforms without pre-built wheels
- Streaming `__init__.py` conditionally imports Rust wrapper classes when extension is available

## [0.2.3] - 2025-07-24

### Added

#### 🎯 Strategy Functions (Major Feature)

- **Bulk Strategy API**: Calculate multiple indicators at once similar to pandas-ta
  - `bulk.strategy("all", high, low, close, volume)` - All 68+ indicators
  - `bulk.strategy("momentum", close=close)` - 17 momentum indicators
  - `bulk.strategy("trend", high, low, close)` - 26 trend indicators  
  - `bulk.strategy("volatility", high, low, close)` - 11 volatility indicators
  - `bulk.strategy("volume", high, low, close, volume)` - 10 volume indicators
  - `bulk.strategy("others", close=close)` - 4 return indicators

- **Streaming Strategy API**: Real-time multiple indicator management
  - `stream.create_strategy("all")` - Manage all streaming indicators
  - `stream.create_strategy("momentum")` - Focus on momentum indicators
  - State management with `get_ready_status()`, `reset_all()`, `get_current_values()`
  - O(1) updates for all indicators simultaneously

#### 🔧 Enhanced Features

- **Flexible Input Handling**: Automatically handles different input requirements (close-only, HLC, HLCV)
- **Custom Parameters**: Override default parameters easily across all indicators
- **Error Handling**: Graceful handling of missing data or failed calculations
- **Comprehensive Testing**: Full test suite for both bulk and streaming strategies

#### 📚 Documentation & Examples

- Complete usage examples in `examples/strategy_usage_examples.py`
- Comprehensive test coverage in `tests/unit/test_strategy.py`
- Performance comparisons and real-time trading simulation examples

### Technical Details

- **Performance**: Leverages existing Numba-optimized functions for maximum speed
- **Architecture**: Clean separation between bulk and streaming modes
- **API Design**: Intuitive pandas-ta-like interface with ta-numba performance
- **Memory Efficiency**: Streaming strategies maintain constant memory usage

## [0.2.2] - 2025-07-17

### Added

#### 🔄 Streaming Indicators (Major Feature)

- **Real-time processing**: O(1) per-update performance for live trading
- **Memory efficient**: Constant memory usage regardless of data size
- **40+ streaming indicators** across all categories:
  - Trend: SMA, EMA, WMA, MACD, ADX, TRIX, CCI, DPO, Aroon, ParabolicSAR, VortexIndicator
  - Momentum: RSI, Stochastic, StochasticRSI, WilliamsR, TSI, UltimateOscillator, AwesomeOscillator, KAMA, PPO, ROC
  - Volatility: ATR, BollingerBands, KeltnerChannel, DonchianChannel, StandardDeviation, Variance, TrueRange, HistoricalVolatility, UlcerIndex
  - Volume: MoneyFlowIndex, AccDistIndex, OnBalanceVolume, ChaikinMoneyFlow, ForceIndex, EaseOfMovement, VolumePriceTrend, NegativeVolumeIndex, VWAP, VWEMA

#### ⚡ JIT Warmup System

- **Fast startup**: `ta_numba.warmup.warmup_all()` eliminates JIT compilation delays
- **Production ready**: Designed for Docker containers and production deployment
- **Selective warmup**: Individual indicator warmup available
- **Performance boost**: First-call latency reduced from ~50ms to ~0.5ms

#### 🎯 Improved API Design

- **Dual namespaces**: `ta_numba.bulk` and `ta_numba.stream` for clarity
- **Clean class names**: `SMA`, `EMA`, `RSI` instead of `SMAStreaming`
- **Better ergonomics**: Simplified imports for real-world usage
- **Legacy compatibility**: Existing imports continue to work

### Changed

- **Package structure**: Reorganized modules with cleaner namespaces
- **Documentation**: Comprehensive README update with v0.2.0 features
- **Performance benchmarks**: Added streaming vs traditional library comparisons
- **Examples**: Updated with both bulk and streaming usage patterns

### Performance

- **Streaming indicators**: 50-90x faster than pandas rolling operations
- **Memory usage**: Constant O(1) vs O(n) for traditional batch processing
- **JIT compilation**: Warmup reduces first-call latency by 99%
- **Real-time processing**: Microsecond-level updates for live trading

### Migration Guide

```python
# Old way (still supported)
import ta_numba.trend as trend
sma_values = trend.sma(prices, window=20)

# New recommended way - Bulk processing
import ta_numba.bulk as bulk
sma_values = bulk.trend.sma(prices, window=20)

# New feature - Streaming
import ta_numba.stream as stream
sma = stream.SMA(window=20)
for price in live_prices:
    current_sma = sma.update(price)
```

## [0.1.0] - 2025-07-06

### Added

- Initial release of ta-numba
- **High-performance bulk processing** for technical indicators
- **44 indicators** across 5 categories:
  - Volume indicators (10)
  - Volatility indicators (5)
  - Trend indicators (15)
  - Momentum indicators (11)
  - Other indicators (4)
- **100x to 8000x+ performance improvements** over traditional libraries
- **1-to-1 compatibility** with the ta library
- **Pure NumPy/Numba implementation** avoiding pandas overhead
- **Mathematical documentation** with precise formulas
- **Comprehensive benchmarking** against ta, ta-lib, and pandas

### Performance Highlights

- Parabolic SAR: 7783x speedup
- Negative Volume Index: 2967x speedup
- Weighted Moving Average: 842x speedup
- Average True Range: 371x speedup
- Money Flow Index: 230x speedup

### Documentation

- Detailed README with usage examples
- Mathematical formulas documentation (ta-numba.pdf)
- Performance benchmarks and comparisons
- Installation and quick start guide
