# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
