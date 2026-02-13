# **TA-Numba: Technical Analysis Library with Numba & Rust Acceleration**

**ta-numba** is a Python library for financial technical analysis that provides **dependency-free installation** and **high-performance computation** through Numba JIT compilation and an optional Rust/PyO3 backend. It offers both **bulk processing** for historical analysis and **real-time streaming** for live trading applications.

## **Key Features**

- **Dependency-Free Installation:** Pure Python with NumPy and Numba — no C compiler needed
- **Dual Processing Modes:** Bulk (vectorized arrays) + Streaming (O(1) per-tick updates)
- **Optional Rust Backend (v0.3.0+):** Up to 13x faster streaming via PyO3 — automatic fallback to Numba
- **45 Streaming + 44 Bulk Indicators:** Trend, Momentum, Volatility, Volume, and more
- **Docker & Cloud Ready:** Reliable installation, optional JIT warmup, constant memory streaming

---

## **What's New in v0.3.0 — Rust/PyO3 Streaming Backend**

v0.3.0 adds an optional Rust backend that accelerates **streaming indicators** for real-time trading. On supported platforms, `pip install ta-numba` automatically includes pre-built Rust extensions — no Rust toolchain needed.

### **Architecture**

ta-numba uses each backend where it performs best:

| Mode | Backend | Reason |
|---|---|---|
| **Bulk** (array operations) | Numba JIT | Operates directly on NumPy memory — no FFI overhead |
| **Streaming** (per-tick updates) | Rust/PyO3 | Native state management — 2-13x faster for complex indicators |

### **Streaming Benchmark: Rust vs Numba**

10,000 price ticks, 10 iterations, median timing. Full results across 45 streaming indicators:

<details open>
<summary><strong>Top Rust Wins (Complex Indicators)</strong></summary>

| Indicator | Rust (ms) | Numba (ms) | Speedup |
|---|---|---|---|
| UlcerIndex | 16.9 | 225.2 | **13.3x** |
| StochasticRSI | 10.3 | 94.0 | **9.2x** |
| AwesomeOscillator | 6.7 | 54.1 | **8.1x** |
| BollingerBands | 11.2 | 81.2 | **7.2x** |
| UltimateOscillator | 15.8 | 109.0 | **6.9x** |
| CCI | 8.5 | 52.8 | **6.2x** |
| TSI | 5.0 | 28.7 | **5.8x** |
| DPO | 5.2 | 26.9 | **5.2x** |
| KAMA | 8.5 | 42.4 | **5.0x** |
| SMA | 5.1 | 25.6 | **5.0x** |
| MassIndex | 8.4 | 37.5 | **4.5x** |
| MACD | 5.3 | 20.6 | **3.9x** |
| TRIX | 5.0 | 18.5 | **3.7x** |
| EMA | 5.0 | 17.2 | **3.4x** |
| WMA | 5.3 | 16.8 | **3.2x** |
| RSI | 5.0 | 13.9 | **2.8x** |
| PPO | 5.1 | 12.1 | **2.4x** |
| KST | 5.1 | 11.6 | **2.3x** |
| ATR | 8.6 | 17.9 | **2.1x** |
| ADX | 11.2 | 22.2 | **2.0x** |
| VortexIndicator | 8.5 | 15.1 | **1.8x** |
| Aroon | 6.1 | 10.1 | **1.7x** |
| Ichimoku | 8.9 | 13.2 | **1.5x** |
| MFI | 11.1 | 14.7 | **1.3x** |
| CMF | 9.5 | 12.0 | **1.3x** |
| STC | 5.3 | 6.3 | **1.2x** |
| VWAP | 8.9 | 10.2 | **1.1x** |
| ForceIndex | 5.7 | 6.5 | **1.1x** |
| KeltnerChannel | 12.8 | 13.7 | **1.1x** |

</details>

<details>
<summary><strong>Numba Wins (Simple Indicators)</strong></summary>

Simple indicators with minimal computation per tick — the PyO3 FFI call overhead (~0.5us/call) dominates:

| Indicator | Rust (ms) | Numba (ms) | Speedup |
|---|---|---|---|
| StochasticOscillator | 11.7 | 10.0 | 0.9x |
| DailyLogReturn | 5.7 | 4.8 | 0.8x |
| DonchianChannel | 11.5 | 8.2 | 0.7x |
| WilliamsR | 10.5 | 7.3 | 0.7x |
| ParabolicSAR | 6.0 | 3.4 | 0.6x |
| EaseOfMovement | 6.2 | 3.8 | 0.6x |
| ROC | 5.0 | 1.9 | 0.4x |
| AccDistIndex | 7.9 | 3.5 | 0.4x |
| VolumePriceTrend | 6.2 | 2.3 | 0.4x |
| NegativeVolumeIndex | 6.1 | 1.9 | 0.3x |
| CumulativeReturn | 5.4 | 1.7 | 0.3x |
| DailyReturn | 5.6 | 1.8 | 0.3x |
| OnBalanceVolume | 6.3 | 1.5 | 0.2x |

</details>

**Summary:** Average 2.6x faster | Median 1.6x | Complex indicators: **5-13x faster**

### **Bulk Benchmark: Numba vs Rust**

100,000 data points, 50 iterations, median timing.

Numba JIT wins all 44 bulk indicators (geometric mean: **9x faster** than Rust). This is expected — Numba generates native code that operates directly on NumPy memory buffers without any FFI boundary crossing, while Rust/PyO3 bulk calls incur per-call data marshalling overhead.

<details>
<summary><strong>Full Bulk Results (44 indicators)</strong></summary>

| Indicator | Rust (ms) | Numba (ms) | Winner |
|---|---|---|---|
| SMA(20) | 1.34 | 1.06 | Numba 1.3x |
| EMA(20) | 4.59 | 0.11 | Numba 42x |
| WMA(20) | 21.01 | 2.55 | Numba 8.2x |
| MACD | 15.28 | 0.87 | Numba 18x |
| ADX(14) | 18.59 | 4.36 | Numba 4.3x |
| CCI(20) | 23.91 | 3.04 | Numba 7.9x |
| PSAR | 3.46 | 0.33 | Numba 10x |
| TRIX(14) | 18.44 | 0.36 | Numba 52x |
| Aroon(25) | 58.18 | 2.72 | Numba 21x |
| Vortex(14) | 14.16 | 3.96 | Numba 3.6x |
| DPO(20) | 3.55 | 1.10 | Numba 3.2x |
| KST | 45.29 | 15.50 | Numba 2.9x |
| STC | 67.13 | 12.47 | Numba 5.4x |
| Ichimoku | 113.93 | 17.29 | Numba 6.6x |
| MassIndex | 32.81 | 5.01 | Numba 6.6x |
| RSI(14) | 9.21 | 1.18 | Numba 7.8x |
| Stochastic | 26.49 | 6.06 | Numba 4.4x |
| Williams %R | 22.87 | 5.10 | Numba 4.5x |
| KAMA(10) | 22.12 | 0.67 | Numba 33x |
| PPO | 12.84 | 0.58 | Numba 22x |
| ROC(12) | 1.35 | 0.10 | Numba 14x |
| StochRSI | 46.13 | 6.52 | Numba 7.1x |
| AwesomeOsc | 7.37 | 2.21 | Numba 3.3x |
| TSI | 23.60 | 0.74 | Numba 32x |
| UltimateOsc | 21.80 | 7.55 | Numba 2.9x |
| ATR(14) | 3.32 | 0.46 | Numba 7.2x |
| BB(20) | 27.56 | 2.56 | Numba 11x |
| KC(20) | 7.49 | 3.64 | Numba 2.1x |
| Donchian(20) | 30.78 | 4.91 | Numba 6.3x |
| UlcerIndex(14) | 17.26 | 3.64 | Numba 4.7x |
| OBV | 1.75 | 0.05 | Numba 33x |
| MFI(14) | 16.15 | 4.05 | Numba 4.0x |
| CMF(20) | 9.13 | 2.21 | Numba 4.1x |
| ForceIndex(13) | 3.25 | 0.22 | Numba 15x |
| EOM(14) | 1.54 | 0.05 | Numba 30x |
| VWAP(20) | 15.91 | 0.57 | Numba 28x |
| ADI | 4.63 | 0.18 | Numba 25x |
| VPT | 4.54 | 0.19 | Numba 23x |
| NVI | 3.38 | 0.33 | Numba 10x |
| VWEMA | 20.74 | 0.82 | Numba 25x |
| DailyReturn | 1.26 | 0.10 | Numba 12x |
| DailyLogReturn | 1.59 | 0.36 | Numba 4.5x |
| CompoundLogReturn | 330.24 | 330.18 | Tie |
| CumulativeReturn | 1.19 | 0.03 | Numba 35x |

</details>

### **Usage**

```python
import ta_numba

# Check which backend is active
print(ta_numba.get_backend())  # "rust" or "numba"
```

```bash
# Force Numba backend (for debugging or benchmarking)
export TA_NUMBA_DISABLE_RUST=1
```

```bash
# Build from source (requires Rust toolchain from rustup.rs)
pip install maturin
maturin develop --release
```

### **Supported Platforms**

Pre-built wheels with Rust acceleration:
- Linux x86_64 / aarch64
- macOS arm64 / x86_64
- Windows x64

Other platforms: automatic Numba JIT fallback (no Rust needed).

---

## **What's New in v0.2.0 — Real-Time Streaming**

- **45 Streaming Indicators:** O(1) per-update, constant memory, designed for live trading
- **JIT Warmup System:** `ta_numba.warmup.warmup_all()` eliminates cold-start latency
- **Dual Namespaces:** `ta_numba.bulk` and `ta_numba.stream` for clarity
- **Streaming vs Bulk:** 15.8x faster per-tick updates with 547x less memory
- **Legacy Compatible:** Existing `ta_numba.trend`/`ta_numba.momentum` imports still work

---

## **Installation**

```bash
pip install ta-numba
```

Dependencies: `numpy`, `numba` (automatically installed). Rust extensions included on supported platforms.

## **Quick Start**

### **Bulk Processing (Batch Calculations)**

Perfect for backtesting and historical analysis:

```python
import ta_numba.bulk as bulk
import numpy as np

# Your price data
close_prices = np.array([100, 102, 101, 103, 105, 104, 106])

# Calculate indicators on entire dataset
sma_20 = bulk.trend.sma(close_prices, window=20)
rsi_14 = bulk.momentum.rsi(close_prices, window=14)
macd_line, macd_signal, macd_hist = bulk.trend.macd(close_prices)

# Warm up JIT compilation for faster subsequent calls
import ta_numba.warmup
ta_numba.warmup.warmup_all()  # Optional but recommended
```

### **Real-Time Streaming (Live Trading)**

Perfect for live market data and real-time trading:

```python
import ta_numba.stream as stream

# Create streaming indicators
sma = stream.SMA(window=20)
rsi = stream.RSI(window=14)
macd = stream.MACD(fast=12, slow=26, signal=9)

# Process live price updates
def on_new_price(price):
    sma_value = sma.update(price)
    rsi_value = rsi.update(price)
    macd_values = macd.update(price)

    if sma.is_ready:
        print(f"SMA: {sma_value:.2f}")
    if rsi.is_ready:
        print(f"RSI: {rsi_value:.2f}")
    if macd.is_ready:
        print(f"MACD: {macd_values}")

# Simulate live data
for price in [100, 102, 101, 103, 105]:
    on_new_price(price)
```

### **Legacy Compatibility (Direct Import)**

For existing ta library users:

```python
# Same as original ta library
import ta_numba.trend as trend
import ta_numba.momentum as momentum

sma_values = trend.sma(close_prices, window=20)
rsi_values = momentum.rsi(close_prices, window=14)
```

## **Available Indicators**

### **Streaming Indicators (45)**

Real-time indicators with O(1) updates and constant memory usage:

**Trend (11):** SMA, EMA, WMA, MACD, ADX, TRIX, CCI, DPO, Aroon, ParabolicSAR, VortexIndicator

**Momentum (10):** RSI, Stochastic, StochasticRSI, WilliamsR, TSI, UltimateOscillator, AwesomeOscillator, KAMA, PPO, ROC

**Volatility (9):** ATR, BollingerBands, KeltnerChannel, DonchianChannel, StandardDeviation, Variance, TrueRange, HistoricalVolatility, UlcerIndex

**Volume (10):** MoneyFlowIndex, AccDistIndex, OnBalanceVolume, ChaikinMoneyFlow, ForceIndex, EaseOfMovement, VolumePriceTrend, NegativeVolumeIndex, VWAP, VWEMA

**Others (5):** DailyReturn, DailyLogReturn, CompoundLogReturn, CumulativeReturn, SharpeRatio, MaxDrawdown, Volatility

### **Bulk Processing Indicators (44)**

All functions accept NumPy arrays for maximum performance.

<details>
<summary><strong>Trend (15)</strong></summary>

`sma`, `ema`, `wma`, `macd`, `adx`, `vortex_indicator`, `trix`, `mass_index`, `cci`, `dpo`, `kst`, `ichimoku`, `parabolic_sar`, `schaff_trend_cycle`, `aroon`

</details>

<details>
<summary><strong>Momentum (11)</strong></summary>

`rsi`, `stochrsi`, `tsi`, `ultimate_oscillator`, `stoch`, `williams_r`, `awesome_oscillator`, `kama`, `roc`, `ppo`, `pvo`

</details>

<details>
<summary><strong>Volatility (5)</strong></summary>

`average_true_range`, `bollinger_bands`, `keltner_channel`, `donchian_channel`, `ulcer_index`

</details>

<details>
<summary><strong>Volume (10)</strong></summary>

`money_flow_index`, `acc_dist_index`, `on_balance_volume`, `chaikin_money_flow`, `force_index`, `ease_of_movement`, `volume_price_trend`, `negative_volume_index`, `volume_weighted_average_price`, `volume_weighted_exponential_moving_average`

</details>

<details>
<summary><strong>Others (4)</strong></summary>

`daily_return`, `daily_log_return`, `cumulative_return`, `compound_log_return`

</details>

## **Performance & Benchmarks**

### **Library Comparison (Bulk, 100K data points)**

| Aspect | TA-Lib | ta-numba | ta | pandas |
|---|---|---|---|---|
| **Installation** | C compiler required | pip install only | pip install only | pip install only |
| **Avg Performance** | Fastest (baseline) | 4.3x slower | 857x slower | 94x slower |
| **Streaming** | No | Yes (Rust-accelerated) | No | No |
| **Dependency Issues** | Frequent | None | None | Rare |

### **Streaming vs Bulk Recalculation**

```text
Method          Mean      Median    99th %ile   Memory
Bulk            0.347ms   0.346ms   0.699ms     O(n) = 547 KB
Streaming       0.022ms   0.022ms   0.039ms     O(1) = ~1 KB
Speedup         15.8x     15.9x                 547x less
```

### **Library Selection Guide**

- **Choose TA-Lib for:** Maximum bulk speed, stable environment, C compilation acceptable
- **Choose ta-numba for:** Reliable deployment, streaming, Python-only environments, Rust acceleration
- **Choose ta/pandas for:** Prototyping, small datasets, existing pandas workflows

<details>
<summary><strong>Detailed Benchmark Results (ta-numba vs ta library, 200K data points)</strong></summary>

```text
Indicator  | ta Library      | ta-numba (Numba)  | Speedup
---------------------------------------------------------------
PSAR       | 9.464796s       | 0.001216s         | 7,783x
NVI        | 3.244231s       | 0.001093s         | 2,967x
WMA        | 5.459586s       | 0.006479s         | 843x
MFI        | 1.187933s       | 0.005150s         | 231x
ATR        | 0.419494s       | 0.001130s         | 371x
CCI        | 1.055140s       | 0.007558s         | 140x
ADX        | 0.883612s       | 0.007472s         | 118x
KAMA       | 0.130242s       | 0.001560s         | 83x
Aroon      | 0.402076s       | 0.005702s         | 71x
UI         | 0.398492s       | 0.007430s         | 54x
OBV        | 0.001602s       | 0.000122s         | 13x
EOM        | 0.001648s       | 0.000172s         | 10x
EMA        | 0.001192s       | 0.000444s         | 5.2x
VPT        | 0.002104s       | 0.000451s         | 4.7x
TRIX       | 0.004868s       | 0.001166s         | 4.2x
ADI        | 0.001475s       | 0.000434s         | 3.4x
PVO        | 0.003904s       | 0.001216s         | 3.2x
VWAP       | 0.003858s       | 0.001392s         | 2.8x
PPO        | 0.003494s       | 0.001294s         | 2.7x
VWEMA      | 0.005218s       | 0.002011s         | 2.6x
MACD       | 0.003275s       | 0.001290s         | 2.5x
CMF        | 0.004253s       | 0.001713s         | 2.5x
FI         | 0.001479s       | 0.000609s         | 2.4x
UO         | 0.034889s       | 0.014549s         | 2.4x
TSI        | 0.004547s       | 0.001771s         | 2.6x
ROC        | 0.000777s       | 0.000344s         | 2.3x
DR         | 0.000662s       | 0.000300s         | 2.2x
Vortex     | 0.016811s       | 0.007960s         | 2.1x
CR         | 0.000388s       | 0.000184s         | 2.1x
RSI        | 0.004719s       | 0.002710s         | 1.7x
BB         | 0.004472s       | 0.003196s         | 1.4x
SMA        | 0.001696s       | 0.002453s         | 0.7x
STC        | 0.018517s       | 0.019506s         | 0.9x
StochRSI   | 0.012424s       | 0.014490s         | 0.9x
```

Average speedup vs ta library: **857x**

</details>

## **Migration Guide**

### **From v0.2.x to v0.3.0**

No code changes required. The Rust backend is automatically used for streaming when available:

```python
# This code works identically on v0.2.x and v0.3.0
import ta_numba.stream as stream
rsi = stream.RSI(window=14)
result = rsi.update(price)  # Automatically Rust-accelerated on v0.3.0
```

### **From v0.1.x to v0.2.0**

```python
# Old way (still supported)
import ta_numba.trend as trend
sma_values = trend.sma(prices, window=20)

# New recommended way
import ta_numba.bulk as bulk
sma_values = bulk.trend.sma(prices, window=20)

# New feature - Streaming
import ta_numba.stream as stream
sma = stream.SMA(window=20)
for price in live_prices:
    current_sma = sma.update(price)
```

### **From Other Libraries**

```python
# From pandas
df['sma'] = df['close'].rolling(20).mean()
# To ta-numba bulk
sma_values = bulk.trend.sma(df['close'].values, window=20)

# From ta-lib (no streaming equivalent)
# To ta-numba streaming
sma = stream.SMA(window=20)
current_value = sma.update(new_price)
```

## **Advanced Usage**

### **Production Deployment**

```python
# Recommended startup sequence for production
import ta_numba.warmup
import ta_numba.bulk as bulk
import ta_numba.stream as stream

# Warm up all indicators (do this once at startup)
ta_numba.warmup.warmup_all()

# Now all subsequent calls are fast
def process_historical_data(prices):
    return bulk.trend.sma(prices, window=20)

def process_live_data():
    sma = stream.SMA(window=20)
    for price in live_feed:
        yield sma.update(price)
```

### **Docker Integration**

```dockerfile
FROM python:3.11
RUN pip install ta-numba

# Pre-compile Numba indicators at build time
RUN python -c "import ta_numba.warmup; ta_numba.warmup.warmup_all()"

COPY . .
CMD ["python", "your_trading_app.py"]
```

### **Live Trading Example**

```python
import ta_numba.stream as stream

indicators = {
    'sma_20': stream.SMA(window=20),
    'sma_50': stream.SMA(window=50),
    'rsi': stream.RSI(window=14),
    'macd': stream.MACD()
}

def on_price_update(price):
    signals = {}
    for name, indicator in indicators.items():
        signals[name] = indicator.update(price)

    if all(ind.is_ready for ind in indicators.values()):
        if signals['sma_20'] > signals['sma_50']:
            return "BUY_SIGNAL"
        elif signals['rsi'] > 70:
            return "SELL_SIGNAL"
    return "HOLD"
```

## **Acknowledgements**

This library builds upon the excellent work of several projects:

- **[Technical Analysis Library (ta)](https://github.com/bukosabino/ta)** by Dario Lopez Padial - API design and calculation logic foundation
- **[Numba](https://numba.pydata.org/)** - JIT compilation technology that makes the performance possible
- **[NumPy](https://numpy.org/)** - Fundamental array operations and mathematical functions
- **[PyO3](https://pyo3.rs/)** - Rust/Python bindings powering the v0.3.0 backend

## **Mathematical Documentation**

All indicator implementations are based on established formulas documented in: [`ta-numba.pdf`](ta-numba.pdf)

## **Contributing**

We welcome contributions! Whether it's bug reports, new indicators, performance optimizations, documentation improvements, or test coverage expansion.

Please see our [contributing guidelines](CONTRIBUTING.md) for details.

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
