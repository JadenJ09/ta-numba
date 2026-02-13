# **TA-Numba: Technical Analysis Library with Numba Acceleration**

**ta-numba** is a Python library for financial technical analysis that provides **dependency-free installation** and **high-performance computation** through Numba JIT compilation. It offers both **bulk processing** for historical analysis and **real-time streaming** for live trading applications.

## **🚀 Key Features**

### **Dependency-Free Installation**

- **No C Compilation Required:** Pure Python implementation with NumPy and Numba dependencies only
- **Docker Compatible:** Reliable installation in containerized environments
- **Modern Python Support:** Compatible with NumPy 2.0 and recent Python versions
- **Simple Installation:** Standard `pip install` without system-level dependencies

### **Dual Processing Modes**

- **Bulk Processing:** Efficient vectorized calculations for historical analysis
- **Real-Time Streaming:** Constant-time updates for live market data processing

### **Performance Characteristics**

- **Numba JIT Compilation:** Near-native performance through just-in-time compilation
- **Warmup System:** Optional pre-compilation to eliminate first-call latency
- **API Compatibility:** Compatible with the ta library interface
- **Memory Efficiency:** Streaming mode uses constant memory regardless of data history

## **📊 Performance Comparison**

Based on comprehensive benchmarks with 100,000 data points across multiple technical analysis libraries:

| Aspect                  | TA-Lib              | ta-numba           | ta                  | pandas           | cython               |
| ----------------------- | ------------------- | ------------------ | ------------------- | ---------------- | -------------------- |
| **Installation**        | C compiler required | pip install only   | pip install only    | pip install only | Compilation required |
| **Average Performance** | Fastest (baseline)  | 4.3x slower        | 857x slower         | 94x slower       | 2.5x slower          |
| **Best Cases**          | Fastest overall     | MACD: 3.8x faster  | All cases slower    | All cases slower | Mixed results        |
| **Worst Cases**         | WMA, ADX fastest    | WMA: 33x slower    | PSAR: 8,837x slower | ATR: 13x slower  | Variable performance |
| **Dependency Issues**   | Frequent            | None               | None                | Rare             | Build-time only      |
| **Streaming Support**   | No                  | Yes (15.8x faster) | No                  | No               | No                   |

### **Key Findings**

**Performance Trade-offs:**

- TA-Lib remains the fastest for raw computational speed
- ta-numba provides 4.3x slower performance on average, but eliminates installation complexity
- ta-numba significantly outperforms pure Python libraries (ta: 857x, pandas: 94x faster)

**Installation Reliability:**

- TA-Lib requires C compilation and system dependencies, causing frequent deployment issues
- ta-numba installs reliably across environments with standard Python packaging

**Real-Time Processing:**

- ta-numba's streaming mode provides 15.8x performance improvement over bulk recalculation approaches
- Constant memory usage vs. linear growth in traditional approaches

### **Installation Comparison**

```bash
# TA-Lib installation requirements:
# - C compiler (Visual Studio, GCC, or Clang)
# - System-level TA-Lib library
# - Compatible NumPy version
# - Platform-specific build tools

# ta-numba installation:
pip install ta-numba
# Dependencies: numpy, numba (automatically resolved)
```

## **📦 Installation**

```bash
pip install ta-numba
```

Dependencies: `numpy`, `numba` (automatically installed)


## Rust Backend (v0.3.0+)

Starting with v0.3.0, ta-numba includes an optional Rust backend powered by PyO3 that provides additional speedups:

- **Bulk indicators**: ~2.3x faster than Numba
- **Streaming indicators**: ~20x faster than Numba

### How It Works

On supported platforms (Linux x86_64/aarch64, macOS arm64/x86_64, Windows x64), `pip install ta-numba` automatically installs pre-built Rust extensions. No Rust toolchain needed.

On other platforms, ta-numba falls back to the Numba JIT backend transparently.

### Check Your Backend

```python
import ta_numba
print(ta_numba.get_backend())  # "rust" or "numba"
```

### Building from Source

To build with the Rust backend from source:

```bash
# Requires Rust toolchain (rustup.rs)
pip install maturin
maturin develop --release
```

### API Compatibility

The Rust backend is 100% API-compatible. No code changes needed — the same imports work regardless of backend:

```python
import ta_numba.bulk as bulk
import ta_numba.stream as stream

# These automatically use Rust when available
sma = bulk.trend.sma_numba(prices, n=20)
rsi_stream = stream.RSI(window=14)
```

## **🚀 Quick Start**

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

## **📋 Available Indicators**

### **🔄 Streaming Indicators (New in v0.2.0)**

Real-time indicators with O(1) updates and constant memory usage:

**Trend Indicators (11)**

- `SMA`, `EMA`, `WMA` - Moving averages
- `MACD` - Moving Average Convergence Divergence
- `ADX` - Average Directional Index
- `TRIX` - Triple Exponential Average
- `CCI` - Commodity Channel Index
- `DPO` - Detrended Price Oscillator
- `Aroon` - Aroon Oscillator
- `ParabolicSAR` - Parabolic Stop and Reverse
- `VortexIndicator` - Vortex Indicator

**Momentum Indicators (10)**

- `RSI` - Relative Strength Index
- `Stochastic` - Stochastic Oscillator
- `StochasticRSI` - Stochastic RSI
- `WilliamsR` - Williams %R
- `TSI` - True Strength Index
- `UltimateOscillator` - Ultimate Oscillator
- `AwesomeOscillator` - Awesome Oscillator
- `KAMA` - Kaufman's Adaptive Moving Average
- `PPO` - Percentage Price Oscillator
- `ROC` - Rate of Change

**Volatility Indicators (9)**

- `ATR` - Average True Range
- `BollingerBands` - Bollinger Bands
- `KeltnerChannel` - Keltner Channel
- `DonchianChannel` - Donchian Channel
- `StandardDeviation` - Rolling Standard Deviation
- `Variance` - Rolling Variance
- `TrueRange` - True Range
- `HistoricalVolatility` - Historical Volatility
- `UlcerIndex` - Ulcer Index

**Volume Indicators (10)**

- `MoneyFlowIndex` - Money Flow Index
- `AccDistIndex` - Accumulation/Distribution Index
- `OnBalanceVolume` - On Balance Volume
- `ChaikinMoneyFlow` - Chaikin Money Flow
- `ForceIndex` - Force Index
- `EaseOfMovement` - Ease of Movement
- `VolumePriceTrend` - Volume Price Trend
- `NegativeVolumeIndex` - Negative Volume Index
- `VWAP` - Volume Weighted Average Price
- `VWEMA` - Volume Weighted Exponential Moving Average

### **📊 Bulk Processing Indicators**

All functions accept NumPy arrays for maximum performance.

<details>
<summary><strong>Volume Indicators (10)</strong></summary>

- `ta_numba.volume.money_flow_index`
- `ta_numba.volume.acc_dist_index`
- `ta_numba.volume.on_balance_volume`
- `ta_numba.volume.chaikin_money_flow`
- `ta_numba.volume.force_index`
- `ta_numba.volume.ease_of_movement`
- `ta_numba.volume.volume_price_trend`
- `ta_numba.volume.negative_volume_index`
- `ta_numba.volume.volume_weighted_average_price`
- `ta_numba.volume.volume_weighted_exponential_moving_average`

</details>

<details>
<summary><strong>Volatility Indicators (5)</strong></summary>

- `ta_numba.volatility.average_true_range`
- `ta_numba.volatility.bollinger_bands`
- `ta_numba.volatility.keltner_channel`
- `ta_numba.volatility.donchian_channel`
- `ta_numba.volatility.ulcer_index`

</details>

<details>
<summary><strong>Trend Indicators (15)</strong></summary>

- `ta_numba.trend.sma`
- `ta_numba.trend.ema`
- `ta_numba.trend.wma`
- `ta_numba.trend.macd`
- `ta_numba.trend.adx`
- `ta_numba.trend.vortex_indicator`
- `ta_numba.trend.trix`
- `ta_numba.trend.mass_index`
- `ta_numba.trend.cci`
- `ta_numba.trend.dpo`
- `ta_numba.trend.kst`
- `ta_numba.trend.ichimoku`
- `ta_numba.trend.parabolic_sar`
- `ta_numba.trend.schaff_trend_cycle`
- `ta_numba.trend.aroon`

</details>

<details>
<summary><strong>Momentum Indicators (11)</strong></summary>

- `ta_numba.momentum.rsi`
- `ta_numba.momentum.stochrsi`
- `ta_numba.momentum.tsi`
- `ta_numba.momentum.ultimate_oscillator`
- `ta_numba.momentum.stoch`
- `ta_numba.momentum.williams_r`
- `ta_numba.momentum.awesome_oscillator`
- `ta_numba.momentum.kama`
- `ta_numba.momentum.roc`
- `ta_numba.momentum.ppo`
- `ta_numba.momentum.pvo`

</details>

<details>
<summary><strong>Other Indicators (4)</strong></summary>

- `ta_numba.others.daily_return`
- `ta_numba.others.daily_log_return`
- `ta_numba.others.cumulative_return`
- `ta_numba.others.compound_log_return`

</details>

## **⚡ Performance & Benchmarks**

### **📊 Benchmark Methodology**

**Test Environment:**

- Data Size: 100,000 price points
- Iterations: 3 runs per indicator per library
- Hardware: Standard development machine
- Libraries: ta-numba, ta-lib, ta, pandas, cython, NautilusTrader

**Performance Analysis:**

- **ta-numba delivers substantial performance improvements over pure Python libraries**
- **TA-Lib maintains performance leadership in bulk processing**
- **ta-numba provides unique advantages in streaming scenarios**
- **Installation reliability varies significantly between libraries**

### **📊 Comprehensive Benchmark Results (100K data points)**

**Complete Library Comparison:**

```text
Performance Comparison (Average Time per Run):
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Indicator    | ta           | ta-numba     | ta-lib       | pandas       | cython       | nautilus     | Speedup vs ta | Speedup vs talib | Speedup vs pandas | Speedup vs cython | Speedup vs nautilus
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
SMA          | 0.001196s | 0.001082s | 0.000087s    | 0.000713s    | 0.000058s    | 0.105247s    | 1.11x       | 0.08x           | 0.66x            | 0.05x            | 97.29x
EMA          | 0.000577s | 0.000112s | 0.000332s    | 0.000493s    | 0.000168s    | 0.011398s    | 5.16x       | 2.97x           | 4.41x            | 1.50x            | 101.92x
RSI          | 0.002789s | 0.001355s | 0.000433s    | 0.002412s    | 0.001946s    | 0.062416s    | 2.06x       | 0.32x           | 1.78x            | 1.44x            | 46.06x
MACD         | 0.001635s | 0.000642s | 0.002456s    | 0.001860s    | 0.000666s    | 0.012047s    | 2.55x       | 3.83x           | 2.90x            | 1.04x            | 18.77x
ATR          | 0.205986s | 0.000672s | 0.002262s    | 0.008719s    | 0.001687s    | 0.018718s    | 306.60x       | 3.37x           | 12.98x           | 2.51x            | 27.86x
Bollinger Upper | 0.002052s | 0.001432s | 0.000341s    | 0.002129s    | 0.006004s    | 0.214716s    | 1.43x       | 0.24x           | 1.49x            | 4.19x            | 149.92x
OBV          | 0.000685s | 0.000066s | 0.000224s    | N/A          | 0.000275s    | 14.146200s   | 10.43x       | 3.42x           | N/A              | 4.19x            | 215376.26x
MFI          | 0.482099s | 0.002581s | 0.002374s    | 0.003096s    | 0.006168s    | 0.021110s    | 186.77x       | 0.92x           | 1.20x            | 2.39x            | 8.18x
WMA          | 2.456998s | 0.003013s | 0.000092s    | 0.126318s    | 0.002411s    | 0.339517s    | 815.56x       | 0.03x           | 41.93x           | 0.80x            | 112.70x
VWEMA        | 0.000908s | 0.000822s | 0.029710s    | 0.002095s    | 0.004002s    | 0.058675s    | 1.10x       | 36.13x          | 2.55x            | 4.87x            | 71.35x
ADX          | 0.407531s | 0.003533s | 0.000643s    | 0.012459s    | 0.009984s    | 0.002930s    | 115.34x       | 0.18x           | 3.53x            | 2.83x            | 0.83x
PSAR         | 4.123320s | 0.000467s | 0.000346s    | 0.449931s    | 0.001659s    | 0.007989s    | 8837.04x       | 0.74x           | 964.29x          | 3.56x            | 17.12x
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Summary Statistics:
Average speedup vs ta: 857.10x
Average speedup vs ta-lib: 4.35x
Average speedup vs pandas: 94.34x
Average speedup vs cython: 2.45x
Average speedup vs nautilus: 18002.35x
Identical results vs ta: 11/12
Identical results vs ta-lib: 4/12
Identical results vs cython: 5/12
Identical results vs nautilus: 3/12
```

### **📈 Performance Summary**

**Benchmark Results Analysis:**

**vs Pure Python Libraries:**

- ta library: 857x average speedup (range: 1.1x to 8,837x)
- pandas: 94x average speedup (range: 0.66x to 964x)
- Consistent performance advantage across most indicators

**vs Compiled Libraries:**

- TA-Lib: 0.23x average performance (ta-numba is 4.3x slower on average)
- cython: 2.5x average speedup (mixed results depending on indicator)
- Performance varies significantly by indicator complexity

**Streaming Performance:**

- 15.8x faster than bulk recalculation methods
- Constant O(1) memory usage vs. O(n) growth
- Microsecond-level latency for real-time applications

**Library Selection Criteria:**

- **Choose TA-Lib for**: Maximum performance, stable environment, C compilation acceptable
- **Choose ta-numba for**: Reliable deployment, streaming requirements, Python-only environments
- **Choose ta/pandas for**: Simplicity, small datasets, existing pandas workflows

**Real-Time Streaming Performance (per tick):**

```text
🚀 REAL-TIME STREAMING COMPARISON
============================================================
Simulating live market data feed with continuous price updates...

📊 Generating 100 warmup ticks...
🔥 Warming up JIT compilation...
📈 Initializing streaming indicators...

🎯 SIMULATING 10,000 LIVE MARKET TICKS...
------------------------------------------------------------
Progress:  10% | Avg Bulk:  0.039ms | Avg Streaming:  0.017ms | Speedup:   2.3x
Progress:  20% | Avg Bulk:  0.103ms | Avg Streaming:  0.018ms | Speedup:   5.8x
Progress:  30% | Avg Bulk:  0.174ms | Avg Streaming:  0.019ms | Speedup:   9.0x
Progress:  40% | Avg Bulk:  0.244ms | Avg Streaming:  0.021ms | Speedup:  11.6x
Progress:  50% | Avg Bulk:  0.313ms | Avg Streaming:  0.023ms | Speedup:  13.5x
Progress:  60% | Avg Bulk:  0.378ms | Avg Streaming:  0.023ms | Speedup:  16.2x
Progress:  70% | Avg Bulk:  0.447ms | Avg Streaming:  0.024ms | Speedup:  18.7x
Progress:  80% | Avg Bulk:  0.516ms | Avg Streaming:  0.024ms | Speedup:  21.7x
Progress:  90% | Avg Bulk:  0.589ms | Avg Streaming:  0.024ms | Speedup:  24.3x
Progress: 100% | Avg Bulk:  0.671ms | Avg Streaming:  0.026ms | Speedup:  26.1x

📊 FINAL RESULTS
============================================================
Total ticks processed: 10,000
Lookback window size: 10000

⏱️  TIMING STATISTICS (per tick):
Method                Mean     Median     95%ile     99%ile
-------------------------------------------------------
Bulk                0.347ms     0.346ms     0.673ms     0.699ms
Streaming           0.022ms     0.022ms     0.028ms     0.039ms

🚀 PERFORMANCE IMPROVEMENT:
Average speedup: 15.8x faster
Median speedup: 15.9x faster

💾 MEMORY USAGE COMPARISON:
Bulk approach: O(n) = 10000 * 8 bytes * 7 indicators = 546.9 KB
Streaming approach: O(1) = ~1 KB total (constant)
Memory efficiency: 547x less memory

⚡ LATENCY ANALYSIS:
Bulk 99th percentile: 0.699ms
Streaming 99th percentile: 0.039ms
For HFT (<1ms requirement): ✅ Bulk passes, ✅ Streaming passes
```

### **⚖️ Library Selection Guide**

**Choose TA-Lib when:**

- Maximum computational performance is critical
- Working in controlled environments with stable dependencies
- C compilation infrastructure is available and maintained
- Academic or research work with consistent setup

**Choose ta-numba when:**

- Deploying in diverse environments (Docker, cloud, etc.)
- Requiring real-time streaming capabilities
- Building production systems with reliability requirements
- Working with modern Python/NumPy ecosystems

**Choose ta/pandas when:**

- Working with small to medium datasets
- Prototyping or exploratory analysis
- Integration with existing pandas workflows is important
- Performance requirements are modest

### **🚀 Real-Time Performance Advantage**

```python
# Traditional approach (recalculates entire array each time)
def update_traditional(new_price, history, window=20):
    history.append(new_price)
    return talib.SMA(np.array(history), window)[-1]  # 0.347ms per update

# ta-numba streaming (O(1) update)
sma = stream.SMA(window=20)
def update_streaming(new_price):
    return sma.update(new_price)  # 0.022ms per update - 15.8x faster!
```

````

                  Generating sample data of size 200000 with seed None...
                  Sample data generated.

                  --- Warming up Numba functions (JIT Compilation) ---
                      Warm-up complete.

                  --- Running Benchmarks (5 loops each) ---

                      Discrepancy for TRIX:
                      Mean Absolute Difference: 0.007569
                      Zero Status: Normal

                      First 5 differing values (Index, TA, Numba):
                            43, -0.035422, -0.054566
                            44, -0.021285, -0.035508
                            45, 0.001522, -0.006502
                            46, 0.002349, -0.005051
                            47, 0.012865, 0.008175

                      Discrepancy for MI:
                      Mean Absolute Difference: 6.323e-06
                      Zero Status: Normal

                      First 5 differing values (Index, TA, Numba):
                            40, 24.923410, 25.163110
                            41, 25.099305, 25.298424
                            42, 25.092817, 25.256278
                            43, 25.031762, 25.163705
                            44, 25.043005, 25.150111

                      Discrepancy for STC:
                      Mean Absolute Difference: 4.276e-06
                      Zero Status: Normal

                      First 5 differing values (Index, TA, Numba):
                            71, 16.939844, 17.313568
                            72, 8.469922, 8.656784
                            73, 4.234961, 4.328392
                            74, 15.217372, 15.289315
                            75, 28.028082, 28.091572

                      Discrepancy for TSI:
                      Mean Absolute Difference: 0.0004937
                      Zero Status: Normal

                      First 5 differing values (Index, TA, Numba):
                            37, 8.232642, 1.088498
                            38, 7.628686, 0.899511
                            39, 6.338255, -0.030883
                            40, 6.326458, 0.355236
                            41, 3.863873, -1.721345

                  --- Benchmark Results (Average Time per Run) ---

              -----------------------------------------------------------
              Indicator  | `ta` Library    | Numba Version   | Speedup
              -----------------------------------------------------------
              MFI        | 1.187933s       | 0.005150s       | 230.65x
              ADI        | 0.001475s       | 0.000434s       | 3.40x
              OBV        | 0.001602s       | 0.000122s       | 13.08x
              CMF        | 0.004253s       | 0.001713s       | 2.48x
              FI         | 0.001479s       | 0.000609s       | 2.43x
              EOM        | 0.001648s       | 0.000172s       | 9.58x
              VPT        | 0.002104s       | 0.000451s       | 4.66x
              NVI        | 3.244231s       | 0.001093s       | 2967.43x
              VWAP       | 0.003858s       | 0.001392s       | 2.77x
              VWEMA      | 0.005218s       | 0.002011s       | 2.60x
              ATR        | 0.419494s       | 0.001130s       | 371.32x
              BB         | 0.004472s       | 0.003196s       | 1.40x
              KC         | 0.005683s       | 0.007647s       | 0.74x
              DC         | 0.006115s       | 0.009956s       | 0.61x
              UI         | 0.398492s       | 0.007430s       | 53.63x
              SMA        | 0.001696s       | 0.002453s       | 0.69x
              EMA        | 0.001192s       | 0.000444s       | 2.69x
              WMA        | 5.459586s       | 0.006479s       | 842.68x
              MACD       | 0.003275s       | 0.001290s       | 2.54x
              ADX        | 0.883612s       | 0.007472s       | 118.25x
              Vortex     | 0.016811s       | 0.007960s       | 2.11x
              TRIX       | 0.004868s       | 0.001166s       | 4.18x
              MI         | 0.003594s       | 0.008942s       | 0.40x
              CCI        | 1.055140s       | 0.007558s       | 139.60x
              DPO        | 0.001935s       | 0.002446s       | 0.79x
              KST        | 0.011884s       | 0.031931s       | 0.37x
              Ichimoku   | 0.013384s       | 0.027892s       | 0.48x
              PSAR       | 9.464796s       | 0.001216s       | 7783.20x
              STC        | 0.018517s       | 0.019506s       | 0.95x
              Aroon      | 0.402076s       | 0.005702s       | 70.52x
              RSI        | 0.004719s       | 0.002710s       | 1.74x
              StochRSI   | 0.012424s       | 0.014490s       | 0.86x
              TSI        | 0.004547s       | 0.001771s       | 2.57x
              UO         | 0.034889s       | 0.014549s       | 2.40x
              Stoch      | 0.006982s       | 0.011224s       | 0.62x
              WR         | 0.006880s       | 0.009031s       | 0.76x
              AO         | 0.003143s       | 0.004481s       | 0.70x
              KAMA       | 0.130242s       | 0.001560s       | 83.47x
              ROC        | 0.000777s       | 0.000344s       | 2.26x
              PPO        | 0.003494s       | 0.001294s       | 2.70x
              PVO        | 0.003904s       | 0.001216s       | 3.21x
              DR         | 0.000662s       | 0.000300s       | 2.21x
              DLR        | 0.000803s       | 0.001611s       | 0.50x
              CR         | 0.000388s       | 0.000184s       | 2.11x
              CLR        | 11.993333s      | 1.936194s       | 6.19x
              ```

              -----------------------------------------------------------

                    --- Zero Value Status for All Indicators ---

                     Normal (non-zero values): 44 indicators
                   MFI, ADI, OBV, CMF, FI, EOM, VPT, NVI, VWAP, VWEMA,
                   ATR, BB, KC, DC, UI, ... and 29 more

                   All 44 indicators have normal non-zero values!

                            --- Discrepancy Report ---

                 Indicator     Status            MAD     Zero Status
                      TRIX  Different       0.007569          Normal
                        MI  Different   6.323000e-06          Normal
                       STC  Different   4.276000e-06          Normal
                       TSI  Different   4.937000e-04          Normal
              -----------------------------------------------------------

````

## **🆕 What's New in v0.2.0**

### **🔄 Streaming Indicators**

- **Real-time processing**: O(1) per-update performance
- **Memory efficient**: Constant memory usage regardless of data size
- **Production ready**: Designed for live trading systems
- **Clean API**: Simplified class names (`SMA`, `EMA`, `RSI` vs `SMAStreaming`)

### **⚡ JIT Warmup System**

- **Fast startup**: Eliminate JIT compilation delays in production
- **Bulk warmup**: `ta_numba.warmup.warmup_all()` for all indicators
- **Selective warmup**: Individual indicator warmup available
- **Docker friendly**: Persistent compilation across container restarts

### **🎯 Improved API Design**

- **Dual namespaces**: `ta_numba.bulk` and `ta_numba.stream` for clarity
- **Legacy compatibility**: Existing imports continue to work
- **Better ergonomics**: Cleaner function calls for real-world usage

## **🔄 Migration Guide**

### **From v0.1.0 to v0.2.0**

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

### **From Other Libraries**

```python
# From pandas
df['sma'] = df['close'].rolling(20).mean()
# To ta-numba bulk
sma_values = bulk.trend.sma(df['close'].values, window=20)

# From ta-lib streaming simulation
# (multiple function calls for updates)
# To ta-numba streaming
sma = stream.SMA(window=20)
current_value = sma.update(new_price)
```

## **🛠️ Advanced Usage**

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
# Dockerfile optimization
FROM python:3.11
RUN pip install ta-numba

# Pre-compile indicators at build time
RUN python -c "import ta_numba.warmup; ta_numba.warmup.warmup_all()"

# Your application will start faster
COPY . .
CMD ["python", "your_trading_app.py"]
```

## **📈 Real-World Performance Examples**

### **Backtesting (Bulk Processing)**

```python
import ta_numba.bulk as bulk
import time

# Process 1M price points
prices = np.random.randn(1_000_000).cumsum() + 100

start = time.time()
sma_values = bulk.trend.sma(prices, window=50)
rsi_values = bulk.momentum.rsi(prices, window=14)
macd_line, macd_signal, macd_hist = bulk.trend.macd(prices)
elapsed = time.time() - start

print(f"Processed 1M points in {elapsed:.3f}s")
# Output: Processed 1M points in 0.045s
```

### **Live Trading (Streaming)**

```python
import ta_numba.stream as stream

# Set up indicators
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

    # Generate trading signals
    if all(ind.is_ready for ind in indicators.values()):
        if signals['sma_20'] > signals['sma_50']:
            return "BUY_SIGNAL"
        elif signals['rsi'] > 70:
            return "SELL_SIGNAL"

    return "HOLD"

# Process live feed (microsecond latency)
for price in live_price_feed:
    signal = on_price_update(price)  # ~2-5 microseconds


## **Acknowledgements**

## **🙏 Acknowledgements**

This library builds upon the excellent work of several projects:

- **[Technical Analysis Library (ta)](https://github.com/bukosabino/ta)** by Darío López Padial - API design and calculation logic foundation
- **[Numba](https://numba.pydata.org/)** - JIT compilation technology that makes the performance possible
- **[NumPy](https://numpy.org/)** - Fundamental array operations and mathematical functions

ta-numba extends the original ta library with high-performance Numba compilation and adds real-time streaming capabilities while maintaining mathematical accuracy and API compatibility.

## **📊 Mathematical Documentation**

All indicator implementations are based on established formulas documented in: [`ta-numba.pdf`](ta-numba.pdf)

This document provides:
- Precise mathematical definitions for all indicators
- Implementation details and edge case handling
- Verification against reference implementations
- Performance optimization techniques used

## **🤝 Contributing**

We welcome contributions! Whether it's:
- 🐛 Bug reports and fixes
- 📈 New indicator implementations
- ⚡ Performance optimizations
- 📚 Documentation improvements
- 🧪 Test coverage expansion

Please see our [contributing guidelines](CONTRIBUTING.md) for details.

## **📝 License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**⭐ If ta-numba helps your trading or research, please give us a star on GitHub!**
```
