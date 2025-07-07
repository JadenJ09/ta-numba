# **TA-Numba: High-Performance Technical Analysis Library**

**ta-numba** is a Python library for financial technical analysis that provides a high-performance, Numba-accelerated alternative to the popular ta library.

The primary goal of this project is to offer a significant speed increase for calculating technical indicators, especially on large datasets, making it ideal for backtesting, real-time analysis, and large-scale quantitative research.

In developing **ta-numba**, special care was taken to ensure mathematical correctness and transparency. The indicator implementations are based on well-established formulas, as documented in: [`ta-numba/ta-numba.pdf`](ta-numba/ta-numba.pdf). This document details the precise mathematical definitions and serves as the authoritative source for all indicator calculations in this library.

## **Key Features**

- **High Performance:** Uses Numba's just-in-time (JIT) compilation to dramatically accelerate indicator calculations, often by orders of magnitude (100x to 8000x+ speedups on iterative indicators).
- **1-to-1 Compatibility:** Functions are designed to be drop-in replacements for the ta library, producing identical output to ensure reproducibility and easy integration into existing projects.
- **Pure NumPy/Numba:** Operates directly on NumPy arrays, avoiding the overhead of pandas DataFrames in performance-critical calculations.
- **Simple & Clean API:** Provides a straightforward, functional API organized into logical modules: volume, volatility, trend, momentum, and others.

## **Why ta-numba?**

While the original ta library is an excellent and widely-used tool, its reliance on pandas can lead to performance bottlenecks, particularly with iterative indicators that are not easily vectorized. ta-numba solves this problem by compiling these complex loops into highly optimized machine code, offering performance that rivals lower-level languages like C++ or Cython without sacrificing the ease of use of Python.

## **Installation**

You can install ta-numba directly from PyPI:

pip install ta-numba

The library requires numpy, pandas, and numba as dependencies, which will be installed automatically.

## **Quick Start & Usage Example**

The API is designed to be simple and familiar. You can import the library and use the indicator functions directly on your pandas Series or NumPy arrays.

```python
import pandas as pd
import numpy as np
import ta_numba.trend as trend
import ta_numba.momentum as momentum

# Load your data (example with a pandas DataFrame)
# df should have 'High', 'Low', 'Close', 'Volume' columns
# ...

# Example 1: Calculate a 20-period Simple Moving Average
sma_20 = trend.sma(df['Close'].values, window=20)

# The result is a NumPy array. You can add it back to your DataFrame:
df['SMA_20'] = sma_20

# Example 2: Calculate the Parabolic SAR
psar = trend.parabolic_sar(df['High'].values, df['Low'].values, df['Close'].values)
df['PSAR'] = psar

# Example 3: Calculate RSI
rsi = momentum.rsi(df['Close'].values, n=14)
df['RSI'] = rsi

print(df.tail())
```

## **Available Indicators**

All functions accept NumPy arrays as input for maximum performance.
Below is a categorized list of all available indicators. Click to expand each section:

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

## **Benchmarking and Performance**

```

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

```

## **Acknowledgements**

This library's API design and calculation logic are based on the excellent work of the original [Technical Analysis Library (ta)](https://github.com/bukosabino/ta) by Darío López Padial. ta-numba aims to provide a performance-focused alternative while respecting the established and well-regarded API of the original project.

## **License**

This project is licensed under the MIT License \- see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
