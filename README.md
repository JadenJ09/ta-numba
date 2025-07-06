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

## **Acknowledgements**

This library's API design and calculation logic are based on the excellent work of the original [Technical Analysis Library (ta)](https://github.com/bukosabino/ta) by Darío López Padial. ta-numba aims to provide a performance-focused alternative while respecting the established and well-regarded API of the original project.

## **License**

This project is licensed under the MIT License \- see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
