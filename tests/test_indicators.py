# tests/test_indicators.py

import pandas as pd
import numpy as np
import ta
import time

# Use the installed ta-numba package instead of local imports
    
# Import specific functions for direct use
from ta_numba.volume import (
    money_flow_index_numba,
    acc_dist_index_numba,
    on_balance_volume_numba,
    chaikin_money_flow_numba,
    force_index_numba,
    ease_of_movement_numba,
    volume_price_trend_numba,
    negative_volume_index_numba,
    volume_weighted_average_price_numba,
    volume_weighted_exponential_moving_average_numba
)

from ta_numba.volatility import (
    average_true_range_numba,
    bollinger_bands_numba,
    keltner_channel_numba,
    donchian_channel_numba,
    ulcer_index_numba
)

from ta_numba.trend import (
    sma_numba,
    ema_numba,
    weighted_moving_average,
    macd_numba,
    adx_numba,
    vortex_indicator_numba,
    trix_numba,
    mass_index_numba,
    cci_numba,
    dpo_numba,
    kst_numba,
    ichimoku_numba,
    parabolic_sar_numba,
    schaff_trend_cycle_numba,
    aroon_numba
)

from ta_numba.momentum import (
    relative_strength_index_numba,
    stochastic_rsi_numba,
    true_strength_index_numba,
    ultimate_oscillator_numba,
    stochastic_oscillator_numba,
    williams_r_numba,
    awesome_oscillator_numba,
    kaufmans_adaptive_moving_average_numba,
    rate_of_change_numba,
    percentage_price_oscillator_numba,
    percentage_volume_oscillator_numba
)

from ta_numba.others import (
    daily_return_numba,
    daily_log_return_numba,
    cumulative_return_numba,
    compound_log_return_numba
)

from ta_numba.helpers import _sma_numba, _ema_numba_unadjusted


# ==============================================================================
# Test Data Fixture
# ==============================================================================

def generate_data(size=200000, seed=None):
    """Generates a large sample DataFrame for benchmarking."""
    print(f"Generating sample data of size {size} with seed {seed}...")
    np.random.seed(seed)
    data = {
        'High': np.random.random(size) * 20 + 100,
        'Low': np.random.random(size) * 20 + 80,
        'Close': np.random.random(size) * 20 + 90,
        'Volume': np.random.randint(10000, 1000000, size=size)
    }
    df = pd.DataFrame(data)
    df['Low'] = df[['High', 'Low']].min(axis=1)
    df['Close'] = np.clip(df['Close'], df['Low'], df['High'])
    df['Volume'] = df['Volume'].astype(np.float64)
    print("Sample data generated.")
    return df

def compare_series(s1: pd.Series, s2: pd.Series, name: str, tolerance=1e-5):
    """Compares two pandas Series for equality and calculates the difference."""
    s1_clean = s1.dropna()
    s2_clean = s2.dropna()
    
    if s1_clean.empty and s2_clean.empty:
        return "Identical", 0.0, "Both empty"
    if s1_clean.empty or s2_clean.empty:
        return "One series is empty", 0.0, "One empty"

    aligned_s1, aligned_s2 = s1_clean.align(s2_clean, join='inner')
    
    if aligned_s1.empty or aligned_s2.empty:
        return "No overlapping data", 0.0, "No overlap"

    # Check for all-zero values
    ta_values = np.asarray(aligned_s1.values)
    numba_values = np.asarray(aligned_s2.values)
    
    ta_all_zero = np.all(ta_values == 0)
    numba_all_zero = np.all(numba_values == 0)
    
    zero_status = "Normal"
    if ta_all_zero and numba_all_zero:
        zero_status = "Both all-zero"
    elif ta_all_zero:
        zero_status = "TA all-zero"
    elif numba_all_zero:
        zero_status = "Numba all-zero"

    are_equal = np.allclose(ta_values, numba_values, atol=tolerance, equal_nan=True)
    mean_abs_diff = np.mean(np.abs(ta_values - numba_values))

    if are_equal:
        return "Identical", 0.0, zero_status
    else:
        print(f"\n  Discrepancy for {name}:")
        print(f"    Mean Absolute Difference: {mean_abs_diff:.4g}")
        print(f"    Zero Status: {zero_status}")
        print(f"    First 5 differing values (Index, TA, Numba):")
        diff_indices = np.where(~np.isclose(ta_values, numba_values, atol=tolerance, equal_nan=True))[0]
        for i in diff_indices[:5]:
             idx = aligned_s1.index[i]
             print(f"      {idx}, {aligned_s1.iloc[i]:.6f}, {aligned_s2.iloc[i]:.6f}")
        return f"Different", mean_abs_diff, zero_status

# Manual VWEMA implementation for benchmarking (since ta does not provide it)
def vwema_reference(df, n_vwma=14, n_ema=20):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    vwma = (tp * df['Volume']).rolling(window=n_vwma).sum() / df['Volume'].rolling(window=n_vwma).sum()
    vwema = vwma.ewm(span=n_ema, adjust=True).mean()
    return vwema

def compound_log_return_reference(close_series):
    """Reference implementation for compound log return using pandas."""
    log_returns = np.log(close_series / close_series.shift(1))
    compound_log_returns = pd.Series(index=close_series.index, dtype=float)
    compound_log_returns.iloc[0] = np.nan  # First value is NaN
    
    for i in range(1, len(close_series)):
        compound_log_returns.iloc[i] = (np.exp(log_returns.iloc[1:i+1].sum()) - 1) * 100.0
    
    return compound_log_returns

def run_benchmarks(df: pd.DataFrame, num_loops: int = 5):
    """Runs the benchmark for all indicators."""
    
    high, low, close, volume = df['High'].values, df['Low'].values, df['Close'].values, df['Volume'].values
    
    results = []
    discrepancies = []
    all_zero_statuses = []

    indicators_to_benchmark = [
        # Volume
        ("MFI", lambda: ta.volume.MFIIndicator(df['High'], df['Low'], df['Close'], df['Volume']).money_flow_index(), lambda: money_flow_index_numba(high, low, close, volume)),
        ("ADI", lambda: ta.volume.AccDistIndexIndicator(df['High'], df['Low'], df['Close'], df['Volume']).acc_dist_index(), lambda: acc_dist_index_numba(high, low, close, volume)),
        ("OBV", lambda: ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume(), lambda: on_balance_volume_numba(close, volume)),
        ("CMF", lambda: ta.volume.ChaikinMoneyFlowIndicator(df['High'], df['Low'], df['Close'], df['Volume']).chaikin_money_flow(), lambda: chaikin_money_flow_numba(high, low, close, volume)),
        ("FI", lambda: ta.volume.ForceIndexIndicator(df['Close'], df['Volume']).force_index(), lambda: force_index_numba(close, volume)),
        ("EOM", lambda: ta.volume.EaseOfMovementIndicator(df['High'], df['Low'], df['Volume']).ease_of_movement(), lambda: ease_of_movement_numba(high, low, volume)),
        ("VPT", lambda: ta.volume.VolumePriceTrendIndicator(df['Close'], df['Volume']).volume_price_trend(), lambda: volume_price_trend_numba(close, volume)),
        ("NVI", lambda: ta.volume.NegativeVolumeIndexIndicator(df['Close'], df['Volume']).negative_volume_index(), lambda: negative_volume_index_numba(close, volume)),
        ("VWAP", lambda: ta.volume.VolumeWeightedAveragePrice(df['High'], df['Low'], df['Close'], df['Volume']).volume_weighted_average_price(), lambda: volume_weighted_average_price_numba(high, low, close, volume)),
        ("VWEMA", lambda: vwema_reference(df, n_vwma=14, n_ema=20), lambda: volume_weighted_exponential_moving_average_numba(high, low, close, volume, n_vwma=14, n_ema=20)),
        # Volatility
        ("ATR", lambda: ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range(), lambda: average_true_range_numba(high, low, close)),
        ("BB", lambda: ta.volatility.BollingerBands(df['Close']).bollinger_hband(), lambda: bollinger_bands_numba(close)[0]),
        ("KC", lambda: ta.volatility.KeltnerChannel(df['High'], df['Low'], df['Close']).keltner_channel_hband(), lambda: keltner_channel_numba(high, low, close)[0]),
        ("DC", lambda: ta.volatility.DonchianChannel(df['High'], df['Low'], df['Close']).donchian_channel_hband(), lambda: donchian_channel_numba(high, low)[0]),
        ("UI", lambda: ta.volatility.UlcerIndex(df['Close']).ulcer_index(), lambda: ulcer_index_numba(close)),
        # Trend
        ("SMA", lambda: ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator(), lambda: _sma_numba(close)),
        ("EMA", lambda: ta.trend.EMAIndicator(df['Close']).ema_indicator(), lambda: _ema_numba_unadjusted(close, 14)),
        ("WMA", lambda: ta.trend.WMAIndicator(df['Close'], window=20).wma(), lambda: weighted_moving_average(close, 20)),
        ("MACD", lambda: ta.trend.MACD(df['Close']).macd(), lambda: macd_numba(close)[0]),
        ("ADX", lambda: ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx(), lambda: adx_numba(high, low, close)[0]),
        ("Vortex", lambda: ta.trend.VortexIndicator(df['High'], df['Low'], df['Close']).vortex_indicator_pos(), lambda: vortex_indicator_numba(high, low, close)[0]),
        ("TRIX", lambda: ta.trend.TRIXIndicator(df['Close']).trix(), lambda: trix_numba(close)),
        ("MI", lambda: ta.trend.MassIndex(df['High'], df['Low']).mass_index(), lambda: mass_index_numba(high, low)),
        ("CCI", lambda: ta.trend.CCIIndicator(df['High'], df['Low'], df['Close']).cci(), lambda: cci_numba(high, low, close)),
        ("DPO", lambda: ta.trend.DPOIndicator(df['Close']).dpo(), lambda: dpo_numba(close)),
        ("KST", lambda: ta.trend.KSTIndicator(df['Close']).kst(), lambda: kst_numba(close)[0]),
        ("Ichimoku", lambda: ta.trend.IchimokuIndicator(df['High'], df['Low']).ichimoku_a(), lambda: ichimoku_numba(high, low, close)[2]),
        ("PSAR", lambda: ta.trend.PSARIndicator(df['High'], df['Low'], df['Close']).psar(), lambda: parabolic_sar_numba(high, low, close)),
        ("STC", lambda: ta.trend.STCIndicator(df['Close']).stc(), lambda: schaff_trend_cycle_numba(close)),
        ("Aroon", lambda: ta.trend.AroonIndicator(df['High'], df['Low']).aroon_up(), lambda: aroon_numba(high, low)[0]),
        # Momentum
        ("RSI", lambda: ta.momentum.RSIIndicator(df['Close']).rsi(), lambda: relative_strength_index_numba(close)),
        ("StochRSI", lambda: ta.momentum.StochRSIIndicator(df['Close']).stochrsi(), lambda: stochastic_rsi_numba(close)[0]),
        ("TSI", lambda: ta.momentum.TSIIndicator(df['Close']).tsi(), lambda: true_strength_index_numba(close)),
        ("UO", lambda: ta.momentum.UltimateOscillator(df['High'], df['Low'], df['Close']).ultimate_oscillator(), lambda: ultimate_oscillator_numba(high, low, close)),
        ("Stoch", lambda: ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch(), lambda: stochastic_oscillator_numba(high, low, close)[0]),
        ("WR", lambda: ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Close']).williams_r(), lambda: williams_r_numba(high, low, close)),
        ("AO", lambda: ta.momentum.AwesomeOscillatorIndicator(df['High'], df['Low']).awesome_oscillator(), lambda: awesome_oscillator_numba(high, low)),
        ("KAMA", lambda: ta.momentum.KAMAIndicator(df['Close']).kama(), lambda: kaufmans_adaptive_moving_average_numba(close)),
        ("ROC", lambda: ta.momentum.ROCIndicator(df['Close']).roc(), lambda: rate_of_change_numba(close)),
        ("PPO", lambda: ta.momentum.PercentagePriceOscillator(df['Close']).ppo(), lambda: percentage_price_oscillator_numba(close)[0]),
        ("PVO", lambda: ta.momentum.PercentageVolumeOscillator(df['Volume']).pvo(), lambda: percentage_volume_oscillator_numba(volume)[0]),
        # Others
        ("DR", lambda: ta.others.daily_return(df['Close']), lambda: daily_return_numba(close)),
        ("DLR", lambda: ta.others.daily_log_return(df['Close']), lambda: daily_log_return_numba(close)),
        ("CR", lambda: ta.others.cumulative_return(df['Close']), lambda: cumulative_return_numba(close)),
        ("CLR", lambda: compound_log_return_reference(df['Close']), lambda: compound_log_return_numba(close)),
    ]
    
    print("\n--- Warming up Numba functions (JIT Compilation) ---")
    for name, _, numba_func in indicators_to_benchmark:
        _ = numba_func()
    print("Warm-up complete.")

    print(f"\n--- Running Benchmarks ({num_loops} loops each) ---\n")

    for name, ta_func, numba_func in indicators_to_benchmark:
        # print(f"Benchmarking {name}...") # Can be noisy, disable for clean output
        
        ta_start_time = time.perf_counter()
        for _ in range(num_loops):
            ta_result = ta_func()
        ta_end_time = time.perf_counter()
        ta_avg_time = (ta_end_time - ta_start_time) / num_loops

        numba_start_time = time.perf_counter()
        for _ in range(num_loops):
            numba_result_raw = numba_func()
        numba_end_time = time.perf_counter()
        numba_avg_time = (numba_end_time - numba_start_time) / num_loops
        
        if isinstance(numba_result_raw, tuple):
             numba_result = pd.Series(numba_result_raw[0], index=df.index)
        else:
             numba_result = pd.Series(numba_result_raw, index=df.index)

        status, diff, zero_status = compare_series(ta_result, numba_result, name)
        
        # Store zero status for all indicators
        all_zero_statuses.append({'Indicator': name, 'Zero Status': zero_status})
        
        if "Different" in status:
            discrepancies.append({'Indicator': name, 'Status': status, 'MAD': diff, 'Zero Status': zero_status})

        speedup = ta_avg_time / numba_avg_time if numba_avg_time > 0 else float('inf')
        results.append({
            'Indicator': name,
            '`ta` Library': f"{ta_avg_time:.6f}s",
            'Numba Version': f"{numba_avg_time:.6f}s",
            'Speedup': f"{speedup:.2f}x"
        })

    print("\n--- Benchmark Results (Average Time per Run) ---")
    results_df = pd.DataFrame(results)
    header = f"{'Indicator':<10} | {'`ta` Library':<15} | {'Numba Version':<15} | {'Speedup':<10}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for _, row in results_df.iterrows():
        print(f"{row['Indicator']:<10} | {row['`ta` Library']:<15} | {row['Numba Version']:<15} | {row['Speedup']:<10}")
    print("-" * len(header))

    # Show Zero Status for ALL indicators
    print("\n--- Zero Value Status for All Indicators ---")
    zero_status_df = pd.DataFrame(all_zero_statuses)
    
    # Group by zero status for cleaner display
    normal_indicators = zero_status_df[zero_status_df['Zero Status'] == 'Normal']['Indicator'].tolist()
    problematic_indicators = zero_status_df[zero_status_df['Zero Status'] != 'Normal']
    
    print(f"✅ Normal (non-zero values): {len(normal_indicators)} indicators")
    if len(normal_indicators) <= 20:  # Show all if not too many
        print(f"   {', '.join(normal_indicators)}")
    else:  # Show first few and count
        print(f"   {', '.join(normal_indicators[:15])}, ... and {len(normal_indicators)-15} more")
    
    if not problematic_indicators.empty:
        print(f"\n⚠️  Problematic indicators: {len(problematic_indicators)}")
        for _, row in problematic_indicators.iterrows():
            print(f"   {row['Indicator']}: {row['Zero Status']}")
    else:
        print(f"\n🎉 All {len(normal_indicators)} indicators have normal non-zero values!")

    if discrepancies:
        print("\n--- Discrepancy Report ---")
        discrepancies_df = pd.DataFrame(discrepancies)
        print(discrepancies_df.to_string())
        
        # Summary of zero status issues
        zero_issues = discrepancies_df[discrepancies_df['Zero Status'] != 'Normal']
        if not zero_issues.empty:
            print("\n--- Zero Value Issues ---")
            for _, row in zero_issues.iterrows():
                print(f"{row['Indicator']}: {row['Zero Status']}")
    else:
        print("\nSUCCESS: All indicator outputs are identical within tolerance.")

if __name__ == '__main__':
    ohlcv_df = generate_data(size=200000)
    run_benchmarks(ohlcv_df, num_loops=5)
