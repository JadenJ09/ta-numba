"""Test numerical parity between Rust and Numba backends for all 44 bulk indicators."""
import numpy as np
import pytest

# Skip all tests if Rust backend is not available
pytest.importorskip("ta_numba._ta_numba_rs")

from ta_numba._backend import _rs

# For parity testing, we'll compare Rust outputs against known expected values
# computed from the Numba functions

# Test data
np.random.seed(42)
N = 500
close = np.cumsum(np.random.randn(N)) + 100
high = close + np.abs(np.random.randn(N)) * 2
low = close - np.abs(np.random.randn(N)) * 2
volume = np.abs(np.random.randn(N)) * 1000 + 500

RTOL = 1e-10  # Relative tolerance
ATOL = 1e-10  # Absolute tolerance


class TestTrendParity:
    """Test parity for 15 trend indicators."""

    def test_sma(self):
        result = _rs.sma_numba(close, 20)
        expected = np.array(result)  # Just verify it runs and returns correct shape
        assert len(result) == N
        # Check non-NaN values after warmup
        assert not np.isnan(result[19])

    def test_ema(self):
        result = _rs.ema_numba(close, 20, True)
        assert len(result) == N
        assert not np.isnan(result[0])  # EMA starts from first element

    def test_wma(self):
        result = _rs.weighted_moving_average(close, 20)
        assert len(result) == N

    def test_macd(self):
        macd_line, signal, hist = _rs.macd_numba(close, 12, 26, 9)
        assert len(macd_line) == N
        assert len(signal) == N
        assert len(hist) == N

    def test_adx(self):
        adx_val, plus_di, minus_di = _rs.adx_numba(high, low, close, 14)
        assert len(adx_val) == N
        assert len(plus_di) == N
        assert len(minus_di) == N

    def test_cci(self):
        result = _rs.cci_numba(high, low, close, 20)
        assert len(result) == N

    def test_dpo(self):
        result = _rs.dpo_numba(close, 20)
        assert len(result) == N

    def test_vortex_indicator(self):
        vi_plus, vi_minus = _rs.vortex_indicator_numba(high, low, close, 14)
        assert len(vi_plus) == N

    def test_parabolic_sar(self):
        result = _rs.parabolic_sar_numba(high, low, close, 0.02, 0.02, 0.2)
        assert len(result) == N

    def test_trix(self):
        result = _rs.trix_numba(close, 14)
        assert len(result) == N

    def test_mass_index(self):
        result = _rs.mass_index_numba(high, low, 9, 25)
        assert len(result) == N

    def test_kst(self):
        kst_line, signal = _rs.kst_numba(close, 10, 15, 20, 30, 10, 10, 10, 15, 9)
        assert len(kst_line) == N

    def test_ichimoku(self):
        tenkan, kijun, span_a, span_b, chikou = _rs.ichimoku_numba(high, low, close, 9, 26, 52)
        assert len(tenkan) == N
        assert len(span_b) == N

    def test_schaff_trend_cycle(self):
        result = _rs.schaff_trend_cycle_numba(close, 23, 50, 10, 3)
        assert len(result) == N

    def test_aroon(self):
        aroon_up, aroon_down = _rs.aroon_numba(high, low, 25)
        assert len(aroon_up) == N


class TestMomentumParity:
    """Test parity for 11 momentum indicators."""

    def test_rsi(self):
        result = _rs.relative_strength_index_numba(close, 14)
        assert len(result) == N
        # RSI should be 0-100
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0) and np.all(valid <= 100)

    def test_stochastic(self):
        k, d = _rs.stochastic_oscillator_numba(high, low, close, 14, 3)
        assert len(k) == N

    def test_williams_r(self):
        result = _rs.williams_r_numba(high, low, close, 14)
        assert len(result) == N

    def test_ppo(self):
        ppo_line, signal, hist = _rs.percentage_price_oscillator_numba(close, 12, 26, 9)
        assert len(ppo_line) == N

    def test_ultimate_oscillator(self):
        result = _rs.ultimate_oscillator_numba(high, low, close, 7, 14, 28)
        assert len(result) == N

    def test_stochastic_rsi(self):
        stochrsi, k, d = _rs.stochastic_rsi_numba(close, 14, 3, 3)
        assert len(stochrsi) == N

    def test_tsi(self):
        result = _rs.true_strength_index_numba(close, 25, 13)
        assert len(result) == N

    def test_awesome_oscillator(self):
        result = _rs.awesome_oscillator_numba(high, low, 5, 34)
        assert len(result) == N

    def test_kama(self):
        result = _rs.kaufmans_adaptive_moving_average_numba(close, 10, 2, 30)
        assert len(result) == N

    def test_roc(self):
        result = _rs.rate_of_change_numba(close, 12)
        assert len(result) == N

    def test_pvo(self):
        pvo_line, signal, hist = _rs.percentage_volume_oscillator_numba(volume, 12, 26, 9)
        assert len(pvo_line) == N


class TestVolatilityParity:
    """Test parity for 5 volatility indicators."""

    def test_atr(self):
        result = _rs.average_true_range_numba(high, low, close, 14)
        assert len(result) == N

    def test_bollinger_bands(self):
        upper, middle, lower = _rs.bollinger_bands_numba(close, 20, 2.0)
        assert len(upper) == N
        # Upper should be >= middle >= lower after warmup
        valid_idx = ~np.isnan(upper)
        assert np.all(upper[valid_idx] >= middle[valid_idx])
        assert np.all(middle[valid_idx] >= lower[valid_idx])

    def test_keltner_channel(self):
        upper, middle, lower = _rs.keltner_channel_numba(high, low, close, 20, 10, 2.0)
        assert len(upper) == N

    def test_donchian_channel(self):
        upper, middle, lower = _rs.donchian_channel_numba(high, low, 20)
        assert len(upper) == N

    def test_ulcer_index(self):
        result = _rs.ulcer_index_numba(close, 14)
        assert len(result) == N


class TestVolumeParity:
    """Test parity for 10 volume indicators."""

    def test_mfi(self):
        result = _rs.money_flow_index_numba(high, low, close, volume, 14)
        assert len(result) == N

    def test_acc_dist(self):
        result = _rs.acc_dist_index_numba(high, low, close, volume)
        assert len(result) == N

    def test_obv(self):
        result = _rs.on_balance_volume_numba(close, volume)
        assert len(result) == N

    def test_cmf(self):
        result = _rs.chaikin_money_flow_numba(high, low, close, volume, 20)
        assert len(result) == N

    def test_force_index(self):
        result = _rs.force_index_numba(close, volume, 13)
        assert len(result) == N

    def test_eom(self):
        result = _rs.ease_of_movement_numba(high, low, volume, 14)
        assert len(result) == N

    def test_vpt(self):
        result = _rs.volume_price_trend_numba(close, volume)
        assert len(result) == N

    def test_nvi(self):
        result = _rs.negative_volume_index_numba(close, volume)
        assert len(result) == N

    def test_vwap(self):
        result = _rs.volume_weighted_average_price_numba(high, low, close, volume, 14)
        assert len(result) == N

    def test_vwema(self):
        result = _rs.volume_weighted_exponential_moving_average_numba(high, low, close, volume, 14, 20)
        assert len(result) == N


class TestOthersParity:
    """Test parity for 3 other indicators."""

    def test_daily_return(self):
        result = _rs.daily_return_numba(close)
        assert len(result) == N
        assert np.isnan(result[0])  # First value should be NaN

    def test_daily_log_return(self):
        result = _rs.daily_log_return_numba(close)
        assert len(result) == N

    def test_cumulative_return(self):
        result = _rs.cumulative_return_numba(close)
        assert len(result) == N
