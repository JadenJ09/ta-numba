"""
ta-numba Streaming Module

High-performance streaming technical indicators for real-time trading systems.
Provides O(1) per-update performance with minimal memory footprint.

Usage:
    # Import streaming indicators
    from ta_numba.stream import SMA, EMA, RSI, MACD

    # Or import the whole namespace
    import ta_numba.stream as stream

    # Create indicators (cleaner class names)
    sma = stream.SMA(window=20)
    ema = stream.EMA(window=20)
    rsi = stream.RSI(window=14)
    macd = stream.MACD()

    # Update with new prices
    for price in live_prices:
        sma_value = sma.update(price)
        ema_value = ema.update(price)
        rsi_value = rsi.update(price)
        macd_value = macd.update(price)

    # Get current state
    print(f"Current SMA: {sma.current_value}")
    print(f"Is ready: {sma.is_ready}")
"""

from .base import StreamingIndicator, StreamingIndicatorMultiple

# Momentum indicators
from .momentum import AwesomeOscillatorStreaming
from .momentum import AwesomeOscillatorStreaming as AwesomeOscillator
from .momentum import KAMAStreaming
from .momentum import KAMAStreaming as KAMA
from .momentum import MomentumStreaming
from .momentum import MomentumStreaming as Momentum
from .momentum import PPOStreaming
from .momentum import PPOStreaming as PPO
from .momentum import ROCStreaming
from .momentum import ROCStreaming as ROC
from .momentum import RSIStreaming
from .momentum import RSIStreaming as RSI
from .momentum import StochasticRSIStreaming
from .momentum import StochasticRSIStreaming as StochasticRSI
from .momentum import StochasticStreaming
from .momentum import StochasticStreaming as Stochastic
from .momentum import TSIStreaming
from .momentum import TSIStreaming as TSI
from .momentum import UltimateOscillatorStreaming
from .momentum import UltimateOscillatorStreaming as UltimateOscillator
from .momentum import WilliamsRStreaming
from .momentum import WilliamsRStreaming as WilliamsR

# Others (returns) indicators
from .others import (
    CalmarRatioStreaming,
    CompoundLogReturnStreaming,
    CumulativeReturnStreaming,
    DailyLogReturnStreaming,
    DailyReturnStreaming,
    MaxDrawdownStreaming,
    RollingReturnStreaming,
    SharpeRatioStreaming,
    VolatilityStreaming,
)

# Legacy names for backward compatibility (keep original names available)
# Trend indicators (cleaner names without "Streaming" suffix)
from .trend import ADXStreaming
from .trend import ADXStreaming as ADX
from .trend import AroonStreaming
from .trend import AroonStreaming as Aroon
from .trend import CCIStreaming
from .trend import CCIStreaming as CCI
from .trend import DPOStreaming
from .trend import DPOStreaming as DPO
from .trend import EMAStreaming
from .trend import EMAStreaming as EMA
from .trend import MACDStreaming
from .trend import MACDStreaming as MACD
from .trend import ParabolicSARStreaming
from .trend import ParabolicSARStreaming as ParabolicSAR
from .trend import SMAStreaming
from .trend import SMAStreaming as SMA
from .trend import TRIXStreaming
from .trend import TRIXStreaming as TRIX
from .trend import VortexIndicatorStreaming
from .trend import VortexIndicatorStreaming as VortexIndicator
from .trend import WMAStreaming
from .trend import WMAStreaming as WMA

# Volatility indicators
from .volatility import ATRStreaming
from .volatility import ATRStreaming as ATR
from .volatility import BBandsStreaming
from .volatility import BBandsStreaming as BollingerBands
from .volatility import DonchianChannelStreaming
from .volatility import DonchianChannelStreaming as DonchianChannel
from .volatility import HistoricalVolatilityStreaming
from .volatility import HistoricalVolatilityStreaming as HistoricalVolatility
from .volatility import KeltnerChannelStreaming
from .volatility import KeltnerChannelStreaming as KeltnerChannel
from .volatility import RangeStreaming
from .volatility import RangeStreaming as TrueRange
from .volatility import StandardDeviationStreaming
from .volatility import StandardDeviationStreaming as StandardDeviation
from .volatility import UlcerIndexStreaming
from .volatility import UlcerIndexStreaming as UlcerIndex
from .volatility import VarianceStreaming
from .volatility import VarianceStreaming as Variance

# Volume indicators
from .volume import AccDistIndexStreaming
from .volume import AccDistIndexStreaming as AccDistIndex
from .volume import ChaikinMoneyFlowStreaming
from .volume import ChaikinMoneyFlowStreaming as ChaikinMoneyFlow
from .volume import EaseOfMovementStreaming
from .volume import EaseOfMovementStreaming as EaseOfMovement
from .volume import ForceIndexStreaming
from .volume import ForceIndexStreaming as ForceIndex
from .volume import MoneyFlowIndexStreaming
from .volume import MoneyFlowIndexStreaming as MoneyFlowIndex
from .volume import NegativeVolumeIndexStreaming
from .volume import NegativeVolumeIndexStreaming as NegativeVolumeIndex
from .volume import OnBalanceVolumeStreaming
from .volume import OnBalanceVolumeStreaming as OnBalanceVolume
from .volume import VolumePriceTrendStreaming
from .volume import VolumePriceTrendStreaming as VolumePriceTrend
from .volume import VWAPStreaming
from .volume import VWAPStreaming as VWAP
from .volume import VWEMAStreaming
from .volume import VWEMAStreaming as VWEMA

__all__ = [
    # Base classes
    "StreamingIndicator",
    "StreamingIndicatorMultiple",
    # Trend indicators
    "SMAStreaming",
    "EMAStreaming",
    "WMAStreaming",
    "MACDStreaming",
    "ADXStreaming",
    "VortexIndicatorStreaming",
    "TRIXStreaming",
    "CCIStreaming",
    "DPOStreaming",
    "AroonStreaming",
    "ParabolicSARStreaming",
    # Momentum indicators
    "RSIStreaming",
    "StochasticStreaming",
    "WilliamsRStreaming",
    "ROCStreaming",
    "UltimateOscillatorStreaming",
    "StochasticRSIStreaming",
    "TSIStreaming",
    "AwesomeOscillatorStreaming",
    "KAMAStreaming",
    "PPOStreaming",
    "MomentumStreaming",
    # Volatility indicators
    "ATRStreaming",
    "BBandsStreaming",
    "KeltnerChannelStreaming",
    "DonchianChannelStreaming",
    "StandardDeviationStreaming",
    "VarianceStreaming",
    "RangeStreaming",
    "HistoricalVolatilityStreaming",
    "UlcerIndexStreaming",
    # Volume indicators
    "MoneyFlowIndexStreaming",
    "AccDistIndexStreaming",
    "OnBalanceVolumeStreaming",
    "ChaikinMoneyFlowStreaming",
    "ForceIndexStreaming",
    "EaseOfMovementStreaming",
    "VolumePriceTrendStreaming",
    "NegativeVolumeIndexStreaming",
    "VWAPStreaming",
    "VWEMAStreaming",
    # Others (returns) indicators
    "DailyReturnStreaming",
    "DailyLogReturnStreaming",
    "CumulativeReturnStreaming",
    "CompoundLogReturnStreaming",
    "RollingReturnStreaming",
    "VolatilityStreaming",
    "SharpeRatioStreaming",
    "MaxDrawdownStreaming",
    "CalmarRatioStreaming",
]


# --- Rust backend dispatch (transparent acceleration) ---
# Override Numba streaming classes with Rust-backed wrappers when available.
# Python-only classes (no Rust equivalent) are NOT overridden:
#   CompoundLogReturnStreaming, VolatilityStreaming, StandardDeviationStreaming,
#   VarianceStreaming, RangeStreaming (TrueRange), HistoricalVolatilityStreaming
from .._backend import _RUST_AVAILABLE
if _RUST_AVAILABLE:
    from ._rust_streaming import (
        # Trend (11)
        SMAStreaming,
        EMAStreaming,
        WMAStreaming,
        MACDStreaming,
        ADXStreaming,
        CCIStreaming,
        DPOStreaming,
        VortexIndicatorStreaming,
        TRIXStreaming,
        AroonStreaming,
        ParabolicSARStreaming,
        # Momentum (11)
        RSIStreaming,
        StochasticStreaming,
        WilliamsRStreaming,
        ROCStreaming,
        PPOStreaming,
        UltimateOscillatorStreaming,
        StochasticRSIStreaming,
        TSIStreaming,
        AwesomeOscillatorStreaming,
        KAMAStreaming,
        MomentumStreaming,
        # Volatility (5)
        ATRStreaming,
        BBandsStreaming,
        KeltnerChannelStreaming,
        DonchianChannelStreaming,
        UlcerIndexStreaming,
        # Volume (10)
        MoneyFlowIndexStreaming,
        AccDistIndexStreaming,
        OnBalanceVolumeStreaming,
        ChaikinMoneyFlowStreaming,
        ForceIndexStreaming,
        EaseOfMovementStreaming,
        VolumePriceTrendStreaming,
        NegativeVolumeIndexStreaming,
        VWAPStreaming,
        VWEMAStreaming,
        # Others (7)
        DailyReturnStreaming,
        DailyLogReturnStreaming,
        CumulativeReturnStreaming,
        RollingReturnStreaming,
        MaxDrawdownStreaming,
        SharpeRatioStreaming,
        CalmarRatioStreaming,
    )
    # Update convenience aliases to point to Rust-backed wrappers
    SMA = SMAStreaming
    EMA = EMAStreaming
    WMA = WMAStreaming
    MACD = MACDStreaming
    ADX = ADXStreaming
    CCI = CCIStreaming
    DPO = DPOStreaming
    VortexIndicator = VortexIndicatorStreaming
    TRIX = TRIXStreaming
    Aroon = AroonStreaming
    ParabolicSAR = ParabolicSARStreaming
    RSI = RSIStreaming
    Stochastic = StochasticStreaming
    WilliamsR = WilliamsRStreaming
    ROC = ROCStreaming
    PPO = PPOStreaming
    UltimateOscillator = UltimateOscillatorStreaming
    StochasticRSI = StochasticRSIStreaming
    TSI = TSIStreaming
    AwesomeOscillator = AwesomeOscillatorStreaming
    KAMA = KAMAStreaming
    Momentum = MomentumStreaming
    ATR = ATRStreaming
    BollingerBands = BBandsStreaming
    BBands = BBandsStreaming
    DonchianChannel = DonchianChannelStreaming
    KeltnerChannel = KeltnerChannelStreaming
    UlcerIndex = UlcerIndexStreaming
    MoneyFlowIndex = MoneyFlowIndexStreaming
    AccDistIndex = AccDistIndexStreaming
    OnBalanceVolume = OnBalanceVolumeStreaming
    ChaikinMoneyFlow = ChaikinMoneyFlowStreaming
    ForceIndex = ForceIndexStreaming
    EaseOfMovement = EaseOfMovementStreaming
    VolumePriceTrend = VolumePriceTrendStreaming
    NegativeVolumeIndex = NegativeVolumeIndexStreaming
    VWAP = VWAPStreaming
    VWEMA = VWEMAStreaming
