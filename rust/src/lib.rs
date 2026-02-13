use pyo3::prelude::*;

mod helpers;
mod trend;
mod momentum;
mod volatility;
mod volume;
mod others;
mod streaming;

/// _ta_numba_rs: Rust backend for ta-numba v0.3.0
#[pymodule]
fn _ta_numba_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Trend indicators (bulk)
    m.add_function(wrap_pyfunction!(trend::sma, m)?)?;
    m.add_function(wrap_pyfunction!(trend::ema, m)?)?;
    m.add_function(wrap_pyfunction!(trend::wma, m)?)?;
    m.add_function(wrap_pyfunction!(trend::macd, m)?)?;
    m.add_function(wrap_pyfunction!(trend::adx, m)?)?;
    m.add_function(wrap_pyfunction!(trend::cci, m)?)?;
    m.add_function(wrap_pyfunction!(trend::dpo, m)?)?;
    m.add_function(wrap_pyfunction!(trend::vortex_indicator, m)?)?;
    m.add_function(wrap_pyfunction!(trend::parabolic_sar, m)?)?;
    m.add_function(wrap_pyfunction!(trend::trix, m)?)?;
    m.add_function(wrap_pyfunction!(trend::mass_index, m)?)?;
    m.add_function(wrap_pyfunction!(trend::kst, m)?)?;
    m.add_function(wrap_pyfunction!(trend::ichimoku, m)?)?;
    m.add_function(wrap_pyfunction!(trend::schaff_trend_cycle, m)?)?;
    m.add_function(wrap_pyfunction!(trend::aroon, m)?)?;

    // Momentum indicators (bulk)
    m.add_function(wrap_pyfunction!(momentum::rsi, m)?)?;
    m.add_function(wrap_pyfunction!(momentum::stochastic, m)?)?;
    m.add_function(wrap_pyfunction!(momentum::williams_r, m)?)?;
    m.add_function(wrap_pyfunction!(momentum::ppo, m)?)?;
    m.add_function(wrap_pyfunction!(momentum::ultimate_oscillator, m)?)?;
    m.add_function(wrap_pyfunction!(momentum::stochastic_rsi, m)?)?;
    m.add_function(wrap_pyfunction!(momentum::tsi, m)?)?;
    m.add_function(wrap_pyfunction!(momentum::awesome_oscillator, m)?)?;
    m.add_function(wrap_pyfunction!(momentum::kama, m)?)?;
    m.add_function(wrap_pyfunction!(momentum::roc, m)?)?;
    m.add_function(wrap_pyfunction!(momentum::pvo, m)?)?;
    m.add_function(wrap_pyfunction!(momentum::momentum, m)?)?;

    // Volatility indicators (bulk)
    m.add_function(wrap_pyfunction!(volatility::atr, m)?)?;
    m.add_function(wrap_pyfunction!(volatility::bollinger_bands, m)?)?;
    m.add_function(wrap_pyfunction!(volatility::keltner_channel, m)?)?;
    m.add_function(wrap_pyfunction!(volatility::donchian_channel, m)?)?;
    m.add_function(wrap_pyfunction!(volatility::ulcer_index, m)?)?;

    // Volume indicators (bulk)
    m.add_function(wrap_pyfunction!(volume::mfi, m)?)?;
    m.add_function(wrap_pyfunction!(volume::acc_dist_index, m)?)?;
    m.add_function(wrap_pyfunction!(volume::obv, m)?)?;
    m.add_function(wrap_pyfunction!(volume::chaikin_money_flow, m)?)?;
    m.add_function(wrap_pyfunction!(volume::force_index, m)?)?;
    m.add_function(wrap_pyfunction!(volume::eom, m)?)?;
    m.add_function(wrap_pyfunction!(volume::vpt, m)?)?;
    m.add_function(wrap_pyfunction!(volume::nvi, m)?)?;
    m.add_function(wrap_pyfunction!(volume::vwap, m)?)?;
    m.add_function(wrap_pyfunction!(volume::vwema, m)?)?;

    // Other indicators (bulk)
    m.add_function(wrap_pyfunction!(others::daily_return, m)?)?;
    m.add_function(wrap_pyfunction!(others::daily_log_return, m)?)?;
    m.add_function(wrap_pyfunction!(others::cumulative_return, m)?)?;

    // Streaming classes - Trend (11)
    m.add_class::<streaming::SMAStreaming>()?;
    m.add_class::<streaming::EMAStreaming>()?;
    m.add_class::<streaming::WMAStreaming>()?;
    m.add_class::<streaming::MACDStreaming>()?;
    m.add_class::<streaming::ADXStreaming>()?;
    m.add_class::<streaming::CCIStreaming>()?;
    m.add_class::<streaming::DPOStreaming>()?;
    m.add_class::<streaming::VortexStreaming>()?;
    m.add_class::<streaming::TRIXStreaming>()?;
    m.add_class::<streaming::AroonStreaming>()?;
    m.add_class::<streaming::PSARStreaming>()?;

    // Streaming classes - Momentum (12)
    m.add_class::<streaming::RSIStreaming>()?;
    m.add_class::<streaming::StochasticStreaming>()?;
    m.add_class::<streaming::WilliamsRStreaming>()?;
    m.add_class::<streaming::ROCStreaming>()?;
    m.add_class::<streaming::PPOStreaming>()?;
    m.add_class::<streaming::PVOStreaming>()?;
    m.add_class::<streaming::UltimateOscillatorStreaming>()?;
    m.add_class::<streaming::StochasticRSIStreaming>()?;
    m.add_class::<streaming::TSIStreaming>()?;
    m.add_class::<streaming::AwesomeOscillatorStreaming>()?;
    m.add_class::<streaming::KAMAStreaming>()?;
    m.add_class::<streaming::MomentumStreaming>()?;

    // Streaming classes - Volatility (5)
    m.add_class::<streaming::ATRStreaming>()?;
    m.add_class::<streaming::BollingerBandsStreaming>()?;
    m.add_class::<streaming::KeltnerChannelStreaming>()?;
    m.add_class::<streaming::DonchianChannelStreaming>()?;
    m.add_class::<streaming::UlcerIndexStreaming>()?;

    // Streaming classes - Volume (10)
    m.add_class::<streaming::MFIStreaming>()?;
    m.add_class::<streaming::AccDistStreaming>()?;
    m.add_class::<streaming::OBVStreaming>()?;
    m.add_class::<streaming::CMFStreaming>()?;
    m.add_class::<streaming::ForceIndexStreaming>()?;
    m.add_class::<streaming::EOMStreaming>()?;
    m.add_class::<streaming::VPTStreaming>()?;
    m.add_class::<streaming::NVIStreaming>()?;
    m.add_class::<streaming::VWAPStreaming>()?;
    m.add_class::<streaming::VWEMAStreaming>()?;

    // Streaming classes - Others (7)
    m.add_class::<streaming::DailyReturnStreaming>()?;
    m.add_class::<streaming::DailyLogReturnStreaming>()?;
    m.add_class::<streaming::CumulativeReturnStreaming>()?;
    m.add_class::<streaming::RollingReturnStreaming>()?;
    m.add_class::<streaming::MaxDrawdownStreaming>()?;
    m.add_class::<streaming::SharpeRatioStreaming>()?;
    m.add_class::<streaming::CalmarRatioStreaming>()?;

    Ok(())
}
