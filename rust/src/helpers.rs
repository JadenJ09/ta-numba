/// Shared helper functions for technical indicators

/// Simple Moving Average kernel using running sum for O(n) complexity
pub fn sma_kernel(data: &[f64], window: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![f64::NAN; n];

    if window > n || window == 0 {
        return result;
    }

    // Compute initial sum for first window
    let mut sum: f64 = 0.0;
    for i in 0..window {
        sum += data[i];
    }
    result[window - 1] = sum / window as f64;

    // Rolling sum for subsequent values
    for i in window..n {
        sum = sum + data[i] - data[i - window];
        result[i] = sum / window as f64;
    }

    result
}

/// Exponential Moving Average kernel with optional pandas-style adjustment
pub fn ema_kernel(data: &[f64], alpha: f64, adjusted: bool) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![f64::NAN; n];

    if n == 0 {
        return result;
    }

    if adjusted {
        // Pandas-style adjusted EMA
        let mut weighted_sum = data[0];
        let mut divisor = 1.0;
        let one_minus_alpha = 1.0 - alpha;
        result[0] = data[0];

        for i in 1..n {
            let _weight = one_minus_alpha.powi(i as i32);
            weighted_sum = weighted_sum * one_minus_alpha + data[i];
            divisor = divisor * one_minus_alpha + 1.0;
            result[i] = weighted_sum / divisor;
        }
    } else {
        // Standard unadjusted EMA (Wilder's style when alpha=1/window)
        result[0] = data[0];
        for i in 1..n {
            result[i] = alpha * data[i] + (1.0 - alpha) * result[i - 1];
        }
    }

    result
}

/// Wilder's smoothing - equivalent to EMA with alpha=1/window, unadjusted
pub fn wilders_ema_kernel(data: &[f64], window: usize) -> Vec<f64> {
    let alpha = 1.0 / window as f64;
    ema_kernel(data, alpha, false)
}

/// Calculate True Range for each bar
pub fn true_range(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    let n = high.len();
    let mut tr = vec![f64::NAN; n];

    if n == 0 {
        return tr;
    }

    // First bar: TR = high - low (no previous close)
    tr[0] = high[0] - low[0];

    // Subsequent bars: TR = max(high-low, |high-prev_close|, |low-prev_close|)
    for i in 1..n {
        let hl = high[i] - low[i];
        let hc = (high[i] - close[i - 1]).abs();
        let lc = (low[i] - close[i - 1]).abs();
        tr[i] = hl.max(hc).max(lc);
    }

    tr
}

/// Rolling standard deviation
pub fn rolling_std(data: &[f64], window: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![f64::NAN; n];

    if window > n || window == 0 {
        return result;
    }

    for i in (window - 1)..n {
        let slice = &data[(i + 1 - window)..=i];
        let mean: f64 = slice.iter().sum::<f64>() / window as f64;
        let variance: f64 = slice.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / window as f64;
        result[i] = variance.sqrt();
    }

    result
}

/// Rolling minimum over window
pub fn rolling_min(data: &[f64], window: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![f64::NAN; n];

    if window > n || window == 0 {
        return result;
    }

    for i in (window - 1)..n {
        let slice = &data[(i + 1 - window)..=i];
        result[i] = slice.iter().copied().fold(f64::INFINITY, f64::min);
    }

    result
}

/// Rolling maximum over window
pub fn rolling_max(data: &[f64], window: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![f64::NAN; n];

    if window > n || window == 0 {
        return result;
    }

    for i in (window - 1)..n {
        let slice = &data[(i + 1 - window)..=i];
        result[i] = slice.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    }

    result
}

/// Rolling mean for arbitrary window calculations
pub fn rolling_mean(data: &[f64], window: usize) -> Vec<f64> {
    sma_kernel(data, window)
}

/// Rolling sum over window (O(n) complexity using running sum)
pub fn rolling_sum(data: &[f64], window: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![f64::NAN; n];

    if window > n || window == 0 {
        return result;
    }

    // Compute initial sum for first window (skip NaN values)
    let mut sum: f64 = 0.0;
    for i in 0..window {
        if !data[i].is_nan() {
            sum += data[i];
        }
    }
    result[window - 1] = sum;

    // Rolling sum for subsequent values
    for i in window..n {
        // Remove oldest, add newest (skip NaN values)
        if !data[i - window].is_nan() {
            sum -= data[i - window];
        }
        if !data[i].is_nan() {
            sum += data[i];
        }
        result[i] = sum;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sma_kernel() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = sma_kernel(&data, 3);

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!((result[2] - 2.0).abs() < 1e-10);  // (1+2+3)/3 = 2.0
        assert!((result[3] - 3.0).abs() < 1e-10);  // (2+3+4)/3 = 3.0
        assert!((result[4] - 4.0).abs() < 1e-10);  // (3+4+5)/3 = 4.0
    }

    #[test]
    fn test_ema_kernel_unadjusted() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let alpha = 0.5;
        let result = ema_kernel(&data, alpha, false);

        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - 1.5).abs() < 1e-10);  // 0.5*2 + 0.5*1 = 1.5
        assert!((result[2] - 2.25).abs() < 1e-10); // 0.5*3 + 0.5*1.5 = 2.25
    }

    #[test]
    fn test_true_range() {
        let high = vec![10.0, 12.0, 11.0];
        let low = vec![8.0, 9.0, 10.0];
        let close = vec![9.0, 11.0, 10.5];
        let tr = true_range(&high, &low, &close);

        assert!((tr[0] - 2.0).abs() < 1e-10);  // 10-8 = 2
        assert!((tr[1] - 3.0).abs() < 1e-10);  // max(12-9, |12-9|, |9-9|) = 3
        assert!((tr[2] - 1.0).abs() < 1e-10);  // max(11-10, |11-11|, |10-11|) = 1
    }

    #[test]
    fn test_rolling_std() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = rolling_std(&data, 3);

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        // std([1,2,3]) = sqrt(((1-2)^2 + (2-2)^2 + (3-2)^2)/3) = sqrt(2/3) ≈ 0.8165
        assert!((result[2] - 0.816496580927726).abs() < 1e-10);
    }
}
