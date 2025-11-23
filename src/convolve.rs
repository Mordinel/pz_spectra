use num_traits::Float;
use super::{c, pad, fft_ordered, ifft_ordered};

/// Convolution modes matching scipy's signal.convolve
#[derive(Clone, Copy, PartialEq)]
pub enum ConvolveMode {
    /// Full convolution output
    Full,
    /// Valid (no edge effects)
    Valid,
    /// Same-length output as input signal (centered)
    Same,
}

/// Computes a linear convolution in the frequency domain in-place on `signal`.
/// Pads both `signal` and `kernel` to power of 2 length for FFT, multiplies spectra, IFFT, 
/// and then extracts per mode.
pub fn convolve<F>(
    mode: ConvolveMode,
    signal: &mut Vec<c<F>>,
    kernel: &Vec<c<F>>,
) where
    F: Float,
{
    if signal.is_empty() || kernel.is_empty() {
        return;
    }
    let mut kernel_clone = kernel.clone();
    let len_a = signal.len();
    let len_b = kernel_clone.len();
    let full_len = len_a + len_b.saturating_sub(1);
    let fft_len = full_len.next_power_of_two();
    signal.reserve(fft_len.saturating_sub(len_a));

    pad(signal, fft_len);
    pad(&mut kernel_clone, fft_len);

    fft_ordered(signal);
    fft_ordered(&mut kernel_clone);

    for (a, b) in signal.iter_mut().zip(kernel_clone.iter()) {
        *a = *a * *b;
    }

    ifft_ordered(signal);

    let out_len = match mode {
        ConvolveMode::Full => full_len,
        ConvolveMode::Valid => {
            let min_len = len_a.min(len_b);
            let max_len = len_a.max(len_b);
            max_len.saturating_sub(min_len).saturating_add(1)
        },
        ConvolveMode::Same => len_a,
    };
    if out_len == 0 {
        return;
    }

    match mode {
        ConvolveMode::Full => (),
        ConvolveMode::Valid => {
            let start = len_a.min(len_b).saturating_sub(1);
            signal.drain(0..start.min(full_len));
        },
        ConvolveMode::Same => {
            let start = (len_b - 1) / 2;
            if start > 0 {
                signal.drain(0..start);
            }
            pad(signal, len_a);
        },
    }

    signal.truncate(out_len);
}

/// Computes a deconvolution in the frequency domain in-place on `signal`.
/// Pads both `signal` and `divisor` to power of 2 length for FFT, divides spectra, IFFT,
/// and then extracts per mode.
/// Applies an optional weiner filter in the fourier domain with a configurable threshold set by `damping_threshold`.
pub fn deconvolve<F>(
    mode: ConvolveMode,
    signal: &mut Vec<c<F>>,
    divisor: &Vec<c<F>>,
    damping_threshold: Option<F>,
) where
    F: Float
{
    if signal.is_empty() || divisor.is_empty() {
        return;
    }

    let len_signal = signal.len();
    let len_divisor = divisor.len();
    let full_len = len_signal + len_divisor.saturating_sub(1);
    let fft_len = full_len.next_power_of_two();
    signal.reserve(fft_len.saturating_sub(len_signal));

    let mut divisor_clone = divisor.clone();
    pad(signal, fft_len);
    pad(&mut divisor_clone, fft_len);

    fft_ordered(signal);
    fft_ordered(&mut divisor_clone);

    let damping_threshold = damping_threshold.unwrap_or(F::zero());
    let zero = c::new(F::zero(), F::zero());
    for (s, d) in signal.iter_mut().zip(divisor_clone.iter()) {
        let denom = c::new(d.norm_sqr() + damping_threshold, F::zero());
        if denom == zero {
            *s = zero;
        } else {
            *s = *s * d.conj() / denom; // simple wiener filter
        }
    }

    ifft_ordered(signal);

    let out_len = match mode {
        ConvolveMode::Full => full_len,
        ConvolveMode::Valid => {
            let min_len = len_signal.min(len_divisor);
            let max_len = len_signal.max(len_divisor);
            max_len.saturating_sub(min_len).saturating_add(1)
        },
        ConvolveMode::Same => len_signal,
    };

    if out_len == 0 {
        return;
    }

    match mode {
        ConvolveMode::Full => (),
        ConvolveMode::Valid => {
            let start = len_signal.min(len_divisor).saturating_sub(1);
            signal.drain(0..start.min(full_len));
        },
        ConvolveMode::Same => {
            let start = (len_divisor - 1) / 2;
            if start > 0 {
                signal.drain(0..start);
            }
            pad(signal, len_signal);
        },
    }

    signal.truncate(out_len);
}

/// Computes a linear cross-correlation in the frequency domain in-place on `signal_a`.
/// Pads both `signal_a` and `signal_b` to power of 2 length for FFT, 
/// multiplies reversed and conjugate spectra, IFFT and then extracts per mode.
pub fn correlate<F>(
    mode: ConvolveMode,
    signal_a: &mut Vec<c<F>>,
    signal_b: &Vec<c<F>>,
) where
    F: Float,
{
    if signal_a.is_empty() || signal_b.is_empty() {
        return;
    }
    let fb = signal_b.iter().map(|z| z.conj()).rev().collect();
    convolve(mode, signal_a, &fb);
}

#[cfg(test)]
mod convolution_tests {
    use core::fmt::Debug;
    use crate::to_complex;

    use super::*;

    type CType = c<f64>;

    fn approx_eq<F: Float>(a: &c<F>, b: &c<F>, eps: F) -> bool {
        (a - b).norm() < eps
    }

    fn assert_slices_approx_eq<F: Float + Debug>(actual: &[c<F>], expected: &[c<F>], eps: F) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!(approx_eq(a, e, eps), "actual: {:?}, expected: {:?}", a, e);
        }
    }

    #[test]
    fn test_convolve_full() {
        let mut a: Vec<CType> = vec![CType::new(1.0, 0.0), CType::new(2.0, 0.0), CType::new(3.0, 0.0)];
        let mut b: Vec<CType> = vec![CType::new(1.0, 0.0), CType::new(0.5, 0.0)];
        let expected: Vec<CType> = vec![CType::new(1.0, 0.0), CType::new(2.5, 0.0), CType::new(4.0, 0.0), CType::new(1.5, 0.0)];
        convolve(ConvolveMode::Full, &mut a, &mut b);
        assert_slices_approx_eq(&a, &expected, 1e-5);
    }

    #[test]
    fn test_convolve_valid() {
        let mut a: Vec<CType> = vec![CType::new(1.0, 0.0), CType::new(2.0, 0.0), CType::new(3.0, 0.0)];
        let mut b: Vec<CType> = vec![CType::new(1.0, 0.0), CType::new(0.5, 0.0)];
        let expected = vec![CType::new(2.5, 0.0), CType::new(4.0, 0.0)];
        convolve(ConvolveMode::Valid, &mut a, &mut b);
        assert_slices_approx_eq(&a, &expected, 1e-5);
    }

    #[test]
    fn test_convolve_same() {
        let mut a: Vec<CType> = vec![CType::new(1.0, 0.0), CType::new(2.0, 0.0), CType::new(3.0, 0.0)];
        let mut b: Vec<CType> = vec![CType::new(1.0, 0.0), CType::new(0.5, 0.0)];
        let expected: Vec<CType> = vec![CType::new(1.0, 0.0), CType::new(2.5, 0.0), CType::new(4.0, 0.0)];
        convolve(ConvolveMode::Same, &mut a, &mut b);
        assert_slices_approx_eq(&a, &expected, 1e-5);
    }

    #[test]
    fn test_convolve_empty_a() {
        let mut a: Vec<CType> = vec![];
        let b: Vec<CType> = vec![CType::new(1.0, 0.0)];
        convolve(ConvolveMode::Full, &mut a, &b);
        assert!(a.is_empty());
    }

    #[test]
    fn test_convolve_large_diff() {
        let mut a: Vec<CType> = vec![CType::new(1.0, 0.0); 100];
        let b: Vec<CType> = vec![CType::new(1.0, 0.0), CType::new(0.5, 0.0)];
        convolve(ConvolveMode::Valid, &mut a, &b);
        assert_eq!(a.len(), 99);
        assert!((a[0].re - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_correlate_full() {
        let mut a: Vec<CType> = vec![CType::new(1.0, 0.0), CType::new(2.0, 0.0), CType::new(3.0, 0.0)];
        let b: Vec<CType> = vec![CType::new(1.0, 0.0), CType::new(0.5, 0.0)];
        let expected: Vec<CType> = vec![CType::new(0.5, 0.0), CType::new(2.0, 0.0), CType::new(3.5, 0.0), CType::new(3.0, 0.0)];
        correlate(ConvolveMode::Full, &mut a, &b);
        assert_slices_approx_eq(&a, &expected, 1e-5);
    }

    #[test]
    fn test_correlate_valid() {
        let mut a: Vec<CType> = vec![CType::new(1.0, 0.0), CType::new(2.0, 0.0), CType::new(3.0, 0.0)];
        let b: Vec<CType> = vec![CType::new(1.0, 0.0), CType::new(0.5, 0.0)];
        let expected: Vec<CType> = vec![CType::new(2.0, 0.0), CType::new(3.5, 0.0)];
        correlate(ConvolveMode::Valid, &mut a, &b);
        assert_slices_approx_eq(&a, &expected, 1e-5);
    }

    #[test]
    fn test_correlate_same() {
        let mut a: Vec<CType> = vec![CType::new(1.0, 0.0), CType::new(2.0, 0.0), CType::new(3.0, 0.0)];
        let b: Vec<CType> = vec![CType::new(1.0, 0.0), CType::new(0.5, 0.0)];
        let expected: Vec<CType> = vec![CType::new(0.5, 0.0), CType::new(2.0, 0.0), CType::new(3.5, 0.0)];
        correlate(ConvolveMode::Same, &mut a, &b);
        assert_slices_approx_eq(&a, &expected, 1e-5);
    }

    #[test]
    fn test_correlate_complex_full() {
        let mut a: Vec<CType> = vec![CType::new(1.0, 1.0), CType::new(2.0, 2.0)];
        let b: Vec<CType> = vec![CType::new(1.0, 1.0), CType::new(0.5, 0.5)];
        let expected: Vec<CType> = vec![CType::new(1.0, 0.0), CType::new(4.0, 0.0), CType::new(4.0, 0.0)];
        correlate(ConvolveMode::Full, &mut a, &b);
        assert_slices_approx_eq(&a, &expected, 1e-5);
    }

    #[test]
    fn test_correlate_complex_valid() {
        let mut a = vec![CType::new(1.0, 1.0), CType::new(2.0, 2.0)];
        let b = vec![CType::new(1.0, 1.0), CType::new(0.5, 0.5)];
        let expected = vec![CType::new(4.0, 0.0)];
        correlate(ConvolveMode::Valid, &mut a, &b);
        assert_slices_approx_eq(&a, &expected, 1e-5);
    }

    #[test]
    fn test_correlate_empty_a() {
        let mut a: Vec<CType> = vec![];
        let b = vec![CType::new(1.0, 0.0)];
        correlate(ConvolveMode::Full, &mut a, &b);
        assert!(a.is_empty());
    }

    #[test]
    fn test_deconvolve_full() {
        let kernel: Vec<CType> = to_complex(&[1.0, 0.5]);
        let original: Vec<CType> = to_complex(&[1.0, 2.0, 3.0]);
        let mut signal = original.clone();
        convolve(ConvolveMode::Full, &mut signal, &kernel);
        deconvolve(ConvolveMode::Full, &mut signal, &kernel, None);
        signal.truncate(original.len());
        assert_slices_approx_eq(&signal, &original, 1e-5);
    }
}
