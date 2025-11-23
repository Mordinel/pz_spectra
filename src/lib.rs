#[deny(unsafe_code)]

use num_traits::Float;
use num_complex::Complex;

#[allow(non_camel_case_types)]
pub type c<F> = Complex<F>;

#[allow(dead_code)]
#[allow(non_camel_case_types)]
pub type c32 = Complex<f32>;

#[allow(dead_code)]
#[allow(non_camel_case_types)]
pub type c64 = Complex<f64>;

/// Pads a vec of complex values with zeroes to the length `to_len`
#[inline]
fn pad<F: Float>(data: &mut Vec<c<F>>, to_len: usize) {
    data.resize(to_len, c::new(F::zero(), F::zero()));
}

mod fft;
pub use fft::{fft, fft_ordered, ifft_ordered, to_complex, bit_reverse};

#[cfg(feature="convolve")]
mod convolve;
#[cfg(feature="convolve")]
pub use convolve::{ConvolveMode, convolve, deconvolve, correlate};

