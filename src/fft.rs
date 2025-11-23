use std::iter;
use super::{Float, c, pad};

/// Converts a slice of real numbers to complex.
pub fn to_complex<F: Float>(data: &[F]) -> Vec<c<F>> {
    data.iter()
        .map(|f| c::new(*f, F::zero()))
        .collect()
}

/// Computes the fourier transform in-place on a vec of complex numbers.
/// Output length is padded to the next power of 2.
pub fn fft<F: Float>(data: &mut Vec<c<F>>) {
    if data.is_empty() {
        return;
    }

    let original_len = data.len();
    let padded_len = original_len.next_power_of_two();
    pad(data, padded_len);

    cooley_tukey_radix2_dif(data, false);
}
/// Performs FFT and applied natural bit ordering
pub fn fft_ordered<F: Float>(data: &mut Vec<c<F>>) {
    fft(data);
    bit_reverse(data);
}

/// Computes the inverse fourier transform in-place on a vec of complex numbers.
/// Output length is padded to the next power of 2.
pub fn ifft<F: Float>(data: &mut Vec<c<F>>) {
    if data.is_empty() {
        return;
    }

    let original_len = data.len();
    let padded_len = original_len.next_power_of_two();
    pad(data, padded_len);

    cooley_tukey_radix2_dif(data, true);

    let scalar = F::from(padded_len).unwrap().recip();
    for c in data.iter_mut() {
        *c = c.scale(scalar)
    }
}
/// Performs IFFT and applied natural bit ordering
pub fn ifft_ordered<F: Float>(data: &mut Vec<c<F>>) {
    ifft(data);
    bit_reverse(data);
}

/// Roots of unity for twiddle factors, scaled by `k`
/// Returns an iterator over n/2 complex exp(i * angle)
fn roots_of_unity<F: Float>(n: usize, k: F) -> impl Iterator<Item = c<F>> {
    let pi = F::acos(F::one().neg());
    let f_n = F::from(n).unwrap();
    (0..n/2)
        .map(move |x| c::from_polar(F::one(), k * F::from(x).unwrap() * pi / f_n))
}

/// Permutes bit reverse ordering into natural ordering 
/// and natural ordering into bit reverse ordering.
pub fn bit_reverse<F: Float>(data: &mut[c<F>]) {
    #[allow(non_snake_case)]
    let N = data.len();
    // reorder to even and odd indices
    let bits = N.trailing_zeros() as usize;
    for n in 0..N {
        let i = (n as u64).reverse_bits() as usize >> (64 - bits);
        if i > n {
            data.swap(n, i);
        }
    }
}

/// Core radix-2 decimation in freq FFT/IDFT using the Cooley-Tukey algorithm.
/// Is in-place on the given slice, assuming power of 2 length.
/// Panics if length is not power of 2.
#[inline]
fn cooley_tukey_radix2_dif<F: Float>(data: &mut [c<F>], inverse: bool) {
    #[allow(non_snake_case)]
    let N = data.len();
    if N == 1 {
        return;
    }

    assert!(N.is_power_of_two(), "length {N} is not a power of two");

    let one = F::one();
    let two = one+one;
    let coef = inverse.then(|| two).unwrap_or_else(|| two.neg());
    let exp_table = roots_of_unity(N, coef).collect::<Vec<_>>();

    let groups = iter::successors(
        Some(N), 
        |&sz| (sz > 2).then(|| sz / 2)
    );

    let fft_dif = groups.map_while(|sz| {
        if sz < 2 {
            return None;
        }
        let half_sz = sz / 2;
        let table_step = N / sz;

        for i in (0..N).step_by(sz) {
            let mut k = 0;
            for j in i..i + half_sz {
                let add = data[j] + data[j + half_sz];
                let sub = data[j] - data[j + half_sz];
                data[j] = add;
                data[j + half_sz] = sub * exp_table[k];
                k += table_step;
            }
        }
        Some(sz / 2)
    });

    // evaluate the iterator for the side effects
    let _ = fft_dif.count();
}

#[cfg(test)]
mod fft_tests {
    use core::fmt::Debug;
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
    fn test_fft_n1() {
        let mut data: Vec<CType> = vec![CType::new(5.0, 0.0)];
        let expected: Vec<CType> = vec![CType::new(5.0, 0.0)];

        fft(&mut data);
        assert_slices_approx_eq(&data, &expected, 1e-5);
    }

    #[test]
    fn test_ifft_n1() {
        let mut data: Vec<CType> = vec![CType::new(5.0, 0.0)];
        let expected: Vec<CType> = vec![CType::new(5.0, 0.0)];

        ifft(&mut data);
        assert_slices_approx_eq(&data, &expected, 1e-5);
    }

    #[test]
    fn test_fft_n2() {
        let mut data: Vec<CType> = vec![CType::new(1.0, 0.0), CType::new(2.0, 0.0)];
        let expected: Vec<CType> = vec![CType::new(3.0, 0.0), CType::new(-1.0, 0.0)];

        fft(&mut data);
        assert_slices_approx_eq(&data, &expected, 1e-5);
    }

    #[test]
    fn test_ifft_n2() {
        let mut data: Vec<CType> = vec![CType::new(1.0, 0.0), CType::new(2.0, 0.0)];
        let expected: Vec<CType> = vec![CType::new(1.5, 0.0), CType::new(-0.5, 0.0)];

        ifft(&mut data);
        assert_slices_approx_eq(&data, &expected, 1e-5);
    }

    #[test]
    fn test_round_trip_n3() {
        let original: Vec<CType> = vec![CType::new(1.0, 0.0), CType::new(2.0, 0.0), CType::new(3.0, 0.0)];
        let mut data: Vec<CType> = original.clone();
        fft(&mut data);
        bit_reverse(&mut data);
        ifft(&mut data);
        bit_reverse(&mut data);
        data.truncate(original.len());
        assert_slices_approx_eq(&data, &original, 1e-5);
    }

    #[test]
    fn test_fft_n4() {
        let mut data: Vec<CType> = vec![
            CType::new(1.0, 0.0),
            CType::new(2.0, 0.0),
            CType::new(3.0, 0.0),
            CType::new(4.0, 0.0),
        ];
        let expected = data.clone();

        fft_ordered(&mut data);
        ifft_ordered(&mut data);
        assert_slices_approx_eq(&data, &expected, 1e-5);
    }

    #[test]
    fn test_ifft_n4() {
        let mut data: Vec<CType> = vec![
            CType::new(1.0, 0.0),
            CType::new(2.0, 0.0),
            CType::new(3.0, 0.0),
            CType::new(4.0, 0.0),
        ];
        let expected = data.clone();

        ifft(&mut data);
        bit_reverse(&mut data);
        fft(&mut data);
        bit_reverse(&mut data);
        assert_slices_approx_eq(&data, &expected, 1e-5);
    }

    #[test]
    fn test_round_trip_n4() {
        let original: Vec<CType> = vec![
            CType::new(1.0, 0.0),
            CType::new(2.0, 0.0),
            CType::new(3.0, 0.0),
            CType::new(4.0, 0.0),
        ];
        let mut data = original.clone();

        fft_ordered(&mut data);
        ifft_ordered(&mut data);

        assert_slices_approx_eq(&data, &original, 1e-5);
    }

    #[test]
    fn test_fft_empty() {
        let mut data: Vec<CType> = vec![];
        fft(&mut data);
        assert!(data.is_empty());
    }

    #[test]
    fn test_ifft_complex_input() {
        let input: Vec<CType> = vec![CType::new(1.0, 1.0), CType::new(2.0, 2.0)];
        let mut data = input.clone();
        ifft(&mut data);
        let expected = vec![CType::new(1.5, 1.5), CType::new(-0.5, -0.5)];
        assert_slices_approx_eq(&data, &expected, 1e-5);
    }
}
