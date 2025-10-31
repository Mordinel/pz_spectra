# pz\_spectra

ezpz fft, optional convolution, deconvolution and cross-correlation.

## Usage

In your `Cargo.toml`:
```toml
[dependencies]
pz_spectra = "0.0.1"
# or with convolve:
# pz_spectra = { version = "0.0.1", features = ["convolve"] }
```

In your code:
```rust
use pz_spectra::{fft, ifft, to_complex}
// or with convolve
// use pz_spectra::{fft, ifft, convolve, to_complex, ConvolveMode};

let mut signal = to_complex(&[1.0, 2.0, 3.0]);

// in-place
fft(&mut signal);
ifft(&mut signal);

// let kernel = to_complex(&[1.0, 0.5]);
// convolve(ConvolveMode::Full, &mut signal, &kernel);
```

## Todo

Memoize twiddle factors.

