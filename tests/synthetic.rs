/// Synthetic lightcurve generator for tests.
///
/// Produces multi-band Bazin-model lightcurves with realistic noise,
/// suitable for feeding into `build_mag_bands` / `build_flux_bands`.

/// Bazin model: f(t) = A * exp(-dt/tau_fall) / (1 + exp(-dt/tau_rise)) + B
fn bazin_mag(t: f64, a: f64, b: f64, t0: f64, tau_rise: f64, tau_fall: f64) -> f64 {
    let dt = t - t0;
    let flux = a * (-dt / tau_fall).exp() / (1.0 + (-dt / tau_rise).exp()) + b;
    if flux > 0.0 {
        -2.5 * flux.log10() + 23.9
    } else {
        25.0 // faint fallback
    }
}

/// Simple xorshift64 PRNG for reproducible tests without extra dependencies.
struct Rng64 {
    state: u64,
}

impl Rng64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.max(1),
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Uniform [0, 1)
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    /// Box-Muller normal(0, 1)
    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-15);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Generate a single synthetic Bazin source with g, r, i bands.
///
/// Returns `(times, mags, mag_errs, bands)`.
pub fn generate_bazin_source(
    n_per_band: usize,
    seed: u64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<String>) {
    let mut rng = Rng64::new(seed);

    // Bazin parameters (flux-space)
    let a = 1.0;
    let b = 0.01;
    let t0 = 30.0;
    let tau_rise = 3.0;
    let tau_fall = 25.0;

    // Band color offsets in magnitudes (g brighter, i fainter)
    let bands_info: &[(&str, f64)] = &[("g", -0.5), ("r", 0.0), ("i", 0.3)];

    let mut times = Vec::new();
    let mut mags = Vec::new();
    let mut mag_errs = Vec::new();
    let mut bands = Vec::new();

    let jd_base = 2460000.0;

    for &(band_name, color_offset) in bands_info {
        for j in 0..n_per_band {
            let t = jd_base + (j as f64) * 100.0 / n_per_band as f64;
            let mag = bazin_mag(t - jd_base, a, b, t0, tau_rise, tau_fall) + color_offset;
            let sigma = 0.05 + 0.03 * rng.uniform();
            let noisy_mag = mag + sigma * rng.normal();

            times.push(t);
            mags.push(noisy_mag);
            mag_errs.push(sigma);
            bands.push(band_name.to_string());
        }
    }

    (times, mags, mag_errs, bands)
}

/// Generate N synthetic sources for throughput benchmarking.
pub fn generate_n_sources(
    n_sources: usize,
    n_per_band: usize,
) -> Vec<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<String>)> {
    (0..n_sources)
        .map(|i| generate_bazin_source(n_per_band, 1000 + i as u64))
        .collect()
}
