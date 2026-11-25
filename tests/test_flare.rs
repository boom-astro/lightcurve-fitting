use lightcurve_fitting::{
    davenport_area, davenport_template, fit_flares, fit_flares_batch, flare_flux, BandData,
    FlareFitOptions,
};

const MINUTE: f64 = 1.0 / 1440.0;

fn xorshift(state: &mut u64) -> f64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    (x >> 11) as f64 / ((1u64 << 53) as f64)
}

fn normal(rng: &mut u64) -> f64 {
    let u1 = xorshift(rng).max(1e-15);
    let u2 = xorshift(rng);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

struct Flare {
    t_peak: f64,
    t_half: f64,
    amp: f64,
}

/// Template averaged over each exposure by brute-force sub-sampling, so the
/// test does not reuse the closed-form integral under test.
fn simulate(
    flares: &[Flare],
    span: f64,
    cadence: f64,
    exposure: f64,
    noise: f64,
    baseline: (f64, f64),
    seed: u64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // Scramble so small seeds don't start xorshift on near-zero draws.
    let mut rng = (seed ^ 0x9E37_79B9_7F4A_7C15).wrapping_mul(0xBF58_476D_1CE4_E5B9) | 1;
    let n = (span / cadence) as usize;
    let sub = 400;
    let mut t = Vec::with_capacity(n);
    let mut f = Vec::with_capacity(n);
    for i in 0..n {
        let ti = i as f64 * cadence;
        let mut v = 0.0;
        for fl in flares {
            if exposure > 0.0 {
                let mut s = 0.0;
                for k in 0..sub {
                    let tk = ti - 0.5 * exposure + exposure * (k as f64 + 0.5) / sub as f64;
                    s += davenport_template((tk - fl.t_peak) / fl.t_half);
                }
                v += fl.amp * s / sub as f64;
            } else {
                v += fl.amp * davenport_template((ti - fl.t_peak) / fl.t_half);
            }
        }
        t.push(ti);
        f.push(v + baseline.0 + baseline.1 * ti + noise * normal(&mut rng));
    }
    let e = vec![noise; n];
    (t, f, e)
}

#[test]
fn template_shape_and_area() {
    assert!((davenport_template(0.0) - 1.0).abs() < 1e-12);
    assert!(davenport_template(-1.0).abs() < 0.01);
    assert_eq!(davenport_template(-1.5), 0.0);
    assert!((davenport_template(1e-9) - 1.0).abs() < 0.01);
    assert!(davenport_template(-0.5) > 0.0 && davenport_template(-0.5) < 1.0);

    // Trapezoid integral of the point template on a fine grid.
    let (lo, hi, n) = (-1.0, 80.0, 2_000_000);
    let h = (hi - lo) / n as f64;
    let mut s = 0.5 * (davenport_template(lo) + davenport_template(hi));
    for i in 1..n {
        s += davenport_template(lo + i as f64 * h);
    }
    s *= h;
    assert!((davenport_area() - s).abs() < 1e-4, "area {} vs {}", davenport_area(), s);
    assert!((davenport_area() - 1.827).abs() < 1e-3);
}

#[test]
fn exposure_average_matches_subsampling() {
    let exposure = 0.7;
    for &x in &[-2.0, -1.2, -0.8, -0.3, 0.0, 0.2, 1.0, 5.0, 30.0] {
        let analytic = flare_flux(x, 0.0, 1.0, 1.0, exposure);
        let sub = 20_000;
        let brute: f64 = (0..sub)
            .map(|k| davenport_template(x - 0.5 * exposure + exposure * (k as f64 + 0.5) / sub as f64))
            .sum::<f64>()
            / sub as f64;
        // The midpoint sum carries O(step) error at the small jump at x = -1.
        assert!((analytic - brute).abs() < 1e-6, "x={x}: {analytic} vs {brute}");
    }
    assert_eq!(flare_flux(0.3, 0.0, 1.0, 2.0, 0.0), 2.0 * davenport_template(0.3));
}

#[test]
fn recovers_single_flare_at_short_cadence() {
    let truth = Flare { t_peak: 0.06, t_half: 5.0 * MINUTE, amp: 0.02 };
    let exposure = MINUTE;
    let (t, f, e) = simulate(&[truth], 0.2, MINUTE, exposure, 1e-3, (0.0, 0.0), 7);
    let opts = FlareFitOptions { exposure_time: Some(exposure), ..Default::default() };
    let res = fit_flares(&t, &f, &e, &opts).unwrap();

    assert_eq!(res.flares.len(), 1, "per-K BIC {:?}", res.per_k_bic);
    let fl = &res.flares[0];
    let truth = Flare { t_peak: 0.06, t_half: 5.0 * MINUTE, amp: 0.02 };
    let pull = |v: f64, tv: f64, s: Option<f64>| (v - tv) / s.unwrap();
    assert!(pull(fl.t_peak, truth.t_peak, fl.t_peak_err).abs() < 4.0);
    assert!(pull(fl.t_half, truth.t_half, fl.t_half_err).abs() < 4.0);
    assert!(pull(fl.amplitude, truth.amp, fl.amplitude_err).abs() < 4.0);
    let ed_true = truth.amp * truth.t_half * davenport_area();
    assert!(pull(fl.equivalent_duration, ed_true, fl.equivalent_duration_err).abs() < 4.0);
    assert!(res.reduced_chi2 > 0.7 && res.reduced_chi2 < 1.4);
}

#[test]
fn long_exposure_recovers_unresolved_flare() {
    // 30-min exposures on a flare with a 10-min FWHM: only a couple of bins
    // are elevated, and a point-sampled template cannot match their shape.
    let exposure = 30.0 * MINUTE;
    let (tp, th, amp) = (0.513, 10.0 * MINUTE, 0.2);
    let truth = [Flare { t_peak: tp, t_half: th, amp }];
    let (t, f, e) = simulate(&truth, 2.0, exposure, exposure, 1e-3, (0.0, 0.0), 11);
    let ed_true = amp * th * davenport_area();

    let with = fit_flares(&t, &f, &e, &FlareFitOptions { exposure_time: Some(exposure), ..Default::default() }).unwrap();
    let without = fit_flares(&t, &f, &e, &FlareFitOptions::default()).unwrap();

    assert_eq!(with.flares.len(), 1);
    let fw = &with.flares[0];
    let pull = |v: f64, tv: f64, s: Option<f64>| (v - tv) / s.unwrap();
    assert!(pull(fw.t_peak, tp, fw.t_peak_err).abs() < 4.0);
    assert!(pull(fw.t_half, th, fw.t_half_err).abs() < 4.0);
    assert!(pull(fw.amplitude, amp, fw.amplitude_err).abs() < 4.0);
    assert!(pull(fw.equivalent_duration, ed_true, fw.equivalent_duration_err).abs() < 4.0);
    assert!(with.reduced_chi2 < 1.3, "reduced chi2 {}", with.reduced_chi2);
    assert!(without.chi2 - with.chi2 > 10.0, "chi2 {} vs {}", with.chi2, without.chi2);
}

#[test]
fn separates_overlapping_flares() {
    let flares = [
        Flare { t_peak: 0.050, t_half: 4.0 * MINUTE, amp: 0.03 },
        Flare { t_peak: 0.058, t_half: 6.0 * MINUTE, amp: 0.015 },
    ];
    let (t, f, e) = simulate(&flares, 0.2, MINUTE, MINUTE, 1e-3, (0.0, 0.0), 3);
    let opts = FlareFitOptions { exposure_time: Some(MINUTE), ..Default::default() };
    let res = fit_flares(&t, &f, &e, &opts).unwrap();

    assert_eq!(res.flares.len(), 2, "per-K BIC {:?}", res.per_k_bic);
    for (fit, truth) in res.flares.iter().zip(&flares) {
        assert!((fit.t_peak - truth.t_peak).abs() < 1.5 * MINUTE, "t_peak {} vs {}", fit.t_peak, truth.t_peak);
        assert!((fit.amplitude / truth.amp - 1.0).abs() < 0.3, "amp {} vs {}", fit.amplitude, truth.amp);
    }
}

#[test]
fn rejects_pure_noise() {
    let (t, f, e) = simulate(&[], 0.2, MINUTE, 0.0, 1e-3, (0.01, 0.0), 5);
    let res = fit_flares(&t, &f, &e, &FlareFitOptions::default()).unwrap();
    assert!(res.flares.is_empty(), "spurious flares {:?}", res.flares);
    assert!((res.baseline[0] - 0.01).abs() < 5e-4);
}

#[test]
fn linear_baseline_absorbs_trend() {
    let truth = [Flare { t_peak: 0.1, t_half: 3.0 * MINUTE, amp: 0.01 }];
    let (t, f, e) = simulate(&truth, 0.25, MINUTE, 0.0, 1e-3, (0.002, 0.05), 9);
    let opts = FlareFitOptions { baseline_order: 1, ..Default::default() };
    let res = fit_flares(&t, &f, &e, &opts).unwrap();
    assert_eq!(res.flares.len(), 1);
    assert!((res.baseline[1] - 0.05).abs() < 0.01, "slope {}", res.baseline[1]);
    assert!((res.flares[0].amplitude / 0.01 - 1.0).abs() < 0.3);
}

#[test]
fn error_bars_are_calibrated() {
    let opts = FlareFitOptions { exposure_time: Some(MINUTE), max_flares: 1, ..Default::default() };
    let mut pulls_tp = Vec::new();
    let mut pulls_ed = Vec::new();
    for seed in 1..=60u64 {
        let truth = [Flare { t_peak: 0.06, t_half: 4.0 * MINUTE, amp: 0.01 }];
        let (t, f, e) = simulate(&truth, 0.15, MINUTE, MINUTE, 1e-3, (0.0, 0.0), seed * 7919);
        let res = fit_flares(&t, &f, &e, &opts).unwrap();
        let fl = &res.flares[0];
        let ed_true = 0.01 * 4.0 * MINUTE * davenport_area();
        pulls_tp.push((fl.t_peak - 0.06) / fl.t_peak_err.unwrap());
        pulls_ed.push((fl.equivalent_duration - ed_true) / fl.equivalent_duration_err.unwrap());
    }
    let rms = |v: &[f64]| (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt();
    assert!((0.6..1.6).contains(&rms(&pulls_tp)), "t_peak pull rms {}", rms(&pulls_tp));
    assert!((0.6..1.6).contains(&rms(&pulls_ed)), "ED pull rms {}", rms(&pulls_ed));
}

#[test]
fn batch_matches_single() {
    let opts = FlareFitOptions::default();
    let windows: Vec<BandData> = (0..4u64)
        .map(|s| {
            let truth = [Flare { t_peak: 0.05 + 0.01 * s as f64, t_half: 5.0 * MINUTE, amp: 0.02 }];
            let (times, values, errors) = simulate(&truth, 0.2, MINUTE, 0.0, 1e-3, (0.0, 0.0), s + 100);
            BandData { times, values, errors }
        })
        .collect();
    let batch = fit_flares_batch(&windows, &opts);
    for (w, b) in windows.iter().zip(&batch) {
        let single = fit_flares(&w.times, &w.values, &w.errors, &opts).unwrap();
        let b = b.as_ref().unwrap();
        assert_eq!(single.flares.len(), b.flares.len());
        assert_eq!(single.flares[0].t_peak, b.flares[0].t_peak);
    }
}

#[cfg(any(feature = "cuda", feature = "metal"))]
mod gpu {
    use super::*;
    use lightcurve_fitting::fit_flares_batch_gpu;
    use lightcurve_fitting::gpu::GpuContext;

    fn window(flares: &[Flare], span: f64, cadence: f64, exposure: f64, baseline: (f64, f64), seed: u64) -> BandData {
        let (times, values, errors) = simulate(flares, span, cadence, exposure, 1e-3, baseline, seed);
        BandData { times, values, errors }
    }

    /// GPU and CPU searches must select the same components and land on the
    /// same optimum to a small fraction of the parameter uncertainties.
    fn assert_matches_cpu(windows: &[BandData], opts: &FlareFitOptions) {
        let gpu = GpuContext::new(0).expect("GPU context");
        let g_all = fit_flares_batch_gpu(&gpu, windows, opts).unwrap();
        let c_all = fit_flares_batch(windows, opts);
        for (i, (g, c)) in g_all.iter().zip(&c_all).enumerate() {
            let (g, c) = (g.as_ref().unwrap(), c.as_ref().unwrap());
            assert_eq!(g.flares.len(), c.flares.len(), "window {i}: GPU {:?}\nCPU {:?}", g.flares, c.flares);
            assert!(g.chi2 <= c.chi2 + 0.5, "window {i}: chi2 GPU {} CPU {}", g.chi2, c.chi2);
            for (gf, cf) in g.flares.iter().zip(&c.flares) {
                // A component much narrower than its exposure has a singular
                // covariance; only its ED is constrained.
                if cf.t_peak_err.is_none() || gf.t_peak_err.is_none() {
                    let rel = (gf.equivalent_duration / cf.equivalent_duration - 1.0).abs();
                    assert!(rel < 0.05, "window {i}: {gf:?} vs {cf:?}");
                    continue;
                }
                let tol = |s: Option<f64>| 0.25 * s.unwrap();
                assert!((gf.t_peak - cf.t_peak).abs() < tol(cf.t_peak_err), "window {i}: {gf:?} vs {cf:?}");
                assert!((gf.t_half - cf.t_half).abs() < tol(cf.t_half_err), "window {i}: {gf:?} vs {cf:?}");
                assert!(
                    (gf.equivalent_duration - cf.equivalent_duration).abs() < tol(cf.equivalent_duration_err),
                    "window {i}: {gf:?} vs {cf:?}"
                );
            }
        }
    }

    #[test]
    fn gpu_matches_cpu_at_short_cadence() {
        let mut windows = vec![
            window(&[Flare { t_peak: 0.06, t_half: 5.0 * MINUTE, amp: 0.02 }], 0.2, MINUTE, MINUTE, (0.0, 0.0), 7),
            window(
                &[
                    Flare { t_peak: 0.050, t_half: 4.0 * MINUTE, amp: 0.03 },
                    Flare { t_peak: 0.058, t_half: 6.0 * MINUTE, amp: 0.015 },
                ],
                0.2, MINUTE, MINUTE, (0.0, 0.0), 3,
            ),
            window(&[], 0.2, MINUTE, MINUTE, (0.01, 0.0), 5),
        ];
        // Randomised single flares over a range of widths, amplitudes and window lengths.
        let mut rng = 99u64;
        for s in 0..24u64 {
            let span = 0.1 + 0.3 * xorshift(&mut rng);
            let fl = Flare {
                t_peak: span * (0.2 + 0.5 * xorshift(&mut rng)),
                t_half: MINUTE * (1.0 + 15.0 * xorshift(&mut rng)),
                amp: 0.004 + 0.05 * xorshift(&mut rng),
            };
            windows.push(window(&[fl], span, MINUTE, MINUTE, (0.0, 0.0), 1000 + s));
        }
        let opts = FlareFitOptions { exposure_time: Some(MINUTE), ..Default::default() };
        assert_matches_cpu(&windows, &opts);
    }

    #[test]
    fn gpu_matches_cpu_with_long_exposures() {
        let exposure = 30.0 * MINUTE;
        let windows: Vec<BandData> = (0..6u64)
            .map(|s| {
                let fl = Flare { t_peak: 0.5 + 0.013 * s as f64, t_half: (5.0 + 5.0 * s as f64) * MINUTE, amp: 0.2 };
                window(&[fl], 2.0, exposure, exposure, (0.0, 0.0), 11 + s)
            })
            .collect();
        let opts = FlareFitOptions { exposure_time: Some(exposure), ..Default::default() };
        assert_matches_cpu(&windows, &opts);
    }

    #[test]
    fn gpu_matches_cpu_with_linear_baseline() {
        let windows: Vec<BandData> = (0..4u64)
            .map(|s| {
                let fl = Flare { t_peak: 0.1, t_half: 3.0 * MINUTE, amp: 0.01 };
                window(&[fl], 0.25, MINUTE, 0.0, (0.002, 0.05), 9 + s)
            })
            .collect();
        let opts = FlareFitOptions { baseline_order: 1, ..Default::default() };
        assert_matches_cpu(&windows, &opts);
    }

    #[test]
    fn gpu_rejects_unsupported_options() {
        let gpu = GpuContext::new(0).expect("GPU context");
        let w = [window(&[], 0.1, MINUTE, 0.0, (0.0, 0.0), 1)];
        let too_many = FlareFitOptions { max_flares: 7, ..Default::default() };
        assert!(fit_flares_batch_gpu(&gpu, &w, &too_many).is_err());
        let quadratic = FlareFitOptions { baseline_order: 2, ..Default::default() };
        assert!(fit_flares_batch_gpu(&gpu, &w, &quadratic).is_err());
    }
}
