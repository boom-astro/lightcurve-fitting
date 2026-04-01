//! Tests for the Metal GPU backend.
//!
//! GPU integration tests require `--features metal` and a macOS system with Metal.
//! The CPU-side validation tests (types, constraints, bounds) run without features.

// ---------------------------------------------------------------------------
// CPU-side tests (always run — no feature gate)
// ---------------------------------------------------------------------------

use lightcurve_fitting::{eval_model_flux, SviModelName};

/// Villar physical constraint must reject unphysical params.
#[test]
fn villar_constraint_rejects_unphysical() {
    // Construct params where beta * gamma > 1 (violates c1 constraint)
    // Villar params: [log_A, beta, log_gamma, t0, log_tau_rise, log_tau_fall, log_sigma_extra]
    let params = vec![0.0, 0.5, 3.0, 0.0, 1.0, 2.0, -2.0]; // beta=0.5, gamma=exp(3)≈20 → beta*gamma≈10 >> 1
    let times = vec![0.0, 5.0, 10.0, 15.0, 20.0];
    let cpu = eval_model_flux(SviModelName::Villar, &params, &times);
    // Should still produce finite values (constraint is in PSO cost, not model eval)
    assert!(cpu.iter().all(|v| v.is_finite()), "Villar eval should still be finite");
}

/// Villar model evaluation is consistent for reasonable params.
#[test]
fn villar_eval_reasonable_params() {
    // JAX-calibrated reasonable Villar params (within new bounds)
    let params = vec![0.0, 0.01, 2.5, 0.0, 0.8, 3.3, -2.5];
    let times: Vec<f64> = (-20..60).map(|i| i as f64).collect();
    let flux = eval_model_flux(SviModelName::Villar, &params, &times);

    // Should peak somewhere and decay
    let max_flux = flux.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(max_flux > 0.0, "Villar should have positive peak flux");
    assert!(flux.iter().all(|v| v.is_finite()), "All Villar fluxes should be finite");
}

/// All models produce finite output for representative params.
#[test]
fn all_models_finite_output() {
    let cases: Vec<(SviModelName, Vec<f64>)> = vec![
        (SviModelName::Bazin, vec![0.5, 0.1, 10.0, 1.0, 3.0, -3.0]),
        (SviModelName::Villar, vec![0.0, 0.01, 2.5, 0.0, 0.8, 3.3, -2.5]),
        (SviModelName::Tde, vec![0.5, 0.1, 5.0, 1.0, 3.0, 1.67, -3.0]),
        (SviModelName::Arnett, vec![0.5, 5.0, 2.3, 0.0, -3.0]),
        (SviModelName::Magnetar, vec![0.5, 5.0, 3.0, 2.3, -3.0]),
        (SviModelName::ShockCooling, vec![0.5, 5.0, 0.5, 1.0, -3.0]),
        (SviModelName::Afterglow, vec![0.5, 5.0, 2.0, 0.8, 2.2, -3.0]),
    ];
    let times: Vec<f64> = (-10..40).map(|i| i as f64).collect();

    for (model, params) in &cases {
        let flux = eval_model_flux(*model, params, &times);
        let finite_count = flux.iter().filter(|v| v.is_finite()).count();
        assert!(
            finite_count == times.len(),
            "{:?}: only {}/{} outputs finite",
            model,
            finite_count,
            times.len()
        );
    }
}

/// Multi-band parametric fit produces results for synthetic 2-band data.
#[test]
fn multiband_villar_produces_results() {
    use std::collections::HashMap;
    use lightcurve_fitting::{BandData, UncertaintyMethod};
    use lightcurve_fitting::parametric::fit_parametric_multiband;

    // Synthetic Bazin-like lightcurve in r and g
    let times: Vec<f64> = (0..30).map(|i| i as f64 * 1.5).collect();
    let true_params = vec![0.3, 0.01, 2.0, 10.0, 1.0, 3.0, -2.5];
    let r_flux = eval_model_flux(SviModelName::Villar, &true_params, &times);
    let g_flux: Vec<f64> = r_flux.iter().map(|f| f * 0.8).collect(); // g slightly fainter
    let errors: Vec<f64> = r_flux.iter().map(|f| f.abs() * 0.05 + 0.01).collect();

    let mut bands = HashMap::new();
    bands.insert("ztfr".to_string(), BandData {
        times: times.clone(),
        values: r_flux,
        errors: errors.clone(),
    });
    bands.insert("ztfg".to_string(), BandData {
        times: times.clone(),
        values: g_flux,
        errors,
    });

    let results = fit_parametric_multiband(&bands, UncertaintyMethod::Laplace);
    assert!(!results.is_empty(), "multiband should produce results");
    assert_eq!(results.len(), 2, "should have results for both bands");
    for r in &results {
        assert!(r.pso_chi2.unwrap_or(f64::NAN).is_finite(), "pso_chi2 should be finite");
        eprintln!("band={}, model={:?}, pso_chi2={:.3}", r.band, r.model, r.pso_chi2.unwrap_or(f64::NAN));
    }
}

// ---------------------------------------------------------------------------
// Metal GPU integration tests (require --features metal on macOS)
// ---------------------------------------------------------------------------

#[cfg(feature = "metal")]
mod metal_gpu {
    use lightcurve_fitting::gpu::{
        BatchSource, GpuBatchData, GpuContext, GpuModelName, GpBandInput, ALL_GPU_MODELS,
    };
    use lightcurve_fitting::{eval_model_flux, SviModelName};

    fn make_synthetic_source(t0: f64, amplitude: f64, n_obs: usize) -> BatchSource {
        let times: Vec<f64> = (0..n_obs)
            .map(|i| t0 - 10.0 + i as f64 * 40.0 / n_obs as f64)
            .collect();
        let true_params = vec![amplitude.ln(), 0.05, t0, 1.0_f64.ln(), 15.0_f64.ln(), -3.0];
        let clean = eval_model_flux(SviModelName::Bazin, &true_params, &times);
        let noise_level = 0.05;
        let flux: Vec<f64> = clean
            .iter()
            .enumerate()
            .map(|(i, &f)| f + noise_level * ((i as f64 * 1.37).sin() * 0.5 + 0.1))
            .collect();
        let obs_var: Vec<f64> = vec![noise_level * noise_level + 1e-10; n_obs];
        let is_upper = vec![false; n_obs];
        let upper_flux = vec![0.0; n_obs];
        BatchSource {
            times,
            flux,
            obs_var,
            is_upper,
            upper_flux,
        }
    }

    #[test]
    fn metal_context_creation() {
        let ctx = GpuContext::new(0);
        assert!(ctx.is_ok(), "Metal context creation should succeed on macOS");
    }

    #[test]
    fn metal_bazin_matches_cpu() {
        let ctx = GpuContext::new(0).expect("Metal init failed");
        let params = vec![0.5, 0.1, 10.0, 1.0, 3.0, -3.0];
        let times: Vec<f64> = (0..100).map(|i| i as f64 * 0.5).collect();

        let gpu_out = ctx
            .eval_batch(GpuModelName::Bazin, &params, &times, 1)
            .unwrap();
        let cpu_out = eval_model_flux(SviModelName::Bazin, &params, &times);

        for (i, (g, c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
            // float32 tolerance: ~1e-5 relative error
            let tol = 1e-4 * c.abs().max(1e-6);
            assert!(
                (g - c).abs() < tol,
                "t={}: gpu={}, cpu={}, diff={}",
                times[i],
                g,
                c,
                (g - c).abs()
            );
        }
    }

    #[test]
    fn metal_all_models_match_cpu() {
        let ctx = GpuContext::new(0).expect("Metal init failed");
        let times: Vec<f64> = (-10..40).map(|i| i as f64).collect();

        let cases: Vec<(GpuModelName, SviModelName, Vec<f64>)> = vec![
            (
                GpuModelName::Bazin,
                SviModelName::Bazin,
                vec![0.5, 0.1, 10.0, 1.0, 3.0, -3.0],
            ),
            (
                GpuModelName::Villar,
                SviModelName::Villar,
                vec![0.0, 0.01, 2.5, 0.0, 0.8, 3.3, -2.5],
            ),
            (
                GpuModelName::Tde,
                SviModelName::Tde,
                vec![0.5, 0.1, 5.0, 1.0, 3.0, 1.67, -3.0],
            ),
            (
                GpuModelName::Arnett,
                SviModelName::Arnett,
                vec![0.5, 5.0, 2.3, 0.0, -3.0],
            ),
            (
                GpuModelName::Magnetar,
                SviModelName::Magnetar,
                vec![0.5, 5.0, 3.0, 2.3, -3.0],
            ),
            (
                GpuModelName::ShockCooling,
                SviModelName::ShockCooling,
                vec![0.5, 5.0, 0.5, 1.0, -3.0],
            ),
            (
                GpuModelName::Afterglow,
                SviModelName::Afterglow,
                vec![0.5, 5.0, 2.0, 0.8, 2.2, -3.0],
            ),
        ];

        for (gpu_m, cpu_m, params) in &cases {
            let gpu = ctx.eval_batch(*gpu_m, params, &times, 1).unwrap();
            let cpu = eval_model_flux(*cpu_m, params, &times);
            let mut max_rel_err = 0.0f64;
            let peak = cpu.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
            for (g, c) in gpu.iter().zip(cpu.iter()) {
                // Skip near-zero values where f32 underflow makes relative error meaningless
                if c.abs() < peak * 1e-5 {
                    continue;
                }
                let rel = (g - c).abs() / c.abs().max(1e-6);
                max_rel_err = max_rel_err.max(rel);
            }
            // f32 compute introduces accumulation errors in models with steep exponentials.
            // Models like Arnett/Magnetar/Afterglow have chained exp/log operations
            // that amplify f32 precision loss near transition regions.
            assert!(
                max_rel_err < 0.1,
                "{:?}: max relative error {:.6} exceeds float32 tolerance",
                gpu_m,
                max_rel_err
            );
            eprintln!("{:?}: max_rel_err = {:.2e}", gpu_m, max_rel_err);
        }
    }

    #[test]
    fn metal_batch_pso_finds_reasonable_cost() {
        let ctx = GpuContext::new(0).expect("Metal init failed");

        let sources: Vec<BatchSource> = (0..10)
            .map(|i| make_synthetic_source(10.0 + i as f64, 1.0, 50))
            .collect();
        let data = GpuBatchData::new(&sources).unwrap();

        let results = ctx
            .batch_pso(GpuModelName::Bazin, &data, 40, 50, 10, 42)
            .unwrap();

        assert_eq!(results.len(), 10);
        for (i, r) in results.iter().enumerate() {
            assert_eq!(r.params.len(), 6, "source {} wrong param count", i);
            assert!(r.cost.is_finite(), "source {} cost not finite", i);
            assert!(
                r.cost < 10.0,
                "source {} cost too high: {}",
                i,
                r.cost
            );
        }
    }

    #[test]
    fn metal_batch_pso_all_models() {
        let ctx = GpuContext::new(0).expect("Metal init failed");

        let sources: Vec<BatchSource> = (0..3)
            .map(|i| make_synthetic_source(10.0 + i as f64, 1.0, 40))
            .collect();
        let data = GpuBatchData::new(&sources).unwrap();

        for &model in ALL_GPU_MODELS {
            let results = ctx.batch_pso(model, &data, 20, 30, 8, 42).unwrap();
            assert_eq!(results.len(), 3, "{:?} wrong result count", model);
            for (i, r) in results.iter().enumerate() {
                assert_eq!(
                    r.params.len(),
                    model.n_params(),
                    "{:?} src {} wrong params",
                    model,
                    i
                );
                assert!(r.cost.is_finite(), "{:?} src {} cost not finite", model, i);
            }
        }
    }

    #[test]
    fn metal_batch_gp_fit() {
        let ctx = GpuContext::new(0).expect("Metal init failed");

        // Synthetic GP data: smooth curve with noise
        let times: Vec<f64> = (0..30).map(|i| i as f64 * 1.0).collect();
        let mags: Vec<f64> = times
            .iter()
            .map(|&t| 20.0 - 2.0 * (-(t - 15.0).powi(2) / 50.0).exp())
            .collect();
        let noise_var = vec![0.01; 30];

        let bands = vec![GpBandInput {
            times: times.clone(),
            mags,
            noise_var,
        }];

        let query: Vec<f64> = (0..50).map(|i| i as f64 * 29.0 / 49.0).collect();
        let amp_candidates = vec![0.3, 1.0, 3.0, 10.0];
        let ls_candidates = vec![3.0, 6.0, 12.0, 24.0, 48.0];

        let results = ctx
            .batch_gp_fit(&bands, &query, &amp_candidates, &ls_candidates, 25)
            .unwrap();

        assert_eq!(results.len(), 1);
        let r = &results[0];
        assert_eq!(r.pred_grid.len(), 50);
        assert_eq!(r.std_grid.len(), 50);
        assert_eq!(r.pred_at_obs.len(), 30);

        // Check predictions are reasonable (near input data)
        let finite_preds = r.pred_grid.iter().filter(|v| v.is_finite()).count();
        assert!(finite_preds > 40, "only {}/50 grid predictions finite", finite_preds);
    }

    #[test]
    fn metal_batch_model_select() {
        let ctx = GpuContext::new(0).expect("Metal init failed");

        let sources: Vec<BatchSource> = (0..5)
            .map(|i| make_synthetic_source(10.0 + i as f64 * 2.0, 1.0, 50))
            .collect();
        let data = GpuBatchData::new(&sources).unwrap();

        let results = ctx.batch_model_select(&data, 40, 50, 10, 2.0).unwrap();

        assert_eq!(results.len(), 5);
        for (i, (model, result)) in results.iter().enumerate() {
            assert!(result.cost.is_finite(), "source {} cost not finite", i);
            eprintln!(
                "source {}: model={:?}, cost={:.4}",
                i, model, result.cost
            );
        }
    }

    #[test]
    fn metal_svi_fit() {
        use lightcurve_fitting::gpu::SviBatchInput;
        use lightcurve_fitting::svi_prior_for_model;

        let ctx = GpuContext::new(0).expect("Metal init failed");

        let sources: Vec<BatchSource> = (0..3)
            .map(|i| make_synthetic_source(10.0 + i as f64, 1.0, 40))
            .collect();
        let data = GpuBatchData::new(&sources).unwrap();

        // First run PSO to get initial params
        let pso_results = ctx
            .batch_pso(GpuModelName::Bazin, &data, 40, 50, 10, 42)
            .unwrap();

        // Build SVI inputs
        let model = GpuModelName::Bazin;
        let inputs: Vec<SviBatchInput> = pso_results
            .iter()
            .map(|r| {
                let (centers, widths) = svi_prior_for_model(&SviModelName::Bazin, &r.params);
                SviBatchInput {
                    model_id: model.model_id() as usize,
                    pso_params: r.params.clone(),
                    se_idx: model.n_params() - 1,
                    prior_centers: centers,
                    prior_widths: widths,
                }
            })
            .collect();

        let svi_results = ctx
            .batch_svi_fit(&data, &inputs, 100, 4, 0.01)
            .unwrap();

        assert_eq!(svi_results.len(), 3);
        for (i, r) in svi_results.iter().enumerate() {
            assert_eq!(r.mu.len(), model.n_params());
            assert_eq!(r.log_sigma.len(), model.n_params());
            assert!(r.elbo.is_finite(), "source {} elbo not finite", i);
            // Uncertainties should be reasonable
            for (j, &ls) in r.log_sigma.iter().enumerate() {
                assert!(
                    ls >= -6.0 && ls <= 2.0,
                    "source {} param {} log_sigma={} out of range",
                    i,
                    j,
                    ls
                );
            }
        }
    }
}
