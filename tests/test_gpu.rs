#![cfg(feature = "cuda")]

use lightcurve_fitting::gpu::{BatchSource, GpuBatchData, GpuContext, GpuModelName, ALL_GPU_MODELS};
use lightcurve_fitting::{eval_model_flux, SviModelName};

// ---------------------------------------------------------------------------
// Forward evaluation tests
// ---------------------------------------------------------------------------

#[test]
fn gpu_bazin_matches_cpu() {
    let ctx = GpuContext::new(0).expect("CUDA init failed");
    let params = vec![0.5, 0.1, 10.0, 1.0, 3.0, -3.0];
    let times: Vec<f64> = (0..100).map(|i| i as f64 * 0.5).collect();

    let gpu_out = ctx.eval_batch(GpuModelName::Bazin, &params, &times, 1).unwrap();
    let cpu_out = eval_model_flux(SviModelName::Bazin, &params, &times);

    for (i, (g, c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
        assert!((g - c).abs() < 1e-10, "t={}: gpu={}, cpu={}", times[i], g, c);
    }
}

#[test]
fn gpu_all_models_match_cpu() {
    let ctx = GpuContext::new(0).expect("CUDA init failed");
    let times: Vec<f64> = (-10..40).map(|i| i as f64).collect();

    let cases: Vec<(GpuModelName, SviModelName, Vec<f64>)> = vec![
        (GpuModelName::Bazin, SviModelName::Bazin, vec![0.5, 0.1, 10.0, 1.0, 3.0, -3.0]),
        (GpuModelName::Villar, SviModelName::Villar, vec![0.5, 0.01, 2.0, 10.0, 1.0, 3.0, -3.0]),
        (GpuModelName::Tde, SviModelName::Tde, vec![0.5, 0.1, 5.0, 1.0, 3.0, 1.67, -3.0]),
        (GpuModelName::Arnett, SviModelName::Arnett, vec![0.5, 5.0, 2.3, 0.0, -3.0]),
        (GpuModelName::Magnetar, SviModelName::Magnetar, vec![0.5, 5.0, 3.0, 2.3, -3.0]),
        (GpuModelName::ShockCooling, SviModelName::ShockCooling, vec![0.5, 5.0, 0.5, 1.0, -3.0]),
        (GpuModelName::Afterglow, SviModelName::Afterglow, vec![0.5, 5.0, 2.0, 0.8, 2.2, -3.0]),
    ];

    for (gpu_m, cpu_m, params) in &cases {
        let gpu = ctx.eval_batch(*gpu_m, params, &times, 1).unwrap();
        let cpu = eval_model_flux(*cpu_m, params, &times);
        for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            let tol = 1e-8 * c.abs().max(1e-10);
            assert!((g - c).abs() < tol, "{:?} t={}: gpu={}, cpu={}", gpu_m, times[i], g, c);
        }
    }
}

// ---------------------------------------------------------------------------
// Batch PSO tests
// ---------------------------------------------------------------------------

fn make_synthetic_source(t0: f64, amplitude: f64, n_obs: usize) -> BatchSource {
    let times: Vec<f64> = (0..n_obs).map(|i| t0 - 10.0 + i as f64 * 40.0 / n_obs as f64).collect();
    // Generate Bazin lightcurve with known params, add noise
    let true_params = vec![amplitude.ln(), 0.05, t0, 1.0_f64.ln(), 15.0_f64.ln(), -3.0];
    let clean = eval_model_flux(SviModelName::Bazin, &true_params, &times);
    let noise_level = 0.05;
    let flux: Vec<f64> = clean.iter().enumerate().map(|(i, &f)| {
        f + noise_level * ((i as f64 * 1.37).sin() * 0.5 + 0.1)
    }).collect();
    let obs_var: Vec<f64> = vec![noise_level * noise_level + 1e-10; n_obs];
    let is_upper = vec![false; n_obs];
    let upper_flux = vec![0.0; n_obs];

    BatchSource { times, flux, obs_var, is_upper, upper_flux }
}

#[test]
fn gpu_batch_pso_finds_reasonable_cost() {
    let ctx = GpuContext::new(0).expect("CUDA init failed");

    let sources: Vec<BatchSource> = (0..10)
        .map(|i| make_synthetic_source(10.0 + i as f64, 1.0, 50))
        .collect();
    let data = GpuBatchData::new(&sources).unwrap();

    let results = ctx.batch_pso(
        GpuModelName::Bazin, &data, &sources, 40, 50, 10, 42,
    ).unwrap();

    assert_eq!(results.len(), 10);
    for (i, r) in results.iter().enumerate() {
        assert_eq!(r.params.len(), 6, "source {} wrong param count", i);
        assert!(r.cost.is_finite(), "source {} cost not finite", i);
        assert!(r.cost < 10.0, "source {} cost too high: {}", i, r.cost);
    }
}

#[test]
fn gpu_batch_pso_many_sources() {
    let ctx = GpuContext::new(0).expect("CUDA init failed");

    // 100 sources with varying properties
    let sources: Vec<BatchSource> = (0..100)
        .map(|i| make_synthetic_source(5.0 + i as f64 * 0.3, 0.5 + (i as f64) * 0.01, 30 + i % 20))
        .collect();
    let data = GpuBatchData::new(&sources).unwrap();

    let results = ctx.batch_pso(
        GpuModelName::Bazin, &data, &sources, 40, 50, 10, 42,
    ).unwrap();

    assert_eq!(results.len(), 100);
    let finite_count = results.iter().filter(|r| r.cost.is_finite() && r.cost < 100.0).count();
    assert!(finite_count > 90, "only {}/100 sources converged", finite_count);
}

#[test]
fn gpu_batch_model_select() {
    let ctx = GpuContext::new(0).expect("CUDA init failed");

    let sources: Vec<BatchSource> = (0..5)
        .map(|i| make_synthetic_source(10.0 + i as f64 * 2.0, 1.0, 50))
        .collect();
    let data = GpuBatchData::new(&sources).unwrap();

    let results = ctx.batch_model_select(&data, &sources, 40, 50, 10, 2.0).unwrap();

    assert_eq!(results.len(), 5);
    for (i, (model, result)) in results.iter().enumerate() {
        assert!(result.cost.is_finite(), "source {} cost not finite", i);
        // Bazin-generated data should usually select Bazin
        eprintln!("source {}: model={:?}, cost={:.4}", i, model, result.cost);
    }
}

#[test]
fn gpu_batch_pso_all_models() {
    let ctx = GpuContext::new(0).expect("CUDA init failed");

    let sources: Vec<BatchSource> = (0..3)
        .map(|i| make_synthetic_source(10.0 + i as f64, 1.0, 40))
        .collect();
    let data = GpuBatchData::new(&sources).unwrap();

    // Test each model individually
    for &model in ALL_GPU_MODELS {
        let results = ctx.batch_pso(model, &data, &sources, 20, 30, 8, 42).unwrap();
        assert_eq!(results.len(), 3, "{:?} wrong result count", model);
        for (i, r) in results.iter().enumerate() {
            assert_eq!(r.params.len(), model.n_params(), "{:?} src {} wrong params", model, i);
            assert!(r.cost.is_finite(), "{:?} src {} cost not finite", model, i);
        }
    }
}
