#![cfg(feature = "cuda")]

use std::collections::HashMap;
use std::fs;
use std::io::BufRead;

use lightcurve_fitting::common::BandData;
use lightcurve_fitting::gpu::{BatchSource, GpuBatchData, GpuContext, GpuModelName, ALL_GPU_MODELS};
use lightcurve_fitting::{build_flux_bands, fit_parametric, UncertaintyMethod};

fn load_source_csv(name: &str) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<String>)> {
    let path = format!("/fred/oz480/mcoughli/data_ztf/{}/photometry.csv", name);
    let file = fs::File::open(&path).ok()?;
    let reader = std::io::BufReader::new(file);

    let mut times = Vec::new();
    let mut mags = Vec::new();
    let mut errs = Vec::new();
    let mut bands = Vec::new();

    let mut lines = reader.lines();
    let header = lines.next()?.ok()?;
    let cols: Vec<&str> = header.split(',').collect();
    let jd_idx = cols.iter().position(|c| *c == "jd")?;
    let fid_idx = cols.iter().position(|c| *c == "fid")?;
    let mag_idx = cols.iter().position(|c| *c == "magpsf")?;
    let err_idx = cols.iter().position(|c| *c == "sigmapsf")?;

    for line in lines {
        let line = line.ok()?;
        let fields: Vec<&str> = line.split(',').collect();
        let mag_str = fields.get(mag_idx)?.trim();
        let err_str = fields.get(err_idx)?.trim();
        if mag_str.is_empty() || err_str.is_empty() { continue; }
        let jd: f64 = fields[jd_idx].parse().ok()?;
        let mag: f64 = mag_str.parse().ok()?;
        let err: f64 = err_str.parse().ok()?;
        if !mag.is_finite() || !err.is_finite() || err <= 0.0 { continue; }
        let band = match fields[fid_idx].trim() {
            "1" => "g", "2" => "r", "3" => "i", _ => continue,
        };
        times.push(jd);
        mags.push(mag);
        errs.push(err);
        bands.push(band.to_string());
    }

    if times.is_empty() { return None; }
    Some((times, mags, errs, bands))
}

/// Convert flux BandData into a BatchSource for GPU (using the reference band).
fn flux_bands_to_batch_source(flux_bands: &HashMap<String, BandData>) -> Option<BatchSource> {
    // Pick the band with the most observations (same as CPU code)
    let (_, band_data) = flux_bands.iter().max_by_key(|(_, b)| b.values.len())?;

    let fluxes = &band_data.values;
    let flux_errs = &band_data.errors;
    let times = &band_data.times;

    let peak_flux = fluxes.iter().cloned().fold(f64::MIN, f64::max);
    if peak_flux <= 0.0 { return None; }

    let snr_threshold = 3.0;
    let normalized_flux: Vec<f64> = fluxes.iter().map(|f| f / peak_flux).collect();
    let normalized_err: Vec<f64> = flux_errs.iter().map(|e| e / peak_flux).collect();
    let obs_var: Vec<f64> = normalized_err.iter().map(|e| e * e + 1e-10).collect();
    let is_upper: Vec<bool> = fluxes.iter().zip(flux_errs.iter())
        .map(|(f, e)| *e > 0.0 && (*f / *e) < snr_threshold).collect();
    let upper_flux: Vec<f64> = flux_errs.iter()
        .map(|e| snr_threshold * e / peak_flux).collect();

    Some(BatchSource {
        times: times.clone(),
        flux: normalized_flux,
        obs_var,
        is_upper,
        upper_flux,
    })
}

#[test]
#[ignore]
fn bench_gpu_vs_cpu() {
    let targets = [
        "ZTF17aaaacjo", "ZTF17aaabgkn", "ZTF17aaabmmr", "ZTF17aaacdos", "ZTF17aaadyei",
        "ZTF25aaftbug", "ZTF25aaivcgm", "ZTF25aaovvcg", "ZTF25aaxdccu", "ZTF25aahmbod",
        "ZTF25aabylkr", "ZTF25aaizxrf", "ZTF25aafkxdu", "ZTF25aapwhnu", "ZTF25aacaxre",
        "ZTF25aahccao", "ZTF25aadevqv", "ZTF25aadnogd", "ZTF25aairhqk", "ZTF25aajqtfg",
    ];

    // Load all sources
    let mut source_names = Vec::new();
    let mut flux_bands_list = Vec::new();
    let mut batch_sources = Vec::new();
    let mut n_obs_list = Vec::new();

    for name in &targets {
        let Some((times, mags, errs, bands)) = load_source_csv(name) else { continue; };
        let flux_bands = build_flux_bands(&times, &mags, &errs, &bands);
        if let Some(batch_src) = flux_bands_to_batch_source(&flux_bands) {
            n_obs_list.push(batch_src.times.len());
            batch_sources.push(batch_src);
            flux_bands_list.push(flux_bands);
            source_names.push(name.to_string());
        }
    }

    let n = source_names.len();
    eprintln!("\nLoaded {} sources", n);

    // --- CPU: fit each source sequentially (Laplace path) ---
    let cpu_t0 = std::time::Instant::now();
    let cpu_results: Vec<_> = flux_bands_list.iter()
        .map(|fb| fit_parametric(fb, false, UncertaintyMethod::Laplace))
        .collect();
    let cpu_ms = cpu_t0.elapsed().as_millis();

    // --- GPU: batch model selection ---
    let ctx = GpuContext::new(0).expect("CUDA init failed");
    let data = GpuBatchData::new(&batch_sources).expect("GPU upload failed");

    let gpu_t0 = std::time::Instant::now();
    let gpu_results = ctx.batch_model_select(&data, &batch_sources, 40, 50, 10, 2.0)
        .expect("GPU batch model select failed");
    let gpu_ms = gpu_t0.elapsed().as_millis();

    // --- Compare ---
    eprintln!("\n{:<16} {:>6} {:>10} {:>10} {:>12} {:>12} {:>10} {:>10}",
        "source", "n_obs", "cpu_ms", "gpu_ms", "cpu_cost", "gpu_cost", "cpu_model", "gpu_model");
    eprintln!("{}", "-".repeat(96));

    for i in 0..n {
        let cpu_cost = cpu_results[i].first()
            .and_then(|r| r.pso_chi2)
            .unwrap_or(f64::NAN);
        let cpu_model = cpu_results[i].first()
            .map(|r| format!("{:?}", r.model))
            .unwrap_or_else(|| "?".to_string());

        let (gpu_model, ref gpu_r) = gpu_results[i];

        eprintln!("{:<16} {:>6} {:>10} {:>10} {:>12.4} {:>12.4} {:>10} {:>10?}",
            source_names[i], n_obs_list[i],
            "", "",  // individual times not available for GPU batch
            cpu_cost, gpu_r.cost,
            cpu_model, gpu_model);
    }

    eprintln!("{}", "-".repeat(96));
    eprintln!("CPU total (sequential Laplace): {} ms", cpu_ms);
    eprintln!("GPU total (batch model select): {} ms", gpu_ms);
    eprintln!("Speedup: {:.1}x", cpu_ms as f64 / gpu_ms.max(1) as f64);
    eprintln!();
}
