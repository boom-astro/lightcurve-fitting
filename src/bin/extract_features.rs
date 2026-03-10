//! GPU-accelerated feature extraction for transient lightcurves.
//!
//! Reads photometry from AppleCider NPZ or GOPREAUX datacube CSV files,
//! runs all fitters (1D GP, 2D GP, parametric with all models + MultiBazin)
//! on GPU, and outputs per-source feature vectors.
//!
//! Usage:
//!   extract-features --input-dir /path/to/data --source-list sources.csv \
//!                    --format npz --output-dir /path/to/output

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::Array2;
use ndarray_npy::NpzReader;

use lightcurve_fitting::{
    BandData, build_flux_bands, build_mag_bands,
    fit_nonparametric_batch_gpu, fit_gp_2d_batch_gpu,
    finalize_parametric_with_gpu_svi, svi_prior_for_model, svi_model_meta,
    GpuPsoBandResult, SviBatchInput,
    features::extract_features_from_results,
    gpu::{GpuContext, BatchSource, GpuBatchData},
    thermal::fit_thermal,
};

#[derive(Parser)]
#[command(name = "extract-features")]
#[command(about = "GPU-accelerated feature extraction for transient lightcurves")]
struct Cli {
    /// Input directory containing photometry data.
    #[arg(long)]
    input_dir: PathBuf,

    /// Source list CSV file.
    /// For NPZ format: obj_id,split columns.
    /// For datacube format: Name,Type,Subtype,... (GOPREAUX caat.csv).
    #[arg(long)]
    source_list: PathBuf,

    /// Input format: "npz" (AppleCider) or "datacube" (GOPREAUX).
    #[arg(long, default_value = "npz")]
    format: String,

    /// Directory to write per-source JSON results and features.
    #[arg(long)]
    output_dir: PathBuf,

    /// Reference band for feature extraction.
    #[arg(long, default_value = "r")]
    ref_band: String,

    /// GPU device ID.
    #[arg(long, default_value = "0")]
    gpu_device: i32,

    /// Batch size for GPU processing.
    #[arg(long, default_value = "512")]
    batch_size: usize,

    /// Number of rayon threads for CPU steps.
    #[arg(long, default_value = "8")]
    threads: usize,

    /// Skip sources that already have output files.
    #[arg(long)]
    skip_existing: bool,
}

// ---------------------------------------------------------------------------
// Source entry types
// ---------------------------------------------------------------------------

struct SourceEntry {
    id: String,
    split: String,    // for NPZ: train/val/test
    #[allow(dead_code)]
    subtype: String,  // for datacube: classification label
}

// ---------------------------------------------------------------------------
// NPZ reader (AppleCider format)
// ---------------------------------------------------------------------------

fn read_photometry_npz(path: &Path) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<String>)> {
    let file = fs::File::open(path).ok()?;
    let mut npz = NpzReader::new(file).ok()?;
    let data: Array2<f32> = npz.by_name("data.npy").ok()?;

    if data.ncols() < 5 { return None; }

    let n_rows = data.nrows();
    let mut times = Vec::with_capacity(n_rows);
    let mut mags = Vec::with_capacity(n_rows);
    let mut mag_errs = Vec::with_capacity(n_rows);
    let mut bands = Vec::with_capacity(n_rows);

    for i in 0..n_rows {
        let dt = data[[i, 0]] as f64;
        let band_id = data[[i, 2]] as i32;
        let logflux = data[[i, 3]] as f64;
        let logflux_err = data[[i, 4]] as f64;

        if !logflux.is_finite() || !logflux_err.is_finite() || logflux_err <= 0.0 {
            continue;
        }

        let mag = -2.5 * logflux + 23.9;
        let mag_err = 2.5 * logflux_err;

        let band_name = match band_id {
            0 => "g", 1 => "r", 2 => "i",
            _ => continue,
        };

        times.push(dt);
        mags.push(mag);
        mag_errs.push(mag_err);
        bands.push(band_name.to_string());
    }

    if times.is_empty() { return None; }
    Some((times, mags, mag_errs, bands))
}

// ---------------------------------------------------------------------------
// Datacube CSV reader (GOPREAUX format)
// ---------------------------------------------------------------------------

fn read_photometry_datacube(dir: &Path, obj_id: &str) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<String>)> {
    // Find the datacube_mangled.csv file
    let csv_name = format!("{}_datacube_mangled.csv", obj_id);
    let csv_path = dir.join(&csv_name);

    if !csv_path.exists() {
        // Search subdirectories
        return find_datacube_recursive(dir, obj_id);
    }

    parse_datacube_csv(&csv_path)
}

fn find_datacube_recursive(dir: &Path, obj_id: &str) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<String>)> {
    let csv_name = format!("{}_datacube_mangled.csv", obj_id);

    for entry in fs::read_dir(dir).ok()? {
        let entry = entry.ok()?;
        let path = entry.path();
        if path.is_dir() {
            // Check if this dir contains the object
            let candidate = path.join(&csv_name);
            if candidate.exists() {
                return parse_datacube_csv(&candidate);
            }
            // Check one level deeper (Type/Subtype/Name/)
            if let Some(result) = find_datacube_recursive(&path, obj_id) {
                return Some(result);
            }
        }
    }
    None
}

fn parse_datacube_csv(path: &Path) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<String>)> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_path(path).ok()?;

    let headers = rdr.headers().ok()?.clone();
    let phase_idx = headers.iter().position(|h| h == "Phase")?;
    let filter_idx = headers.iter().position(|h| h == "Filter")?;
    let mag_idx = headers.iter().position(|h| h == "Mag")?;
    let magerr_idx = headers.iter().position(|h| h == "Magerr")?;
    let nondet_idx = headers.iter().position(|h| h == "Nondetection");

    let mut times = Vec::new();
    let mut mags = Vec::new();
    let mut mag_errs = Vec::new();
    let mut bands = Vec::new();

    for result in rdr.records() {
        let record = match result {
            Ok(r) => r,
            Err(_) => continue,
        };

        // Skip non-detections
        if let Some(nd_idx) = nondet_idx {
            if let Some(val) = record.get(nd_idx) {
                if val.trim() == "True" || val.trim() == "true" || val.trim() == "1" {
                    continue;
                }
            }
        }

        let phase: f64 = match record.get(phase_idx).and_then(|s| s.parse::<f64>().ok()) {
            Some(v) if v.is_finite() => v,
            _ => continue,
        };
        let mag: f64 = match record.get(mag_idx).and_then(|s| s.parse::<f64>().ok()) {
            Some(v) if v.is_finite() => v,
            _ => continue,
        };
        let mag_err: f64 = match record.get(magerr_idx).and_then(|s| s.parse::<f64>().ok()) {
            Some(v) if v.is_finite() && v > 0.0 => v,
            _ => continue,
        };

        let filter = match record.get(filter_idx) {
            Some(f) => f.trim().to_string(),
            None => continue,
        };

        // Map GOPREAUX filter names to standard band names
        let band = match filter.as_str() {
            "g" | "ztfg" | "ps1::g" => "g",
            "r" | "ztfr" | "ps1::r" => "r",
            "i" | "ztfi" | "ps1::i" => "i",
            "z" | "ps1::z" => "z",
            "y" | "ps1::y" => "y",
            "u" => "u",
            "c" => "c",      // ATLAS cyan
            "o" => "o",      // ATLAS orange
            // Skip bands without known wavelengths in our system
            _ => continue,
        };

        times.push(phase);
        mags.push(mag);
        mag_errs.push(mag_err);
        bands.push(band.to_string());
    }

    if times.is_empty() { return None; }
    Some((times, mags, mag_errs, bands))
}

// ---------------------------------------------------------------------------
// Source list readers
// ---------------------------------------------------------------------------

fn read_npz_source_list(path: &Path) -> Vec<SourceEntry> {
    let mut rdr = csv::Reader::from_path(path).expect("Failed to open source list CSV");
    let headers = rdr.headers().expect("No headers in source list").clone();
    // Find column indices by name, with fallbacks
    let id_idx = headers.iter().position(|h| h == "obj_id").unwrap_or(0);
    let split_idx = headers.iter().position(|h| h == "split").unwrap_or(1);

    let mut entries = Vec::new();
    for result in rdr.records() {
        let record = result.expect("Failed to read CSV record");
        let obj_id = record.get(id_idx).unwrap_or("").to_string();
        let split = record.get(split_idx).unwrap_or("").to_string();
        if !obj_id.is_empty() {
            entries.push(SourceEntry {
                id: obj_id,
                split,
                subtype: String::new(),
            });
        }
    }
    entries
}

fn read_datacube_source_list(path: &Path) -> Vec<SourceEntry> {
    let mut rdr = csv::Reader::from_path(path).expect("Failed to open CAAT CSV");
    let headers = rdr.headers().expect("No headers").clone();
    let name_idx = headers.iter().position(|h| h == "Name").unwrap_or(0);
    let type_idx = headers.iter().position(|h| h == "Type");
    let subtype_idx = headers.iter().position(|h| h == "Subtype");

    let mut entries = Vec::new();
    for result in rdr.records() {
        let record = result.expect("Failed to read CAAT record");
        let name = record.get(name_idx).unwrap_or("").to_string();
        let typ = type_idx.and_then(|i| record.get(i)).unwrap_or("").to_string();
        let subtype = subtype_idx.and_then(|i| record.get(i)).unwrap_or("").to_string();
        if !name.is_empty() {
            entries.push(SourceEntry {
                id: name,
                split: typ,
                subtype,
            });
        }
    }
    entries
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();

    rayon::ThreadPoolBuilder::new()
        .num_threads(cli.threads)
        .build_global()
        .expect("Failed to set rayon thread pool");

    fs::create_dir_all(&cli.output_dir).expect("Failed to create output directory");

    // Read source list
    let sources = match cli.format.as_str() {
        "npz" => read_npz_source_list(&cli.source_list),
        "datacube" => read_datacube_source_list(&cli.source_list),
        other => {
            eprintln!("Unknown format '{}'. Use 'npz' or 'datacube'.", other);
            std::process::exit(1);
        }
    };
    let total = sources.len();
    eprintln!("Found {} sources", total);

    // Filter to pending sources
    let pending: Vec<&SourceEntry> = if cli.skip_existing {
        sources.iter()
            .filter(|s| !cli.output_dir.join(format!("{}.json", s.id)).exists())
            .collect()
    } else {
        sources.iter().collect()
    };
    let n_pending = pending.len();
    eprintln!("{} already done, {} remaining", total - n_pending, n_pending);

    if n_pending == 0 {
        eprintln!("Nothing to do.");
        return;
    }

    // Initialize GPU
    eprintln!("Initializing GPU device {}...", cli.gpu_device);
    let gpu = GpuContext::new(cli.gpu_device).expect("Failed to initialize GPU");
    eprintln!("GPU initialized.");

    // Process in batches
    let batch_size = cli.batch_size;
    let n_batches = (n_pending + batch_size - 1) / batch_size;

    let pb = ProgressBar::new(n_pending as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
            .unwrap()
            .progress_chars("=>-"),
    );

    let success_count = AtomicUsize::new(0);
    let fail_count = AtomicUsize::new(0);
    let total_start = Instant::now();

    for batch_idx in 0..n_batches {
        let batch_start = batch_idx * batch_size;
        let batch_end = (batch_start + batch_size).min(n_pending);
        let batch = &pending[batch_start..batch_end];

        // Read all photometry for this batch
        let mut mag_bands_vec: Vec<Option<HashMap<String, BandData>>> = Vec::with_capacity(batch.len());
        let mut flux_bands_vec: Vec<Option<HashMap<String, BandData>>> = Vec::with_capacity(batch.len());
        let mut source_ids: Vec<&str> = Vec::with_capacity(batch.len());

        for entry in batch {
            source_ids.push(&entry.id);

            let photo = match cli.format.as_str() {
                "npz" => {
                    let npz_path = cli.input_dir.join(&entry.split).join(format!("{}.npz", entry.id));
                    read_photometry_npz(&npz_path)
                }
                "datacube" => read_photometry_datacube(&cli.input_dir, &entry.id),
                _ => None,
            };

            match photo {
                Some((times, mags, mag_errs, bands)) if times.len() >= 5 => {
                    let mb = build_mag_bands(&times, &mags, &mag_errs, &bands);
                    let fb = build_flux_bands(&times, &mags, &mag_errs, &bands);
                    mag_bands_vec.push(Some(mb));
                    flux_bands_vec.push(Some(fb));
                }
                _ => {
                    mag_bands_vec.push(None);
                    flux_bands_vec.push(None);
                }
            }
        }

        // Separate valid sources for GPU batch processing
        let valid_indices: Vec<usize> = (0..batch.len())
            .filter(|&i| mag_bands_vec[i].is_some())
            .collect();

        if valid_indices.is_empty() {
            for _ in batch {
                fail_count.fetch_add(1, Ordering::Relaxed);
                pb.inc(1);
            }
            continue;
        }

        let valid_mag_bands: Vec<HashMap<String, BandData>> = valid_indices.iter()
            .map(|&i| mag_bands_vec[i].clone().unwrap())
            .collect();

        let valid_flux_bands: Vec<HashMap<String, BandData>> = valid_indices.iter()
            .map(|&i| flux_bands_vec[i].clone().unwrap())
            .collect();

        // ---- Step 1: GPU nonparametric batch (1D GP) ----
        let t_step = Instant::now();
        let np_results = fit_nonparametric_batch_gpu(&gpu, &valid_mag_bands);
        eprintln!("  Step 1 (1D GP): {:.2}s", t_step.elapsed().as_secs_f64());

        // ---- Step 2: GPU 2D GP batch ----
        let t_step = Instant::now();
        let gp2d_results = fit_gp_2d_batch_gpu(&gpu, &valid_mag_bands, 50);
        eprintln!("  Step 2 (2D GP): {:.2}s", t_step.elapsed().as_secs_f64());

        // ---- Step 3: GPU parametric (all 8 models + MultiBazin) ----
        // Build per-band BatchSources for all valid sources.
        // Track (source_vi, band_name) for each batch entry.
        let mut all_batch_sources: Vec<BatchSource> = Vec::new();
        let mut band_map: Vec<(usize, String)> = Vec::new(); // (vi, band_name)

        for (vi, flux_bands) in valid_flux_bands.iter().enumerate() {
            // Sort bands by descending obs count (matches fit_parametric ordering)
            let mut sorted_bands: Vec<(&String, &BandData)> = flux_bands.iter().collect();
            sorted_bands.sort_by(|a, b| b.1.times.len().cmp(&a.1.times.len()));

            for (band_name, bd) in &sorted_bands {
                if bd.values.is_empty() { continue; }
                let peak_flux = bd.values.iter().cloned().fold(f64::MIN, f64::max);
                if peak_flux <= 0.0 { continue; }

                let snr_threshold = 3.0;
                let norm_flux: Vec<f64> = bd.values.iter().map(|f| f / peak_flux).collect();
                let norm_err: Vec<f64> = bd.errors.iter().map(|e| e / peak_flux).collect();
                let obs_var: Vec<f64> = norm_err.iter().map(|e| e * e + 1e-10).collect();
                let is_upper: Vec<bool> = bd.values.iter().zip(bd.errors.iter())
                    .map(|(f, e)| *e > 0.0 && (*f / *e) < snr_threshold).collect();
                let upper_flux: Vec<f64> = bd.errors.iter()
                    .map(|e| snr_threshold * e / peak_flux).collect();

                all_batch_sources.push(BatchSource {
                    times: bd.times.clone(),
                    flux: norm_flux,
                    obs_var,
                    is_upper,
                    upper_flux,
                });
                band_map.push((vi, (*band_name).clone()));
            }
        }

        // GPU PSO: run batch_model_select for all bands at once
        let t_step = Instant::now();
        let gpu_data = match GpuBatchData::new(&all_batch_sources) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("WARN: GPU batch data upload failed: {}", e);
                for _ in batch { fail_count.fetch_add(1, Ordering::Relaxed); pb.inc(1); }
                continue;
            }
        };

        let model_select_results = gpu.batch_model_select(
            &gpu_data, 30, 60, 12, -5.0, // n_particles, max_iters, stall_iters, bazin_good_enough (run all)
        ).unwrap_or_default();
        eprintln!("  Step 3a (PSO model select): {:.2}s", t_step.elapsed().as_secs_f64());

        // GPU MultiBazin for the first (most-populated) band of each source
        let t_step = Instant::now();
        let mut first_band_indices: Vec<usize> = Vec::new();
        let mut seen_sources: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for (bi, (vi, _)) in band_map.iter().enumerate() {
            if seen_sources.insert(*vi) {
                first_band_indices.push(bi);
            }
        }

        // Build BatchSources for just the first bands
        let mb_sources: Vec<BatchSource> = first_band_indices.iter()
            .map(|&bi| all_batch_sources[bi].clone())
            .collect();
        let mb_data = GpuBatchData::new(&mb_sources).ok();
        let mb_results = mb_data.as_ref().and_then(|data| {
            gpu.batch_pso_multi_bazin(data, &mb_sources, 30, 60, 12, 42).ok()
        });

        // Assemble per-source GpuPsoBandResult arrays
        let n_valid = valid_indices.len();
        let mut per_source_gpu_results: Vec<Vec<GpuPsoBandResult>> = vec![Vec::new(); n_valid];

        for (bi, (vi, _band_name)) in band_map.iter().enumerate() {
            if bi >= model_select_results.len() { break; }
            let (gpu_model, ref pso_result) = model_select_results[bi];

            // per_model_chi2: we only have the best from batch_model_select
            let mut per_model_chi2 = HashMap::new();
            per_model_chi2.insert(gpu_model.to_svi_name(), Some(pso_result.cost * 2.0));

            // MultiBazin: only for the first band of this source
            let mb = if Some(bi) == first_band_indices.iter().find(|&&fbi| band_map[fbi].0 == *vi).copied() {
                let mb_idx = first_band_indices.iter().position(|&fbi| fbi == bi);
                mb_idx.and_then(|mi| {
                    mb_results.as_ref().and_then(|results| results.get(mi).cloned())
                })
            } else {
                None
            };

            per_source_gpu_results[*vi].push(GpuPsoBandResult {
                model: gpu_model.to_svi_name(),
                pso_params: pso_result.params.clone(),
                pso_cost: pso_result.cost,
                per_model_chi2,
                per_model_params: HashMap::new(),
                multi_bazin: mb,
            });
        }
        eprintln!("  Step 3b (MultiBazin + assemble): {:.2}s", t_step.elapsed().as_secs_f64());

        // ---- Step 4: GPU SVI + thermal + feature extraction ----
        let t_step = Instant::now();

        // Build SVI batch inputs from PSO results (all bands across all sources)
        let mut svi_inputs: Vec<SviBatchInput> = Vec::new();
        let mut svi_band_map: Vec<(usize, usize)> = Vec::new(); // (source_vi, band_idx_in_source)
        for vi in 0..n_valid {
            for (bi, gpu_res) in per_source_gpu_results[vi].iter().enumerate() {
                if gpu_res.pso_params.is_empty() { continue; }
                let (model_id, _np, se_idx) = svi_model_meta(&gpu_res.model);
                let (centers, widths) = svi_prior_for_model(&gpu_res.model, &gpu_res.pso_params);
                svi_inputs.push(SviBatchInput {
                    model_id,
                    pso_params: gpu_res.pso_params.clone(),
                    se_idx,
                    prior_centers: centers,
                    prior_widths: widths,
                });
                svi_band_map.push((vi, bi));
            }
        }

        // Run GPU SVI for all bands at once
        let svi_outputs = if !svi_inputs.is_empty() {
            // We need a GpuBatchData with the same observation data layout as PSO
            // Reuse all_batch_sources which is still in scope
            let svi_gpu_data = GpuBatchData::new(&all_batch_sources).ok();
            match svi_gpu_data {
                Some(ref svi_data) if svi_inputs.len() == all_batch_sources.len() => {
                    gpu.batch_svi_fit(svi_data, &svi_inputs, 150, 2, 0.01)
                        .unwrap_or_default()
                }
                _ => {
                    // Fallback: build SVI-specific batch data matching svi_inputs ordering
                    // This happens if some bands were filtered. Build matching BatchSources.
                    let svi_sources: Vec<BatchSource> = svi_band_map.iter().enumerate().map(|(si, _)| {
                        // svi_inputs parallels band_map order since we iterate same order
                        // Find corresponding all_batch_sources entry
                        // The band_map entries are 1:1 with all_batch_sources
                        all_batch_sources[si.min(all_batch_sources.len() - 1)].clone()
                    }).collect();
                    match GpuBatchData::new(&svi_sources) {
                        Ok(svi_data) => {
                            gpu.batch_svi_fit(&svi_data, &svi_inputs, 1000, 4, 0.01)
                                .unwrap_or_default()
                        }
                        Err(_) => Vec::new(),
                    }
                }
            }
        } else {
            Vec::new()
        };

        // Distribute GPU SVI outputs back to per-source arrays
        let mut per_source_svi: Vec<Vec<(Vec<f64>, Vec<f64>, f64)>> = vec![Vec::new(); n_valid];
        for (si, out) in svi_outputs.iter().enumerate() {
            if si < svi_band_map.len() {
                let (vi, _bi) = svi_band_map[si];
                per_source_svi[vi].push((out.mu.clone(), out.log_sigma.clone(), out.elbo));
            }
        }

        eprintln!("  Step 4a (GPU SVI): {:.2}s", t_step.elapsed().as_secs_f64());

        // Finalize (t0 refine + mag_chi2 on CPU) + thermal
        let t_step = Instant::now();
        use rayon::prelude::*;
        let final_results: Vec<_> = (0..n_valid).into_par_iter().map(|vi| {
            let orig_idx = valid_indices[vi];
            let flux = flux_bands_vec[orig_idx].as_ref().unwrap();
            let mag = mag_bands_vec[orig_idx].as_ref().unwrap();

            let param = finalize_parametric_with_gpu_svi(
                flux, &per_source_gpu_results[vi], &per_source_svi[vi],
            );

            // Thermal using trained GPs from nonparametric
            let (_, ref gps) = np_results[vi];
            let thermal = fit_thermal(mag, Some(gps));

            (param, thermal)
        }).collect();
        eprintln!("  Step 4b (thermal + finalize): {:.2}s", t_step.elapsed().as_secs_f64());

        // ---- Step 5: Write results ----
        for (vi, &orig_idx) in valid_indices.iter().enumerate() {
            let source_id = source_ids[orig_idx];
            let (ref np_bands, _) = np_results[vi];
            let (ref param_bands, ref thermal) = &final_results[vi];

            let (gp2d_result, gp2d_thermal) = match &gp2d_results[vi] {
                Some((r, t)) => (Some(r.clone()), Some(t.clone())),
                None => (None, None),
            };

            let features = extract_features_from_results(
                np_bands, param_bands, thermal,
                &gp2d_result, &gp2d_thermal,
                &cli.ref_band,
            );

            let output = serde_json::json!({
                "source_id": source_id,
                "nonparametric": np_bands,
                "parametric": param_bands,
                "thermal": thermal,
                "gp2d": gp2d_result,
                "gp2d_thermal": gp2d_thermal,
                "features": features,
            });

            let out_path = cli.output_dir.join(format!("{}.json", source_id));
            match fs::write(&out_path, serde_json::to_string(&output).unwrap()) {
                Ok(()) => { success_count.fetch_add(1, Ordering::Relaxed); }
                Err(e) => {
                    eprintln!("WARN: write error for {}: {}", source_id, e);
                    fail_count.fetch_add(1, Ordering::Relaxed);
                }
            }
            pb.inc(1);
        }

        // Mark invalid sources
        for i in 0..batch.len() {
            if mag_bands_vec[i].is_none() {
                fail_count.fetch_add(1, Ordering::Relaxed);
                pb.inc(1);
            }
        }
    }

    pb.finish_with_message("done");

    let elapsed = total_start.elapsed();
    let s = success_count.load(Ordering::Relaxed);
    let f = fail_count.load(Ordering::Relaxed);
    eprintln!(
        "Finished: {} succeeded, {} failed out of {} total in {:.1}s ({:.0} sources/sec)",
        s, f, n_pending, elapsed.as_secs_f64(),
        s as f64 / elapsed.as_secs_f64()
    );
}
