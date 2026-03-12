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
    finalize_all_models_with_gpu_svi, svi_prior_for_model, svi_model_meta,
    GpuPsoBandResult, SviBatchInput, SviModelName,
    features::extract_features_from_results,
    gpu::{GpuContext, GpuModelName, BatchSource, GpuBatchData},
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
    /// For ZTF format: not required (sources discovered from directories).
    #[arg(long)]
    source_list: Option<PathBuf>,

    /// Input format: "npz" (AppleCider), "datacube" (GOPREAUX), or "ztf" (ZTF alerts).
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

/// Build an index of all datacube files in a directory tree (walk once).
fn build_datacube_index(dir: &Path) -> HashMap<String, PathBuf> {
    let mut index = HashMap::new();
    fn walk(dir: &Path, index: &mut HashMap<String, PathBuf>) {
        let entries = match fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, index);
            } else if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.ends_with("_datacube_mangled.csv") {
                    let obj_id = name.trim_end_matches("_datacube_mangled.csv").to_string();
                    index.insert(obj_id, path);
                }
            }
        }
    }
    walk(dir, &mut index);
    index
}

fn read_photometry_datacube(index: &HashMap<String, PathBuf>, obj_id: &str) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<String>)> {
    let path = index.get(obj_id)?;
    parse_datacube_csv(path)
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
// ZTF photometry reader (per-source directory with photometry.csv)
// ---------------------------------------------------------------------------

/// Read ZTF photometry.csv from a source directory.
/// Columns: jd, fid (1=g, 2=r, 3=i), magpsf, sigmapsf, diffmaglim.
/// Rows without magpsf are non-detections (upper limits via diffmaglim).
fn read_photometry_ztf(dir: &Path) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<String>)> {
    let csv_path = dir.join("photometry.csv");
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_path(&csv_path).ok()?;

    let headers = rdr.headers().ok()?.clone();
    let jd_idx = headers.iter().position(|h| h == "jd")?;
    let fid_idx = headers.iter().position(|h| h == "fid")?;
    let mag_idx = headers.iter().position(|h| h == "magpsf")?;
    let err_idx = headers.iter().position(|h| h == "sigmapsf")?;
    let lim_idx = headers.iter().position(|h| h == "diffmaglim");

    let mut times = Vec::new();
    let mut mags = Vec::new();
    let mut mag_errs = Vec::new();
    let mut bands = Vec::new();

    for result in rdr.records() {
        let record = match result {
            Ok(r) => r,
            Err(_) => continue,
        };

        let jd: f64 = match record.get(jd_idx).and_then(|s| s.parse().ok()) {
            Some(v) if v > 0.0 => v,
            _ => continue,
        };

        let fid: i32 = match record.get(fid_idx).and_then(|s| s.parse().ok()) {
            Some(v) => v,
            _ => continue,
        };

        let band = match fid {
            1 => "g",
            2 => "r",
            3 => "i",
            _ => continue,
        };

        // Detection: magpsf is present and valid
        let mag_str = record.get(mag_idx).unwrap_or("").trim().to_string();
        if let Ok(mag) = mag_str.parse::<f64>() {
            if !mag.is_finite() { continue; }
            let err: f64 = match record.get(err_idx).and_then(|s| s.parse::<f64>().ok()) {
                Some(v) if v > 0.0 && v.is_finite() => v,
                _ => continue,
            };
            times.push(jd);
            mags.push(mag);
            mag_errs.push(err);
            bands.push(band.to_string());
        } else if let Some(li) = lim_idx {
            // Non-detection: use diffmaglim as upper limit magnitude with large error
            if let Some(lim) = record.get(li).and_then(|s| s.parse::<f64>().ok()) {
                if lim.is_finite() && lim > 0.0 {
                    times.push(jd);
                    mags.push(lim);
                    mag_errs.push(1.0); // Large error → SNR < 3 → treated as upper limit
                    bands.push(band.to_string());
                }
            }
        }
    }

    if times.is_empty() { return None; }
    Some((times, mags, mag_errs, bands))
}

/// Discover ZTF source directories (each contains photometry.csv).
/// Uses DirEntry::file_type() to avoid extra stat calls on network filesystems.
fn discover_ztf_sources(dir: &Path) -> Vec<SourceEntry> {
    let mut entries = Vec::new();
    let Ok(read_dir) = fs::read_dir(dir) else { return entries; };

    for entry in read_dir.flatten() {
        // Use cached file_type; follow symlinks if needed
        let is_dir = entry.file_type()
            .map(|ft| ft.is_dir() || (ft.is_symlink() && entry.path().is_dir()))
            .unwrap_or(false);
        if !is_dir { continue; }
        let id = entry.file_name().to_string_lossy().to_string();
        if !id.is_empty() && !id.starts_with('.') {
            entries.push(SourceEntry {
                id,
                split: String::new(),
                subtype: String::new(),
            });
        }
    }
    entries.sort_by(|a, b| a.id.cmp(&b.id));
    entries
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
        "npz" => {
            let sl = cli.source_list.as_ref().expect("--source-list required for npz format");
            read_npz_source_list(sl)
        }
        "datacube" => {
            let sl = cli.source_list.as_ref().expect("--source-list required for datacube format");
            read_datacube_source_list(sl)
        }
        "ztf" => discover_ztf_sources(&cli.input_dir),
        other => {
            eprintln!("Unknown format '{}'. Use 'npz', 'datacube', or 'ztf'.", other);
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

    // Build datacube file index (walk directory tree once)
    let datacube_index = if cli.format == "datacube" {
        eprintln!("Indexing datacube files...");
        let idx = build_datacube_index(&cli.input_dir);
        eprintln!("Found {} datacube files.", idx.len());
        idx
    } else {
        HashMap::new()
    };

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
                "datacube" => read_photometry_datacube(&datacube_index, &entry.id),
                "ztf" => read_photometry_ztf(&cli.input_dir.join(&entry.id)),
                _ => None,
            };

            match photo {
                Some((times, mags, mag_errs, bands)) if times.len() >= 2 => {
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

        // GPU PSO: run ALL 8 models for every band
        let t_step = Instant::now();
        let gpu_data = match GpuBatchData::new(&all_batch_sources) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("WARN: GPU batch data upload failed: {}", e);
                for _ in batch { fail_count.fetch_add(1, Ordering::Relaxed); pb.inc(1); }
                continue;
            }
        };

        let all_model_results = gpu.batch_all_models(
            &gpu_data, 30, 60, 12,
        ).unwrap_or_default();
        eprintln!("  Step 3a (PSO all models): {:.2}s", t_step.elapsed().as_secs_f64());

        // GPU MultiBazin for the first (most-populated) band of each source
        let t_step = Instant::now();
        let mut first_band_indices: Vec<usize> = Vec::new();
        let mut seen_sources: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for (bi, (vi, _)) in band_map.iter().enumerate() {
            if seen_sources.insert(*vi) {
                first_band_indices.push(bi);
            }
        }

        let mb_sources: Vec<BatchSource> = first_band_indices.iter()
            .map(|&bi| all_batch_sources[bi].clone())
            .collect();
        let mb_data = GpuBatchData::new(&mb_sources).ok();
        let mb_results = mb_data.as_ref().and_then(|data| {
            gpu.batch_pso_multi_bazin(data, &mb_sources, 30, 60, 12, 42).ok()
        });

        // Assemble per-source, per-band, per-model GpuPsoBandResult arrays
        // per_source_gpu_results[vi] has one entry per (band, model) pair
        let n_valid = valid_indices.len();
        let n_bands_total = all_batch_sources.len();

        // Collect all per-model chi2 and params per band
        let mut per_band_model_chi2: Vec<HashMap<SviModelName, Option<f64>>> = vec![HashMap::new(); n_bands_total];
        let mut per_band_model_params: Vec<HashMap<SviModelName, Vec<f64>>> = vec![HashMap::new(); n_bands_total];
        let mut per_band_best_model: Vec<(GpuModelName, f64)> = vec![(GpuModelName::Bazin, f64::INFINITY); n_bands_total];

        for (model_name, ref results) in &all_model_results {
            let svi_name = model_name.to_svi_name();
            for (bi, r) in results.iter().enumerate() {
                per_band_model_chi2[bi].insert(svi_name.clone(), Some(r.cost * 2.0));
                per_band_model_params[bi].insert(svi_name.clone(), r.params.clone());
                if r.cost < per_band_best_model[bi].1 {
                    per_band_best_model[bi] = (*model_name, r.cost);
                }
            }
        }

        // Build per-source GpuPsoBandResult arrays — one entry per (band, model)
        // Also build a flat list + observation-data mapping for SVI
        let n_models = all_model_results.len();
        let mut per_source_gpu_results: Vec<Vec<GpuPsoBandResult>> = vec![Vec::new(); n_valid];
        let mut per_source_band_names: Vec<Vec<String>> = vec![Vec::new(); n_valid]; // band name for each entry
        let mut svi_inputs: Vec<SviBatchInput> = Vec::new();
        let mut svi_obs_sources: Vec<BatchSource> = Vec::new(); // obs data for each SVI entry
        let mut svi_result_map: Vec<(usize, usize)> = Vec::new(); // (vi, idx in per_source_gpu_results[vi])

        for (bi, (vi, _band_name)) in band_map.iter().enumerate() {
            // MultiBazin: only for the first band of this source
            let mb = if Some(bi) == first_band_indices.iter().find(|&&fbi| band_map[fbi].0 == *vi).copied() {
                let mb_idx = first_band_indices.iter().position(|&fbi| fbi == bi);
                mb_idx.and_then(|mi| {
                    mb_results.as_ref().and_then(|results| results.get(mi).cloned())
                })
            } else {
                None
            };

            for (model_name, ref model_results) in &all_model_results {
                if bi >= model_results.len() { continue; }
                let ref pso_result = model_results[bi];
                let svi_name = model_name.to_svi_name();

                let (best_model, _) = per_band_best_model[bi];
                let gpu_res_idx = per_source_gpu_results[*vi].len();

                per_source_gpu_results[*vi].push(GpuPsoBandResult {
                    model: svi_name.clone(),
                    pso_params: pso_result.params.clone(),
                    pso_cost: pso_result.cost,
                    per_model_chi2: per_band_model_chi2[bi].clone(),
                    per_model_params: HashMap::new(),
                    // MultiBazin only on best model of first band
                    multi_bazin: if *model_name == best_model { mb.clone() } else { None },
                });
                per_source_band_names[*vi].push(band_map[bi].1.clone());

                // Prepare SVI input for this (band, model) pair
                if !pso_result.params.is_empty() {
                    let (model_id, _np, se_idx) = svi_model_meta(&svi_name);
                    let (centers, widths) = svi_prior_for_model(&svi_name, &pso_result.params);
                    svi_inputs.push(SviBatchInput {
                        model_id,
                        pso_params: pso_result.params.clone(),
                        se_idx,
                        prior_centers: centers,
                        prior_widths: widths,
                    });
                    svi_obs_sources.push(all_batch_sources[bi].clone());
                    svi_result_map.push((*vi, gpu_res_idx));
                }
            }
        }
        eprintln!("  Step 3b (MultiBazin + assemble): {:.2}s ({} bands × {} models = {} SVI jobs)",
            t_step.elapsed().as_secs_f64(), n_bands_total, n_models, svi_inputs.len());

        // ---- Step 4: GPU SVI for all (band, model) pairs ----
        let t_step = Instant::now();

        let svi_outputs = if !svi_inputs.is_empty() {
            match GpuBatchData::new(&svi_obs_sources) {
                Ok(svi_data) => {
                    gpu.batch_svi_fit(&svi_data, &svi_inputs, 150, 2, 0.01)
                        .unwrap_or_default()
                }
                Err(_) => Vec::new(),
            }
        } else {
            Vec::new()
        };

        // Distribute SVI outputs back to per-source arrays
        let mut per_source_svi: Vec<Vec<(Vec<f64>, Vec<f64>, f64)>> = vec![Vec::new(); n_valid];
        // Initialize with empty entries matching per_source_gpu_results
        for vi in 0..n_valid {
            per_source_svi[vi] = vec![(Vec::new(), Vec::new(), f64::NAN); per_source_gpu_results[vi].len()];
        }
        for (si, out) in svi_outputs.iter().enumerate() {
            if si < svi_result_map.len() {
                let (vi, idx) = svi_result_map[si];
                per_source_svi[vi][idx] = (out.mu.clone(), out.log_sigma.clone(), out.elbo);
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

            let param = finalize_all_models_with_gpu_svi(
                flux, &per_source_gpu_results[vi], &per_source_svi[vi],
                &per_source_band_names[vi],
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
