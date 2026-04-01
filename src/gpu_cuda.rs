//! CUDA backend for GPU-accelerated batch model evaluation and fitting.

use std::ffi::c_int;
use std::ptr;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use super::gpu_types::*;

// ---------------------------------------------------------------------------
// CUDA runtime FFI
// ---------------------------------------------------------------------------

type CudaResult = c_int;

extern "C" {
    fn cudaSetDevice(device: c_int) -> CudaResult;
    fn cudaMalloc(devPtr: *mut *mut u8, size: usize) -> CudaResult;
    fn cudaFree(devPtr: *mut u8) -> CudaResult;
    fn cudaMemcpy(dst: *mut u8, src: *const u8, count: usize, kind: c_int) -> CudaResult;
    fn cudaDeviceSynchronize() -> CudaResult;
    fn cudaGetLastError() -> CudaResult;
    fn cudaGetErrorString(error: CudaResult) -> *const i8;
}

const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;

// Host-side launch wrappers (forward eval)
extern "C" {
    fn launch_bazin(p: *const f64, t: *const f64, o: *mut f64, nd: c_int, nt: c_int, np: c_int, grid: c_int, block: c_int);
    fn launch_villar(p: *const f64, t: *const f64, o: *mut f64, nd: c_int, nt: c_int, np: c_int, grid: c_int, block: c_int);
    fn launch_tde(p: *const f64, t: *const f64, o: *mut f64, nd: c_int, nt: c_int, np: c_int, grid: c_int, block: c_int);
    fn launch_arnett(p: *const f64, t: *const f64, o: *mut f64, nd: c_int, nt: c_int, np: c_int, grid: c_int, block: c_int);
    fn launch_magnetar(p: *const f64, t: *const f64, o: *mut f64, nd: c_int, nt: c_int, np: c_int, grid: c_int, block: c_int);
    fn launch_shock_cooling(p: *const f64, t: *const f64, o: *mut f64, nd: c_int, nt: c_int, np: c_int, grid: c_int, block: c_int);
    fn launch_afterglow(p: *const f64, t: *const f64, o: *mut f64, nd: c_int, nt: c_int, np: c_int, grid: c_int, block: c_int);
    fn launch_metzger_kn(p: *const f64, t: *const f64, o: *mut f64, nd: c_int, nt: c_int, np: c_int, grid: c_int, block: c_int);
}

// Batch GP fit + predict launchers
extern "C" {
    fn launch_batch_gp_fit_predict(
        all_times: *const f64,
        all_mags: *const f64,
        all_noise_var: *const f64,
        band_offsets: *const c_int,
        query_times: *const f64,
        hp_amps: *const f64,
        hp_ls: *const f64,
        gp_state: *mut f64,
        pred_grid: *mut f64,
        std_grid: *mut f64,
        n_bands: c_int,
        n_hp_amp: c_int,
        n_hp_ls: c_int,
        max_subsample: c_int,
        grid: c_int,
        block: c_int,
    );

    fn launch_batch_gp_predict_obs(
        gp_state: *const f64,
        all_times: *const f64,
        obs_to_band: *const c_int,
        pred_obs: *mut f64,
        total_obs: c_int,
        grid: c_int,
        block: c_int,
    );
}

// Batch 2D GP fit + predict launcher
extern "C" {
    fn launch_batch_gp2d_fit_predict(
        all_times: *const f64,
        all_waves: *const f64,
        all_mags: *const f64,
        all_noise_var: *const f64,
        src_offsets: *const c_int,
        query_times: *const f64,
        query_waves: *const f64,
        hp_amps: *const f64,
        hp_lst: *const f64,
        hp_lsw: *const f64,
        gp_state: *mut f64,
        pred_grid: *mut f64,
        std_grid: *mut f64,
        n_sources: c_int,
        n_pred: c_int,
        n_hp_amp: c_int,
        n_hp_lst: c_int,
        n_hp_lsw: c_int,
        max_subsample: c_int,
        grid: c_int,
        block: c_int,
    );
}

// Batch PSO cost launcher
extern "C" {
    fn launch_batch_pso_cost(
        all_times: *const f64,
        all_flux: *const f64,
        all_obs_var: *const f64,
        all_is_upper: *const c_int,
        all_upper_flux: *const f64,
        source_offsets: *const c_int,
        positions: *const f64,
        costs: *mut f64,
        n_sources: c_int,
        n_particles: c_int,
        n_params: c_int,
        model_id: c_int,
        grid: c_int,
        block: c_int,
    );

    fn launch_batch_pso_cost_multi_bazin(
        all_times: *const f64,
        all_flux: *const f64,
        all_obs_var: *const f64,
        all_is_upper: *const c_int,
        all_upper_flux: *const f64,
        source_offsets: *const c_int,
        positions: *const f64,
        costs: *mut f64,
        source_k: *const c_int,
        n_sources: c_int,
        n_particles: c_int,
        n_params: c_int,
        grid: c_int,
        block: c_int,
    );
}

// Batch SVI fit launcher
extern "C" {
    fn launch_batch_svi_fit(
        all_times: *const f64,
        all_flux: *const f64,
        all_obs_var: *const f64,
        all_is_upper: *const c_int,
        all_upper_flux: *const f64,
        source_offsets: *const c_int,
        pso_params: *const f64,
        model_ids: *const c_int,
        n_params_arr: *const c_int,
        se_idx_arr: *const c_int,
        prior_centers: *const f64,
        prior_widths: *const f64,
        out_mu: *mut f64,
        out_log_sigma: *mut f64,
        out_elbo: *mut f64,
        n_sources: c_int,
        max_params: c_int,
        n_steps: c_int,
        n_samples: c_int,
        lr: f64,
        grid: c_int,
        block: c_int,
    );
}

// ---------------------------------------------------------------------------
// Safe wrappers
// ---------------------------------------------------------------------------

fn cuda_check(code: CudaResult) -> Result<(), String> {
    if code == 0 {
        Ok(())
    } else {
        let msg = unsafe {
            let ptr = cudaGetErrorString(code);
            if ptr.is_null() {
                "unknown CUDA error".to_string()
            } else {
                std::ffi::CStr::from_ptr(ptr)
                    .to_string_lossy()
                    .into_owned()
            }
        };
        Err(format!("CUDA error {}: {}", code, msg))
    }
}

struct DevBuf {
    ptr: *mut u8,
    #[allow(dead_code)]
    size: usize,
}

impl DevBuf {
    fn alloc(size: usize) -> Result<Self, String> {
        let mut ptr: *mut u8 = ptr::null_mut();
        cuda_check(unsafe { cudaMalloc(&mut ptr, size) })?;
        Ok(Self { ptr, size })
    }

    fn upload<T>(data: &[T]) -> Result<Self, String> {
        let bytes = data.len() * size_of::<T>();
        let buf = Self::alloc(bytes)?;
        cuda_check(unsafe {
            cudaMemcpy(buf.ptr, data.as_ptr() as *const u8, bytes, CUDA_MEMCPY_HOST_TO_DEVICE)
        })?;
        Ok(buf)
    }

    fn download_into<T>(&self, host: &mut [T]) -> Result<(), String> {
        let bytes = host.len() * size_of::<T>();
        cuda_check(unsafe {
            cudaMemcpy(host.as_mut_ptr() as *mut u8, self.ptr, bytes, CUDA_MEMCPY_DEVICE_TO_HOST)
        })
    }

    fn upload_from<T>(&self, data: &[T]) -> Result<(), String> {
        let bytes = data.len() * size_of::<T>();
        cuda_check(unsafe {
            cudaMemcpy(self.ptr, data.as_ptr() as *const u8, bytes, CUDA_MEMCPY_HOST_TO_DEVICE)
        })
    }
}

impl Drop for DevBuf {
    fn drop(&mut self) {
        unsafe { cudaFree(self.ptr); }
    }
}

// ---------------------------------------------------------------------------
// Source data for batch operations
// ---------------------------------------------------------------------------

/// Packed source data resident on the GPU (CUDA backend).
pub struct GpuBatchData {
    d_times: DevBuf,
    d_flux: DevBuf,
    d_obs_var: DevBuf,
    d_is_upper: DevBuf,
    d_upper_flux: DevBuf,
    d_offsets: DevBuf,
    /// Number of sources.
    pub n_sources: usize,
    /// Number of observations per source (host copy of offsets diffs).
    h_n_obs: Vec<usize>,
}

impl GpuBatchData {
    /// Upload packed source data to the GPU.
    pub fn new(sources: &[BatchSource]) -> Result<Self, String> {
        let n_sources = sources.len();

        // Build concatenated arrays + offset table
        let mut all_times = Vec::new();
        let mut all_flux = Vec::new();
        let mut all_obs_var = Vec::new();
        let mut all_is_upper: Vec<c_int> = Vec::new();
        let mut all_upper_flux = Vec::new();
        let mut offsets: Vec<c_int> = Vec::with_capacity(n_sources + 1);
        offsets.push(0);

        for src in sources {
            all_times.extend_from_slice(&src.times);
            all_flux.extend_from_slice(&src.flux);
            all_obs_var.extend_from_slice(&src.obs_var);
            all_is_upper.extend(src.is_upper.iter().map(|&b| b as c_int));
            all_upper_flux.extend_from_slice(&src.upper_flux);
            offsets.push(all_times.len() as c_int);
        }

        let h_n_obs: Vec<usize> = sources.iter().map(|s| s.times.len()).collect();

        Ok(Self {
            d_times: DevBuf::upload(&all_times)?,
            d_flux: DevBuf::upload(&all_flux)?,
            d_obs_var: DevBuf::upload(&all_obs_var)?,
            d_is_upper: DevBuf::upload(&all_is_upper)?,
            d_upper_flux: DevBuf::upload(&all_upper_flux)?,
            d_offsets: DevBuf::upload(&offsets)?,
            n_sources,
            h_n_obs,
        })
    }
}

impl GpuBatchData {
    /// Number of observations for a given source index.
    pub fn n_obs_per_source(&self, source_idx: usize) -> usize {
        self.h_n_obs.get(source_idx).copied().unwrap_or(1)
    }
}

// ---------------------------------------------------------------------------
// Batch PSO result
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// GpuContext
// ---------------------------------------------------------------------------

/// A CUDA context tied to a specific GPU device.
pub struct GpuContext {
    _device: i32,
}

impl GpuContext {
    /// Create a new GPU context on the given device (typically 0).
    pub fn new(device: i32) -> Result<Self, String> {
        cuda_check(unsafe { cudaSetDevice(device) })?;
        Ok(Self { _device: device })
    }

    /// Evaluate a parametric model for many parameter draws at many time points.
    ///
    /// Returns a flat `Vec<f64>` of length `n_draws × n_times`, row-major.
    pub fn eval_batch(
        &self,
        model: GpuModelName,
        params: &[f64],
        times: &[f64],
        n_draws: usize,
    ) -> Result<Vec<f64>, String> {
        let n_params = model.n_params();
        let n_times = times.len();
        if params.len() != n_draws * n_params {
            return Err(format!(
                "params length {} != n_draws({}) x n_params({})",
                params.len(), n_draws, n_params
            ));
        }

        let out_len = n_draws * n_times;
        let d_params = DevBuf::upload(params)?;
        let d_times = DevBuf::upload(times)?;
        let d_out = DevBuf::alloc(out_len * size_of::<f64>())?;

        let total = (n_draws * n_times) as c_int;
        let block: c_int = 256;
        let grid: c_int = (total + block - 1) / block;

        unsafe {
            match model {
                GpuModelName::Bazin => launch_bazin(d_params.ptr as _, d_times.ptr as _, d_out.ptr as _, n_draws as _, n_times as _, n_params as _, grid, block),
                GpuModelName::Villar => launch_villar(d_params.ptr as _, d_times.ptr as _, d_out.ptr as _, n_draws as _, n_times as _, n_params as _, grid, block),
                GpuModelName::Tde => launch_tde(d_params.ptr as _, d_times.ptr as _, d_out.ptr as _, n_draws as _, n_times as _, n_params as _, grid, block),
                GpuModelName::Arnett => launch_arnett(d_params.ptr as _, d_times.ptr as _, d_out.ptr as _, n_draws as _, n_times as _, n_params as _, grid, block),
                GpuModelName::Magnetar => launch_magnetar(d_params.ptr as _, d_times.ptr as _, d_out.ptr as _, n_draws as _, n_times as _, n_params as _, grid, block),
                GpuModelName::ShockCooling => launch_shock_cooling(d_params.ptr as _, d_times.ptr as _, d_out.ptr as _, n_draws as _, n_times as _, n_params as _, grid, block),
                GpuModelName::Afterglow => launch_afterglow(d_params.ptr as _, d_times.ptr as _, d_out.ptr as _, n_draws as _, n_times as _, n_params as _, grid, block),
                GpuModelName::MetzgerKN => launch_metzger_kn(d_params.ptr as _, d_times.ptr as _, d_out.ptr as _, n_draws as _, n_times as _, n_params as _, grid, block),
            }
            cuda_check(cudaGetLastError())?;
            cuda_check(cudaDeviceSynchronize())?;
        }

        let mut output = vec![0.0f64; out_len];
        d_out.download_into(&mut output)?;
        Ok(output)
    }

    /// Run PSO for a single model across many sources simultaneously.
    ///
    /// The cost evaluation (model eval + likelihood) runs on GPU.
    /// The swarm update logic runs on CPU.
    ///
    /// Returns one `BatchPsoResult` per source.
    pub fn batch_pso(
        &self,
        model: GpuModelName,
        data: &GpuBatchData,
        n_particles: usize,
        max_iters: usize,
        stall_iters: usize,
        seed: u64,
    ) -> Result<Vec<BatchPsoResult>, String> {
        let n_sources = data.n_sources;
        let n_params = model.n_params();
        let (lower, upper) = model.pso_bounds();

        let total_particles = n_sources * n_particles;
        let dim = n_params;

        // PSO hyperparameters — linearly decaying inertia
        let w_max = 0.9;
        let w_min = 0.4;
        let c1 = 1.5;
        let c2 = 1.5;
        let inv_max_iters = 1.0 / max_iters as f64;

        // Velocity clamp: half the domain width per dimension
        let v_max: Vec<f64> = (0..dim).map(|d| 0.5 * (upper[d] - lower[d])).collect();

        // Initialize particles: positions, velocities, personal bests
        // Each source gets its own RNG seed for diversity
        let mut positions = vec![0.0; total_particles * dim];
        let mut velocities = vec![0.0; total_particles * dim];
        let mut pbest_pos = vec![0.0; total_particles * dim];
        let mut pbest_cost = vec![f64::INFINITY; total_particles];
        let mut gbest_pos = vec![0.0; n_sources * dim];
        let mut gbest_cost = vec![f64::INFINITY; n_sources];

        for s in 0..n_sources {
            let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(s as u64));
            for p in 0..n_particles {
                let base = (s * n_particles + p) * dim;
                for d in 0..dim {
                    positions[base + d] = lower[d] + rng.random::<f64>() * (upper[d] - lower[d]);
                    velocities[base + d] = v_max[d] * 0.2 * (2.0 * rng.random::<f64>() - 1.0);
                }
            }
        }

        // Allocate GPU buffers for positions and costs
        let d_positions = DevBuf::alloc(total_particles * dim * size_of::<f64>())?;
        let d_costs = DevBuf::alloc(total_particles * size_of::<f64>())?;
        let mut h_costs = vec![0.0f64; total_particles];

        let block: c_int = 256;

        // RNG for velocity updates
        let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(n_sources as u64 + 1));

        // Stall tracking per source
        let mut prev_gbest = vec![f64::INFINITY; n_sources];
        let mut stall_count = vec![0usize; n_sources];
        let mut source_done = vec![false; n_sources];

        for iter in 0..max_iters {
            // Linearly decay inertia weight
            let w = w_max - (w_max - w_min) * (iter as f64) * inv_max_iters;

            // Upload positions, launch cost kernel
            d_positions.upload_from(&positions)?;

            let grid: c_int = ((total_particles as c_int) + block - 1) / block;
            unsafe {
                launch_batch_pso_cost(
                    data.d_times.ptr as _,
                    data.d_flux.ptr as _,
                    data.d_obs_var.ptr as _,
                    data.d_is_upper.ptr as _,
                    data.d_upper_flux.ptr as _,
                    data.d_offsets.ptr as _,
                    d_positions.ptr as _,
                    d_costs.ptr as _,
                    n_sources as c_int,
                    n_particles as c_int,
                    n_params as c_int,
                    model.model_id(),
                    grid,
                    block,
                );
                cuda_check(cudaGetLastError())?;
                cuda_check(cudaDeviceSynchronize())?;
            }

            // Download costs
            d_costs.download_into(&mut h_costs)?;

            // Add population prior penalty (host-side, per-particle)
            let pop_priors = crate::parametric::population_priors_for_gpu(model);
            if !pop_priors.is_empty() {
                for idx in 0..total_particles {
                    let s = idx / n_particles;
                    let base = idx * dim;
                    // n_obs for this source (from offsets)
                    let n_obs = data.n_obs_per_source(s).max(1) as f64;
                    let mut neg_lp = 0.0;
                    for (j, &(center, width)) in pop_priors.iter().enumerate() {
                        if j < dim && width > 0.0 {
                            let z = (positions[base + j] - center) / width;
                            neg_lp += 0.5 * z * z;
                        }
                    }
                    // Scale prior by 1/n_obs² so it decreases with more data
                    // (same scaling as CPU: prior/n added to NLL/n)
                    h_costs[idx] += neg_lp / (n_obs * n_obs);
                }
            }

            // Update personal and global bests
            for s in 0..n_sources {
                for p in 0..n_particles {
                    let idx = s * n_particles + p;
                    let cost = h_costs[idx];
                    if cost < pbest_cost[idx] {
                        pbest_cost[idx] = cost;
                        let base = idx * dim;
                        pbest_pos[base..base + dim].copy_from_slice(&positions[base..base + dim]);
                        if cost < gbest_cost[s] {
                            gbest_cost[s] = cost;
                            let gb = s * dim;
                            gbest_pos[gb..gb + dim].copy_from_slice(&positions[base..base + dim]);
                        }
                    }
                }
            }

            // Update velocities and positions with clamping and wall absorption
            for s in 0..n_sources {
                if source_done[s] {
                    continue;
                }
                for p in 0..n_particles {
                    let idx = s * n_particles + p;
                    let base = idx * dim;
                    let gb = s * dim;
                    for d in 0..dim {
                        let r1: f64 = rng.random();
                        let r2: f64 = rng.random();
                        let mut v = w * velocities[base + d]
                            + c1 * r1 * (pbest_pos[base + d] - positions[base + d])
                            + c2 * r2 * (gbest_pos[gb + d] - positions[base + d]);

                        // Clamp velocity magnitude
                        v = v.clamp(-v_max[d], v_max[d]);

                        let new_pos = positions[base + d] + v;

                        // Wall absorption: clamp and zero velocity on impact
                        if new_pos <= lower[d] {
                            positions[base + d] = lower[d];
                            velocities[base + d] = 0.0;
                        } else if new_pos >= upper[d] {
                            positions[base + d] = upper[d];
                            velocities[base + d] = 0.0;
                        } else {
                            positions[base + d] = new_pos;
                            velocities[base + d] = v;
                        }
                    }
                }
            }

            // Per-source stall detection
            let mut all_done = true;
            for s in 0..n_sources {
                if source_done[s] { continue; }
                let improved = prev_gbest[s] - gbest_cost[s] > 0.01 * prev_gbest[s].abs().max(1e-10);
                if improved {
                    stall_count[s] = 0;
                    prev_gbest[s] = gbest_cost[s];
                } else {
                    stall_count[s] += 1;
                    if stall_count[s] >= stall_iters {
                        source_done[s] = true;
                    }
                }
                if !source_done[s] { all_done = false; }
            }
            if all_done { break; }
        }

        // Collect results
        let results: Vec<BatchPsoResult> = (0..n_sources)
            .map(|s| {
                let gb = s * dim;
                BatchPsoResult {
                    params: gbest_pos[gb..gb + dim].to_vec(),
                    cost: gbest_cost[s],
                }
            })
            .collect();

        Ok(results)
    }

    /// Run PSO model selection across all GPU-supported models for many sources.
    ///
    /// For each source, tries all 7 models (Bazin first with early-stop gate)
    /// and returns the best model + params per source.
    pub fn batch_model_select(
        &self,
        data: &GpuBatchData,
        n_particles: usize,
        max_iters: usize,
        stall_iters: usize,
        bazin_good_enough: f64,
    ) -> Result<Vec<(GpuModelName, BatchPsoResult)>, String> {
        let n_sources = data.n_sources;

        // Track best per source
        let mut best_model = vec![GpuModelName::Bazin; n_sources];
        let mut best_result: Vec<BatchPsoResult> = (0..n_sources)
            .map(|_| BatchPsoResult { params: vec![], cost: f64::INFINITY })
            .collect();

        // Run Bazin first (early-stop gate)
        let bazin_results = self.batch_pso(
            GpuModelName::Bazin, data, n_particles, max_iters, stall_iters, 42,
        )?;
        let mut needs_more = vec![false; n_sources];
        for (s, r) in bazin_results.into_iter().enumerate() {
            if r.cost < best_result[s].cost {
                best_model[s] = GpuModelName::Bazin;
                best_result[s] = r.clone();
            }
            if r.cost >= bazin_good_enough {
                needs_more[s] = true;
            }
        }

        // If any source needs more models, run remaining models
        if needs_more.iter().any(|&b| b) {
            for &model in &ALL_GPU_MODELS[1..] {
                let results = self.batch_pso(
                    model, data, n_particles, max_iters, stall_iters, 42,
                )?;
                for (s, r) in results.into_iter().enumerate() {
                    if needs_more[s] && r.cost < best_result[s].cost {
                        best_model[s] = model;
                        best_result[s] = r;
                    }
                }
            }
        }

        Ok(best_model.into_iter().zip(best_result).collect())
    }

    /// Run PSO for **every** model on all sources, returning all results.
    /// Returns a Vec (per model) of Vec (per source) of (GpuModelName, BatchPsoResult).
    pub fn batch_all_models(
        &self,
        data: &GpuBatchData,
        n_particles: usize,
        max_iters: usize,
        stall_iters: usize,
    ) -> Result<Vec<(GpuModelName, Vec<BatchPsoResult>)>, String> {
        let mut all = Vec::with_capacity(ALL_GPU_MODELS.len());
        for &model in ALL_GPU_MODELS {
            let results = self.batch_pso(model, data, n_particles, max_iters, stall_iters, 42)?;
            all.push((model, results));
        }
        Ok(all)
    }

    // -----------------------------------------------------------------------
    // Batch MultiBazin PSO
    // -----------------------------------------------------------------------

    /// Run greedy MultiBazin fitting (K=1..4) across many sources on the GPU.
    ///
    /// For each K, runs PSO with GPU cost evaluation. K>1 seeds one particle
    /// from the K-1 solution extended with a residual-peak component.
    /// Selects best K per source via BIC.
    ///
    /// `sources` is needed on the CPU side for residual computation when seeding.
    pub fn batch_pso_multi_bazin(
        &self,
        data: &GpuBatchData,
        sources: &[BatchSource],
        n_particles: usize,
        max_iters: usize,
        stall_iters: usize,
        seed: u64,
    ) -> Result<Vec<crate::parametric::MultiBazinResult>, String> {
        use crate::parametric::MultiBazinResult;

        const MAX_K: usize = 4;
        const COMP_PARAMS: usize = 4; // log_A, t0, log_tau_rise, log_tau_fall

        let n_sources = data.n_sources;
        assert_eq!(n_sources, sources.len());

        // Compute per-source time ranges
        let t_ranges: Vec<(f64, f64)> = sources
            .iter()
            .map(|s| {
                let tmin = s.times.iter().cloned().fold(f64::INFINITY, f64::min);
                let tmax = s.times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                (tmin, tmax)
            })
            .collect();

        // Global time range for shared bounds
        let global_t_min = t_ranges.iter().map(|r| r.0).fold(f64::INFINITY, f64::min) - 30.0;
        let global_t_max = t_ranges.iter().map(|r| r.1).fold(f64::NEG_INFINITY, f64::max) + 30.0;

        // Per-source tracking
        let mut best_k = vec![1usize; n_sources];
        let mut best_params: Vec<Vec<f64>> = vec![Vec::new(); n_sources];
        let mut best_cost = vec![f64::INFINITY; n_sources];
        let mut best_bic = vec![f64::INFINITY; n_sources];
        let mut per_k_cost: Vec<Vec<f64>> = vec![Vec::with_capacity(MAX_K); n_sources];
        let mut per_k_bic: Vec<Vec<f64>> = vec![Vec::with_capacity(MAX_K); n_sources];
        let mut prev_params: Vec<Vec<f64>> = vec![Vec::new(); n_sources];
        let mut source_stopped = vec![false; n_sources]; // early-stop per source

        // PSO hyperparameters — linearly decaying inertia
        let w_max_mb = 0.9;
        let w_min_mb = 0.4;
        let c1 = 1.5;
        let c2 = 1.5;

        for k in 1..=MAX_K {
            let n_params = COMP_PARAMS * k + 2;

            // Build bounds for this K using global time range
            let mut lower = Vec::with_capacity(n_params);
            let mut upper = Vec::with_capacity(n_params);
            for _ in 0..k {
                lower.extend_from_slice(&[-3.0, global_t_min, -2.0, -2.0]);
                upper.extend_from_slice(&[3.0, global_t_max, 5.0, 6.0]);
            }
            lower.push(-0.3); upper.push(0.3);   // B
            lower.push(-5.0); upper.push(0.0);    // log_sigma_extra

            let dim = n_params;
            let total_particles = n_sources * n_particles;

            // Initialize particles
            let mut positions = vec![0.0; total_particles * dim];
            let mut velocities = vec![0.0; total_particles * dim];
            let mut pbest_pos = vec![0.0; total_particles * dim];
            let mut pbest_cost = vec![f64::INFINITY; total_particles];
            let mut gbest_pos = vec![0.0; n_sources * dim];
            let mut gbest_cost = vec![f64::INFINITY; n_sources];

            for s in 0..n_sources {
                let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(s as u64).wrapping_add(k as u64 * 1000));

                // For k > 1, build a seeded particle from previous K-1 solution
                let seed_particle: Option<Vec<f64>> = if k > 1 && prev_params[s].len() == COMP_PARAMS * (k - 1) + 2 {
                    let prev = &prev_params[s];
                    let prev_n_comp = (k - 1) * COMP_PARAMS;
                    let mut init = Vec::with_capacity(n_params);
                    // Copy previous components
                    init.extend_from_slice(&prev[..prev_n_comp]);

                    // Compute residuals from previous fit on CPU
                    let src = &sources[s];
                    let n_obs = src.times.len();
                    let mut peak_idx = 0;
                    let mut peak_val = f64::NEG_INFINITY;
                    for i in 0..n_obs {
                        let mut pred = 0.0;
                        for c in 0..(k - 1) {
                            let off = c * COMP_PARAMS;
                            let a = prev[off].exp();
                            let t0 = prev[off + 1];
                            let tau_rise = prev[off + 2].exp();
                            let tau_fall = prev[off + 3].exp();
                            let dt = src.times[i] - t0;
                            let sig = 1.0 / (1.0 + (-dt / tau_rise).exp());
                            pred += a * (-dt / tau_fall).exp() * sig;
                        }
                        pred += prev[prev_n_comp]; // B
                        let resid = src.flux[i] - pred;
                        if !src.is_upper[i] && resid > peak_val {
                            peak_val = resid;
                            peak_idx = i;
                        }
                    }

                    let seed_t0 = src.times[peak_idx];
                    let seed_log_a = peak_val.max(1e-10).ln();
                    init.extend_from_slice(&[seed_log_a, seed_t0, 1.0, 1.0]);

                    // Copy B and sigma_extra from previous
                    init.push(prev[prev_n_comp]);
                    init.push(prev[prev_n_comp + 1]);

                    // Clamp to bounds
                    for i in 0..n_params {
                        init[i] = init[i].clamp(lower[i], upper[i]);
                    }
                    Some(init)
                } else {
                    None
                };

                for p in 0..n_particles {
                    let base = (s * n_particles + p) * dim;
                    if p == 0 {
                        if let Some(ref sp) = seed_particle {
                            // First particle is seeded
                            positions[base..base + dim].copy_from_slice(sp);
                            for d in 0..dim {
                                velocities[base + d] = (upper[d] - lower[d]) * 0.02 * (2.0 * rng.random::<f64>() - 1.0);
                            }
                            continue;
                        }
                    }
                    // Random initialization
                    for d in 0..dim {
                        positions[base + d] = lower[d] + rng.random::<f64>() * (upper[d] - lower[d]);
                        let v_max_d = 0.5 * (upper[d] - lower[d]);
                        velocities[base + d] = v_max_d * 0.2 * (2.0 * rng.random::<f64>() - 1.0);
                    }
                }
            }

            // All sources use the same K for this iteration
            let source_k: Vec<c_int> = vec![k as c_int; n_sources];

            // GPU buffers
            let d_positions = DevBuf::alloc(total_particles * dim * size_of::<f64>())?;
            let d_costs = DevBuf::alloc(total_particles * size_of::<f64>())?;
            let d_source_k = DevBuf::upload(&source_k)?;
            let mut h_costs = vec![0.0f64; total_particles];

            let block: c_int = 256;

            let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(n_sources as u64 + k as u64));
            let mut prev_gbest = vec![f64::INFINITY; n_sources];
            let mut stall_count = vec![0usize; n_sources];
            let mut iter_done = vec![false; n_sources];

            // Velocity clamp for this K
            let v_max_k: Vec<f64> = (0..dim).map(|d| 0.5 * (upper[d] - lower[d])).collect();
            let inv_max_iters_k = 1.0 / max_iters as f64;

            for iter in 0..max_iters {
                let w = w_max_mb - (w_max_mb - w_min_mb) * (iter as f64) * inv_max_iters_k;

                d_positions.upload_from(&positions)?;

                let grid: c_int = ((total_particles as c_int) + block - 1) / block;
                unsafe {
                    launch_batch_pso_cost_multi_bazin(
                        data.d_times.ptr as _,
                        data.d_flux.ptr as _,
                        data.d_obs_var.ptr as _,
                        data.d_is_upper.ptr as _,
                        data.d_upper_flux.ptr as _,
                        data.d_offsets.ptr as _,
                        d_positions.ptr as _,
                        d_costs.ptr as _,
                        d_source_k.ptr as _,
                        n_sources as c_int,
                        n_particles as c_int,
                        n_params as c_int,
                        grid,
                        block,
                    );
                    cuda_check(cudaGetLastError())?;
                    cuda_check(cudaDeviceSynchronize())?;
                }

                d_costs.download_into(&mut h_costs)?;

                // Update personal and global bests
                for s in 0..n_sources {
                    for p in 0..n_particles {
                        let idx = s * n_particles + p;
                        let cost = h_costs[idx];
                        if cost < pbest_cost[idx] {
                            pbest_cost[idx] = cost;
                            let base = idx * dim;
                            pbest_pos[base..base + dim]
                                .copy_from_slice(&positions[base..base + dim]);
                            if cost < gbest_cost[s] {
                                gbest_cost[s] = cost;
                                let gb = s * dim;
                                gbest_pos[gb..gb + dim]
                                    .copy_from_slice(&positions[base..base + dim]);
                            }
                        }
                    }
                }

                // Update velocities and positions with clamping and wall absorption
                for s in 0..n_sources {
                    if iter_done[s] || source_stopped[s] {
                        continue;
                    }
                    for p in 0..n_particles {
                        let idx = s * n_particles + p;
                        let base = idx * dim;
                        let gb = s * dim;
                        for d in 0..dim {
                            let r1: f64 = rng.random();
                            let r2: f64 = rng.random();
                            let mut v = w * velocities[base + d]
                                + c1 * r1 * (pbest_pos[base + d] - positions[base + d])
                                + c2 * r2 * (gbest_pos[gb + d] - positions[base + d]);

                            v = v.clamp(-v_max_k[d], v_max_k[d]);

                            let new_pos = positions[base + d] + v;
                            if new_pos <= lower[d] {
                                positions[base + d] = lower[d];
                                velocities[base + d] = 0.0;
                            } else if new_pos >= upper[d] {
                                positions[base + d] = upper[d];
                                velocities[base + d] = 0.0;
                            } else {
                                positions[base + d] = new_pos;
                                velocities[base + d] = v;
                            }
                        }
                    }
                }

                // Per-source stall detection
                let mut all_done = true;
                for s in 0..n_sources {
                    if iter_done[s] || source_stopped[s] { continue; }
                    let improved = prev_gbest[s] - gbest_cost[s] > 0.01 * prev_gbest[s].abs().max(1e-10);
                    if improved {
                        stall_count[s] = 0;
                        prev_gbest[s] = gbest_cost[s];
                    } else {
                        stall_count[s] += 1;
                        if stall_count[s] >= stall_iters {
                            iter_done[s] = true;
                        }
                    }
                    if !iter_done[s] { all_done = false; }
                }
                if all_done { break; }
            }

            // Collect K results and update BIC tracking
            for s in 0..n_sources {
                if source_stopped[s] {
                    per_k_cost[s].push(f64::NAN);
                    per_k_bic[s].push(f64::NAN);
                    continue;
                }

                let cost = gbest_cost[s];
                let n_obs = sources[s].times.len() as f64;
                let k_bic = 2.0 * cost * n_obs + (n_params as f64) * n_obs.ln();

                per_k_cost[s].push(cost);
                per_k_bic[s].push(k_bic);

                if k_bic < best_bic[s] {
                    best_bic[s] = k_bic;
                    best_cost[s] = cost;
                    best_k[s] = k;
                    let gb = s * dim;
                    best_params[s] = gbest_pos[gb..gb + dim].to_vec();
                }

                // Store params for seeding next K
                let gb = s * dim;
                prev_params[s] = gbest_pos[gb..gb + dim].to_vec();

                // Early stop: adding component didn't help BIC
                if k > 1 && k_bic > per_k_bic[s][k - 2] + 2.0 {
                    source_stopped[s] = true;
                }
            }
        }

        // Build results
        let results: Vec<MultiBazinResult> = (0..n_sources)
            .map(|s| MultiBazinResult {
                best_k: best_k[s],
                params: best_params[s].clone(),
                cost: best_cost[s],
                bic: best_bic[s],
                per_k_cost: per_k_cost[s].clone(),
                per_k_bic: per_k_bic[s].clone(),
            })
            .collect();

        Ok(results)
    }

    // -----------------------------------------------------------------------
    // Batch GP fitting
    // -----------------------------------------------------------------------

    /// Fit DenseGPs for many bands in parallel on the GPU.
    ///
    /// Each band is independently subsampled, hyperparameter-searched, and
    /// predicted at `query_times` (shared grid) plus at its own observation
    /// points.
    ///
    /// Returns one `GpBandOutput` per input band.
    pub fn batch_gp_fit(
        &self,
        bands: &[GpBandInput],
        query_times: &[f64],
        amp_candidates: &[f64],
        ls_candidates: &[f64],
        max_subsample: usize,
    ) -> Result<Vec<GpBandOutput>, String> {
        use crate::sparse_gp::DenseGP;

        let n_bands = bands.len();
        if n_bands == 0 {
            return Ok(Vec::new());
        }

        let n_pred = query_times.len();
        assert!(n_pred == 50, "batch_gp_fit expects 50 query points");

        // Pack band data into concatenated arrays + offset table
        let mut all_times = Vec::new();
        let mut all_mags = Vec::new();
        let mut all_noise_var = Vec::new();
        let mut offsets: Vec<c_int> = Vec::with_capacity(n_bands + 1);
        let mut obs_to_band: Vec<c_int> = Vec::new();
        offsets.push(0);

        for (b, band) in bands.iter().enumerate() {
            all_times.extend_from_slice(&band.times);
            all_mags.extend_from_slice(&band.mags);
            all_noise_var.extend_from_slice(&band.noise_var);
            obs_to_band.extend(std::iter::repeat(b as c_int).take(band.times.len()));
            offsets.push(all_times.len() as c_int);
        }

        let total_obs = all_times.len();
        let gp_state_size = 54; // GP_STATE_SIZE in CUDA

        // Upload inputs to GPU
        let d_times = DevBuf::upload(&all_times)?;
        let d_mags = DevBuf::upload(&all_mags)?;
        let d_noise_var = DevBuf::upload(&all_noise_var)?;
        let d_offsets = DevBuf::upload(&offsets)?;
        let d_query = DevBuf::upload(query_times)?;
        let d_amps = DevBuf::upload(amp_candidates)?;
        let d_ls = DevBuf::upload(ls_candidates)?;

        // Allocate outputs
        let d_gp_state = DevBuf::alloc(n_bands * gp_state_size * size_of::<f64>())?;
        let d_pred_grid = DevBuf::alloc(n_bands * n_pred * size_of::<f64>())?;
        let d_std_grid = DevBuf::alloc(n_bands * n_pred * size_of::<f64>())?;

        // Launch kernel 1: fit + predict at grid
        // One block per band, blockDim = n_hp_total (threads per block = hyperparameter combos)
        let n_hp_total = amp_candidates.len() * ls_candidates.len();
        let block: c_int = n_hp_total as c_int;
        let grid: c_int = n_bands as c_int;

        unsafe {
            launch_batch_gp_fit_predict(
                d_times.ptr as _, d_mags.ptr as _, d_noise_var.ptr as _,
                d_offsets.ptr as _, d_query.ptr as _,
                d_amps.ptr as _, d_ls.ptr as _,
                d_gp_state.ptr as _, d_pred_grid.ptr as _, d_std_grid.ptr as _,
                n_bands as c_int,
                amp_candidates.len() as c_int,
                ls_candidates.len() as c_int,
                max_subsample as c_int,
                grid, block,
            );
            cuda_check(cudaGetLastError())?;
            cuda_check(cudaDeviceSynchronize())?;
        }

        // Launch kernel 2: predict at observation points
        let d_obs_to_band = DevBuf::upload(&obs_to_band)?;
        let d_pred_obs = DevBuf::alloc(total_obs * size_of::<f64>())?;

        let block2: c_int = 256;
        let grid2: c_int = (total_obs as c_int + block2 - 1) / block2;

        unsafe {
            launch_batch_gp_predict_obs(
                d_gp_state.ptr as _, d_times.ptr as _,
                d_obs_to_band.ptr as _, d_pred_obs.ptr as _,
                total_obs as c_int, grid2, block2,
            );
            cuda_check(cudaGetLastError())?;
            cuda_check(cudaDeviceSynchronize())?;
        }

        // Download results
        let mut h_gp_state = vec![0.0f64; n_bands * gp_state_size];
        let mut h_pred_grid = vec![0.0f64; n_bands * n_pred];
        let mut h_std_grid = vec![0.0f64; n_bands * n_pred];
        let mut h_pred_obs = vec![0.0f64; total_obs];

        d_gp_state.download_into(&mut h_gp_state)?;
        d_pred_grid.download_into(&mut h_pred_grid)?;
        d_std_grid.download_into(&mut h_std_grid)?;
        d_pred_obs.download_into(&mut h_pred_obs)?;

        // Build output structs
        let mut results = Vec::with_capacity(n_bands);
        let mut obs_offset = 0usize;

        for b in 0..n_bands {
            let state_off = b * gp_state_size;
            let state = &h_gp_state[state_off..state_off + gp_state_size];
            let m = state[53] as usize;
            let n_obs_band = bands[b].times.len();

            let pred_grid_slice = &h_pred_grid[b * n_pred..(b + 1) * n_pred];
            let std_grid_slice = &h_std_grid[b * n_pred..(b + 1) * n_pred];
            let pred_obs_slice = &h_pred_obs[obs_offset..obs_offset + n_obs_band];
            obs_offset += n_obs_band;

            // Reconstruct DenseGP on CPU
            let dense_gp = if m > 0 && m <= 25 {
                let x_train = state[25..25 + m].to_vec();
                let amp = state[50];
                let inv_2ls2 = state[51];

                // Reconstruct the Cholesky factor by re-fitting on CPU
                // (cheaper than transferring m×m matrix from GPU)
                let sub_nv: Vec<f64> = {
                    let n_obs = bands[b].times.len();
                    if n_obs <= m {
                        bands[b].noise_var.clone()
                    } else {
                        let step = (n_obs - 1) as f64 / (m - 1) as f64;
                        (0..m).map(|i| {
                            let idx = (i as f64 * step + 0.5) as usize;
                            bands[b].noise_var[idx.min(n_obs - 1)]
                        }).collect()
                    }
                };

                DenseGP::fit(&x_train, &{
                    // Recover subsampled values: y = alpha solved from (K+noise)^{-1}(y-ymean)
                    // Easier: just re-fit with same hyperparams
                    let sub_v: Vec<f64> = {
                        let n_obs = bands[b].times.len();
                        if n_obs <= m {
                            bands[b].mags.clone()
                        } else {
                            let step = (n_obs - 1) as f64 / (m - 1) as f64;
                            (0..m).map(|i| {
                                let idx = (i as f64 * step + 0.5) as usize;
                                bands[b].mags[idx.min(n_obs - 1)]
                            }).collect()
                        }
                    };
                    sub_v
                }, &sub_nv, amp, (0.5 / inv_2ls2).sqrt())
            } else {
                None
            };

            results.push(GpBandOutput {
                dense_gp,
                pred_grid: pred_grid_slice.to_vec(),
                std_grid: std_grid_slice.to_vec(),
                pred_at_obs: pred_obs_slice.to_vec(),
            });
        }

        Ok(results)
    }

    /// Batch 2D GP (time × wavelength) fitting for many sources simultaneously.
    ///
    /// Each source's observations across all bands are combined into a single
    /// 2D GP with an anisotropic RBF kernel.
    ///
    /// Returns per-source: (predictions at query grid, std at query grid,
    /// best hyperparams: amp, ls_time, ls_wave, train_rms, n_train).
    pub fn batch_gp_2d(
        &self,
        sources: &[Gp2dInput],
        query_times: &[f64],
        query_waves: &[f64],
        amp_candidates: &[f64],
        lst_candidates: &[f64],
        lsw_candidates: &[f64],
        max_subsample: usize,
    ) -> Result<Vec<Gp2dOutput>, String> {
        let n_sources = sources.len();
        if n_sources == 0 {
            return Ok(Vec::new());
        }
        let n_pred = query_times.len();
        assert_eq!(n_pred, query_waves.len(), "query_times and query_waves must have same length");

        let n_hp_total = amp_candidates.len() * lst_candidates.len() * lsw_candidates.len();
        assert!(n_hp_total <= 64, "max 64 hyperparameter combos for 2D GP");

        // Pack source data into concatenated arrays + offset table
        let mut all_times = Vec::new();
        let mut all_waves = Vec::new();
        let mut all_mags = Vec::new();
        let mut all_noise_var = Vec::new();
        let mut offsets: Vec<c_int> = Vec::with_capacity(n_sources + 1);
        offsets.push(0);

        for src in sources {
            all_times.extend_from_slice(&src.times);
            all_waves.extend_from_slice(&src.waves);
            all_mags.extend_from_slice(&src.mags);
            all_noise_var.extend_from_slice(&src.noise_var);
            offsets.push(all_times.len() as c_int);
        }

        let gp2d_state_size = 125; // GP2D_STATE_SIZE in CUDA

        // Upload inputs
        let d_times = DevBuf::upload(&all_times)?;
        let d_waves = DevBuf::upload(&all_waves)?;
        let d_mags = DevBuf::upload(&all_mags)?;
        let d_noise_var = DevBuf::upload(&all_noise_var)?;
        let d_offsets = DevBuf::upload(&offsets)?;
        let d_query_t = DevBuf::upload(query_times)?;
        let d_query_w = DevBuf::upload(query_waves)?;
        let d_amps = DevBuf::upload(amp_candidates)?;
        let d_lst = DevBuf::upload(lst_candidates)?;
        let d_lsw = DevBuf::upload(lsw_candidates)?;

        // Allocate outputs
        let d_gp_state = DevBuf::alloc(n_sources * gp2d_state_size * size_of::<f64>())?;
        let d_pred_grid = DevBuf::alloc(n_sources * n_pred * size_of::<f64>())?;
        let d_std_grid = DevBuf::alloc(n_sources * n_pred * size_of::<f64>())?;

        let block: c_int = n_hp_total as c_int;

        unsafe {
            launch_batch_gp2d_fit_predict(
                d_times.ptr as _, d_waves.ptr as _,
                d_mags.ptr as _, d_noise_var.ptr as _,
                d_offsets.ptr as _,
                d_query_t.ptr as _, d_query_w.ptr as _,
                d_amps.ptr as _, d_lst.ptr as _, d_lsw.ptr as _,
                d_gp_state.ptr as _, d_pred_grid.ptr as _, d_std_grid.ptr as _,
                n_sources as c_int, n_pred as c_int,
                amp_candidates.len() as c_int,
                lst_candidates.len() as c_int,
                lsw_candidates.len() as c_int,
                max_subsample as c_int,
                n_sources as c_int, block,
            );
            cuda_check(cudaGetLastError())?;
            cuda_check(cudaDeviceSynchronize())?;
        }

        // Download results
        let mut h_gp_state = vec![0.0f64; n_sources * gp2d_state_size];
        let mut h_pred_grid = vec![0.0f64; n_sources * n_pred];
        let mut h_std_grid = vec![0.0f64; n_sources * n_pred];

        d_gp_state.download_into(&mut h_gp_state)?;
        d_pred_grid.download_into(&mut h_pred_grid)?;
        d_std_grid.download_into(&mut h_std_grid)?;

        // Build output structs
        let mut results = Vec::with_capacity(n_sources);
        for s in 0..n_sources {
            let state_off = s * gp2d_state_size;
            let state = &h_gp_state[state_off..state_off + gp2d_state_size];
            let m = state[124] as usize;

            let pred = h_pred_grid[s * n_pred..(s + 1) * n_pred].to_vec();
            let std_dev = h_std_grid[s * n_pred..(s + 1) * n_pred].to_vec();

            if m >= 3 {
                let amp = state[120];
                let inv_2lst2 = state[121];
                let inv_2lsw2 = state[122];
                let ls_time = (0.5 / inv_2lst2).sqrt();
                let ls_wave = (0.5 / inv_2lsw2).sqrt();

                // Compute train RMS from predictions vs training data
                let n_train = sources[s].times.len();
                let train_rms = {
                    // Re-predict at training points using the fitted state
                    let alpha = &state[0..m];
                    let x_t = &state[40..40 + m];
                    let x_w = &state[80..80 + m];
                    let y_mean = state[123];
                    let mut rms = 0.0;
                    let n_eval = n_train.min(100); // don't evaluate all if huge
                    let step = if n_eval < n_train { n_train as f64 / n_eval as f64 } else { 1.0 };
                    for i in 0..n_eval {
                        let idx = (i as f64 * step) as usize;
                        let t = sources[s].times[idx];
                        let w = sources[s].waves[idx];
                        let mut pred_val = y_mean;
                        for j in 0..m {
                            let dt = t - x_t[j];
                            let dw = w - x_w[j];
                            pred_val += amp * (-dt * dt * inv_2lst2 - dw * dw * inv_2lsw2).exp() * alpha[j];
                        }
                        let diff = pred_val - sources[s].mags[idx];
                        rms += diff * diff;
                    }
                    (rms / n_eval as f64).sqrt()
                };

                results.push(Gp2dOutput {
                    pred_grid: pred,
                    std_grid: std_dev,
                    amp,
                    ls_time,
                    ls_wave,
                    train_rms,
                    n_train,
                    success: true,
                });
            } else {
                results.push(Gp2dOutput {
                    pred_grid: pred,
                    std_grid: std_dev,
                    amp: 0.0,
                    ls_time: 0.0,
                    ls_wave: 0.0,
                    train_rms: f64::NAN,
                    n_train: sources[s].times.len(),
                    success: false,
                });
            }
        }

        Ok(results)
    }

    /// Run SVI optimization on GPU for many sources simultaneously.
    ///
    /// Each source has its own model, PSO-initialized parameters, and prior.
    /// Uses finite-difference model gradients with Adam optimizer.
    ///
    /// Returns (mu, log_sigma, elbo) per source. The log_sigma already includes
    /// the sigma inflation factor.
    pub fn batch_svi_fit(
        &self,
        data: &GpuBatchData,
        inputs: &[SviBatchInput],
        n_steps: usize,
        n_samples: usize,
        lr: f64,
    ) -> Result<Vec<SviBatchOutput>, String> {
        let n_sources = inputs.len();
        if n_sources == 0 {
            return Ok(Vec::new());
        }
        assert_eq!(n_sources, data.n_sources, "SVI inputs must match batch data size");

        let max_params: usize = 7; // max across all models (Villar/TDE)

        // Build flat arrays for GPU upload
        let mut h_pso_params = vec![0.0f64; n_sources * max_params];
        let mut h_model_ids = vec![0i32; n_sources];
        let mut h_n_params = vec![0i32; n_sources];
        let mut h_se_idx = vec![0i32; n_sources];
        let mut h_prior_centers = vec![0.0f64; n_sources * max_params];
        let mut h_prior_widths = vec![0.0f64; n_sources * max_params];

        for (i, inp) in inputs.iter().enumerate() {
            let np = inp.pso_params.len().min(max_params);
            let base = i * max_params;
            for j in 0..np {
                h_pso_params[base + j] = inp.pso_params[j];
                h_prior_centers[base + j] = inp.prior_centers[j];
                h_prior_widths[base + j] = inp.prior_widths[j];
            }
            h_model_ids[i] = inp.model_id as i32;
            h_n_params[i] = np as i32;
            h_se_idx[i] = inp.se_idx as i32;
        }

        // Upload to GPU
        let d_pso = DevBuf::upload(&h_pso_params)?;
        let d_model_ids = DevBuf::upload(&h_model_ids)?;
        let d_n_params = DevBuf::upload(&h_n_params)?;
        let d_se_idx = DevBuf::upload(&h_se_idx)?;
        let d_prior_centers = DevBuf::upload(&h_prior_centers)?;
        let d_prior_widths = DevBuf::upload(&h_prior_widths)?;

        // Output buffers
        let d_out_mu = DevBuf::alloc(n_sources * max_params * size_of::<f64>())?;
        let d_out_ls = DevBuf::alloc(n_sources * max_params * size_of::<f64>())?;
        let d_out_elbo = DevBuf::alloc(n_sources * size_of::<f64>())?;

        // One warp (32 threads) per source; block=128 = 4 warps per block
        let block: c_int = 128;
        let total_threads = (n_sources as c_int) * 32;
        let grid: c_int = (total_threads + block - 1) / block;

        unsafe {
            launch_batch_svi_fit(
                data.d_times.ptr as _,
                data.d_flux.ptr as _,
                data.d_obs_var.ptr as _,
                data.d_is_upper.ptr as _,
                data.d_upper_flux.ptr as _,
                data.d_offsets.ptr as _,
                d_pso.ptr as _,
                d_model_ids.ptr as _,
                d_n_params.ptr as _,
                d_se_idx.ptr as _,
                d_prior_centers.ptr as _,
                d_prior_widths.ptr as _,
                d_out_mu.ptr as _,
                d_out_ls.ptr as _,
                d_out_elbo.ptr as _,
                n_sources as c_int,
                max_params as c_int,
                n_steps as c_int,
                n_samples as c_int,
                lr,
                grid,
                block,
            );
            cuda_check(cudaGetLastError())?;
            cuda_check(cudaDeviceSynchronize())?;
        }

        // Download results
        let mut h_mu = vec![0.0f64; n_sources * max_params];
        let mut h_ls = vec![0.0f64; n_sources * max_params];
        let mut h_elbo = vec![0.0f64; n_sources];
        d_out_mu.download_into(&mut h_mu)?;
        d_out_ls.download_into(&mut h_ls)?;
        d_out_elbo.download_into(&mut h_elbo)?;

        let results: Vec<SviBatchOutput> = (0..n_sources)
            .map(|i| {
                let np = inputs[i].pso_params.len().min(max_params);
                let base = i * max_params;
                let mu = h_mu[base..base + np].to_vec();
                let log_sigma = h_ls[base..base + np].to_vec();
                SviBatchOutput {
                    mu,
                    log_sigma,
                    elbo: h_elbo[i],
                }
            })
            .collect();

        Ok(results)
    }
}

// Type definitions (SviBatchInput, SviBatchOutput, Gp2dInput, Gp2dOutput,
// GpBandInput, GpBandOutput, BatchSource, BatchPsoResult, GpuModelName,
// ALL_GPU_MODELS) are in gpu_types.rs and re-exported through gpu.rs.
