//! GPU-accelerated batch model evaluation and fitting via CUDA.
//!
//! Two main capabilities:
//!   1. `eval_batch` — forward model evaluation for many draws/sources
//!   2. `batch_pso` — run PSO model fitting for many sources simultaneously

use std::ffi::c_int;
use std::ptr;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Which model to evaluate on the GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GpuModelName {
    Bazin,
    Villar,
    Tde,
    Arnett,
    Magnetar,
    ShockCooling,
    Afterglow,
    MetzgerKN,
}

impl GpuModelName {
    /// Number of parameters (including sigma_extra) for each model.
    pub fn n_params(self) -> usize {
        match self {
            GpuModelName::Bazin => 6,
            GpuModelName::Villar => 7,
            GpuModelName::Tde => 7,
            GpuModelName::Arnett => 5,
            GpuModelName::Magnetar => 5,
            GpuModelName::ShockCooling => 5,
            GpuModelName::Afterglow => 6,
            GpuModelName::MetzgerKN => 5,
        }
    }

    /// Integer model ID matching the CUDA kernel's switch statement.
    fn model_id(self) -> c_int {
        match self {
            GpuModelName::Bazin => 0,
            GpuModelName::Villar => 1,
            GpuModelName::Tde => 2,
            GpuModelName::Arnett => 3,
            GpuModelName::Magnetar => 4,
            GpuModelName::ShockCooling => 5,
            GpuModelName::Afterglow => 6,
            GpuModelName::MetzgerKN => 7,
        }
    }

    /// PSO parameter bounds (lower, upper) matching the CPU implementation.
    pub fn pso_bounds(self) -> (Vec<f64>, Vec<f64>) {
        match self {
            GpuModelName::Bazin => (
                vec![-3.0, -1.0, -100.0, -2.0, -2.0, -5.0],
                vec![3.0, 1.0, 100.0, 5.0, 6.0, 0.0],
            ),
            GpuModelName::Villar => (
                vec![-3.0, -0.05, -3.0, -100.0, -2.0, -2.0, -5.0],
                vec![3.0, 0.1, 5.0, 100.0, 5.0, 7.0, 0.0],
            ),
            GpuModelName::Tde => (
                vec![-3.0, -1.0, -100.0, -2.0, -1.0, 0.5, -5.0],
                vec![3.0, 1.0, 100.0, 5.0, 6.0, 4.0, 0.0],
            ),
            GpuModelName::Arnett => (
                vec![-3.0, -100.0, 0.5, -3.0, -5.0],
                vec![3.0, 100.0, 4.5, 3.0, 0.0],
            ),
            GpuModelName::Magnetar => (
                vec![-3.0, -100.0, 0.0, 0.5, -5.0],
                vec![3.0, 100.0, 6.0, 4.5, 0.0],
            ),
            GpuModelName::ShockCooling => (
                vec![-3.0, -100.0, 0.1, -1.0, -5.0],
                vec![3.0, 100.0, 3.0, 4.0, 0.0],
            ),
            GpuModelName::Afterglow => (
                vec![-3.0, -100.0, -2.0, -2.0, 0.5, -5.0],
                vec![3.0, 100.0, 6.0, 3.0, 5.0, 0.0],
            ),
            GpuModelName::MetzgerKN => (
                vec![-3.0, -2.0, -1.0, -2.0, -5.0],
                vec![-0.5, -0.5, 2.0, 1.0, 0.0],
            ),
        }
    }
}

/// All GPU-supported models.
pub const ALL_GPU_MODELS: &[GpuModelName] = &[
    GpuModelName::Bazin,
    GpuModelName::Arnett,
    GpuModelName::Tde,
    GpuModelName::Afterglow,
    GpuModelName::Villar,
    GpuModelName::Magnetar,
    GpuModelName::ShockCooling,
    GpuModelName::MetzgerKN,
];

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

/// Pre-normalized observation data for one source.
pub struct BatchSource {
    pub times: Vec<f64>,
    pub flux: Vec<f64>,
    pub obs_var: Vec<f64>,
    pub is_upper: Vec<bool>,
    pub upper_flux: Vec<f64>,
}

/// Packed source data resident on the GPU.
pub struct GpuBatchData {
    d_times: DevBuf,
    d_flux: DevBuf,
    d_obs_var: DevBuf,
    d_is_upper: DevBuf,
    d_upper_flux: DevBuf,
    d_offsets: DevBuf,
    /// Number of sources.
    pub n_sources: usize,
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

        Ok(Self {
            d_times: DevBuf::upload(&all_times)?,
            d_flux: DevBuf::upload(&all_flux)?,
            d_obs_var: DevBuf::upload(&all_obs_var)?,
            d_is_upper: DevBuf::upload(&all_is_upper)?,
            d_upper_flux: DevBuf::upload(&all_upper_flux)?,
            d_offsets: DevBuf::upload(&offsets)?,
            n_sources,
        })
    }
}

// ---------------------------------------------------------------------------
// Batch PSO result
// ---------------------------------------------------------------------------

/// Result of GPU batch PSO for one source.
#[derive(Debug, Clone)]
pub struct BatchPsoResult {
    pub params: Vec<f64>,
    pub cost: f64,
}

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

        // PSO hyperparameters
        let w_inertia = 0.7;
        let c1 = 1.5;
        let c2 = 1.5;

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
                    velocities[base + d] = (upper[d] - lower[d]) * 0.1 * (2.0 * rng.random::<f64>() - 1.0);
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

        for _iter in 0..max_iters {
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

            // Update velocities and positions (CPU — lightweight)
            for s in 0..n_sources {
                if source_done[s] { continue; }
                for p in 0..n_particles {
                    let idx = s * n_particles + p;
                    let base = idx * dim;
                    let gb = s * dim;
                    for d in 0..dim {
                        let r1: f64 = rng.random();
                        let r2: f64 = rng.random();
                        velocities[base + d] = w_inertia * velocities[base + d]
                            + c1 * r1 * (pbest_pos[base + d] - positions[base + d])
                            + c2 * r2 * (gbest_pos[gb + d] - positions[base + d]);
                        positions[base + d] = (positions[base + d] + velocities[base + d])
                            .clamp(lower[d], upper[d]);
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
}
