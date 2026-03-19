//! GPU-accelerated batch model evaluation and fitting via CUDA.
//!
//! Capabilities:
//!   1. `eval_batch` — forward model evaluation for many draws/sources
//!   2. `batch_pso` — run PSO model fitting for many sources simultaneously
//!   3. `batch_model_select` — adaptive multi-model PSO with CUDA streams
//!   4. `batch_pso_multi_bazin` — GPU-resident MultiBazin PSO

use std::ffi::c_int;
use std::ptr;


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

    /// Convert to the parametric module's model name type.
    pub fn to_svi_name(self) -> crate::parametric::SviModelName {
        match self {
            GpuModelName::Bazin => crate::parametric::SviModelName::Bazin,
            GpuModelName::Villar => crate::parametric::SviModelName::Villar,
            GpuModelName::Tde => crate::parametric::SviModelName::Tde,
            GpuModelName::Arnett => crate::parametric::SviModelName::Arnett,
            GpuModelName::Magnetar => crate::parametric::SviModelName::Magnetar,
            GpuModelName::ShockCooling => crate::parametric::SviModelName::ShockCooling,
            GpuModelName::Afterglow => crate::parametric::SviModelName::Afterglow,
            GpuModelName::MetzgerKN => crate::parametric::SviModelName::MetzgerKN,
        }
    }

    /// Index of the t0 parameter in the parameter vector (matches SviModel::t0_idx).
    pub fn t0_idx(self) -> usize {
        match self {
            GpuModelName::Bazin => 2,
            GpuModelName::Villar => 3,
            GpuModelName::Tde => 2,
            GpuModelName::Arnett => 1,
            GpuModelName::Magnetar => 1,
            GpuModelName::ShockCooling => 1,
            GpuModelName::Afterglow => 1,
            GpuModelName::MetzgerKN => 3,
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

/// Threshold: use CUDA streams when n_sources <= this value.
const STREAM_THRESHOLD: usize = 32;

// ---------------------------------------------------------------------------
// CUDA runtime FFI
// ---------------------------------------------------------------------------

type CudaResult = c_int;
type CudaStream = *mut std::ffi::c_void;

extern "C" {
    fn cudaSetDevice(device: c_int) -> CudaResult;
    fn cudaMalloc(devPtr: *mut *mut u8, size: usize) -> CudaResult;
    fn cudaFree(devPtr: *mut u8) -> CudaResult;
    fn cudaMemcpy(dst: *mut u8, src: *const u8, count: usize, kind: c_int) -> CudaResult;
    fn cudaDeviceSynchronize() -> CudaResult;
    fn cudaGetLastError() -> CudaResult;
    fn cudaGetErrorString(error: CudaResult) -> *const i8;
    // Streams
    fn cudaStreamCreate(pStream: *mut CudaStream) -> CudaResult;
    fn cudaStreamDestroy(stream: CudaStream) -> CudaResult;
    fn cudaStreamSynchronize(stream: CudaStream) -> CudaResult;
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
#[allow(dead_code)]
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
        prior_centers: *const f64,
        prior_widths: *const f64,
        n_sources: c_int,
        n_particles: c_int,
        n_params: c_int,
        model_id: c_int,
        grid: c_int,
        block: c_int,
    );

    // Separate standard (non-KN) kernel
    fn launch_batch_pso_full_std(
        all_times: *const f64,
        all_flux: *const f64,
        all_obs_var: *const f64,
        all_is_upper: *const c_int,
        all_upper_flux: *const f64,
        source_offsets: *const c_int,
        bounds_lower: *const f64,
        bounds_upper: *const f64,
        prior_centers: *const f64,
        prior_widths: *const f64,
        out_gbest_pos: *mut f64,
        out_gbest_cost: *mut f64,
        n_sources: c_int,
        n_particles: c_int,
        n_params: c_int,
        model_id: c_int,
        max_iters: c_int,
        stall_iters: c_int,
        base_seed: u64,
        max_obs: c_int,
        per_source_t0_lower: *const f64,
        per_source_t0_upper: *const f64,
        t0_idx: c_int,
    );

    // Separate KN kernel (no model_id param)
    fn launch_batch_pso_full_kn(
        all_times: *const f64,
        all_flux: *const f64,
        all_obs_var: *const f64,
        all_is_upper: *const c_int,
        all_upper_flux: *const f64,
        source_offsets: *const c_int,
        bounds_lower: *const f64,
        bounds_upper: *const f64,
        prior_centers: *const f64,
        prior_widths: *const f64,
        out_gbest_pos: *mut f64,
        out_gbest_cost: *mut f64,
        n_sources: c_int,
        n_particles: c_int,
        n_params: c_int,
        max_iters: c_int,
        stall_iters: c_int,
        base_seed: u64,
        max_obs: c_int,
        per_source_t0_lower: *const f64,
        per_source_t0_upper: *const f64,
        t0_idx: c_int,
    );

    // Stream-aware launch wrappers
    fn launch_batch_pso_full_std_stream(
        all_times: *const f64,
        all_flux: *const f64,
        all_obs_var: *const f64,
        all_is_upper: *const c_int,
        all_upper_flux: *const f64,
        source_offsets: *const c_int,
        bounds_lower: *const f64,
        bounds_upper: *const f64,
        prior_centers: *const f64,
        prior_widths: *const f64,
        out_gbest_pos: *mut f64,
        out_gbest_cost: *mut f64,
        n_sources: c_int,
        n_particles: c_int,
        n_params: c_int,
        model_id: c_int,
        max_iters: c_int,
        stall_iters: c_int,
        base_seed: u64,
        max_obs: c_int,
        stream: CudaStream,
        per_source_t0_lower: *const f64,
        per_source_t0_upper: *const f64,
        t0_idx: c_int,
    );

    fn launch_batch_pso_full_kn_stream(
        all_times: *const f64,
        all_flux: *const f64,
        all_obs_var: *const f64,
        all_is_upper: *const c_int,
        all_upper_flux: *const f64,
        source_offsets: *const c_int,
        bounds_lower: *const f64,
        bounds_upper: *const f64,
        prior_centers: *const f64,
        prior_widths: *const f64,
        out_gbest_pos: *mut f64,
        out_gbest_cost: *mut f64,
        n_sources: c_int,
        n_particles: c_int,
        n_params: c_int,
        max_iters: c_int,
        stall_iters: c_int,
        base_seed: u64,
        max_obs: c_int,
        stream: CudaStream,
        per_source_t0_lower: *const f64,
        per_source_t0_upper: *const f64,
        t0_idx: c_int,
    );

    // GPU-resident MultiBazin
    fn launch_batch_pso_full_multi_bazin(
        all_times: *const f64,
        all_flux: *const f64,
        all_obs_var: *const f64,
        all_is_upper: *const c_int,
        all_upper_flux: *const f64,
        source_offsets: *const c_int,
        global_t_min: f64,
        global_t_max: f64,
        out_best_k: *mut c_int,
        out_best_params: *mut f64,
        out_best_cost: *mut f64,
        out_best_bic: *mut f64,
        out_per_k_cost: *mut f64,
        out_per_k_bic: *mut f64,
        n_sources: c_int,
        n_particles: c_int,
        max_iters: c_int,
        stall_iters: c_int,
        base_seed: u64,
        max_obs: c_int,
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
        max_obs: c_int,
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

    #[allow(dead_code)]
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
// CUDA Stream wrapper
// ---------------------------------------------------------------------------

struct GpuStream {
    raw: CudaStream,
}

impl GpuStream {
    fn new() -> Result<Self, String> {
        let mut raw: CudaStream = ptr::null_mut();
        cuda_check(unsafe { cudaStreamCreate(&mut raw) })?;
        Ok(Self { raw })
    }

    fn sync(&self) -> Result<(), String> {
        cuda_check(unsafe { cudaStreamSynchronize(self.raw) })
    }
}

impl Drop for GpuStream {
    fn drop(&mut self) {
        unsafe { cudaStreamDestroy(self.raw); }
    }
}

// ---------------------------------------------------------------------------
// Source data for batch operations
// ---------------------------------------------------------------------------

/// Pre-normalized observation data for one source.
#[derive(Clone)]
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
    /// Number of observations per source (host copy of offsets diffs).
    h_n_obs: Vec<usize>,
}

impl GpuBatchData {
    /// Upload packed source data to the GPU.
    pub fn new(sources: &[BatchSource]) -> Result<Self, String> {
        let n_sources = sources.len();

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

    /// Maximum number of observations across all sources.
    fn max_obs(&self) -> usize {
        self.h_n_obs.iter().copied().max().unwrap_or(0)
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
    /// `sources` provides the raw observation data used to compute per-source
    /// data-informed t0 bounds (matching the CPU path logic).
    pub fn batch_pso(
        &self,
        model: GpuModelName,
        data: &GpuBatchData,
        sources: &[BatchSource],
        n_particles: usize,
        max_iters: usize,
        stall_iters: usize,
        seed: u64,
    ) -> Result<Vec<BatchPsoResult>, String> {
        let n_sources = data.n_sources;
        let n_params = model.n_params();
        let (lower, upper) = model.pso_bounds();
        let max_obs = data.max_obs();
        let t0_idx = model.t0_idx();

        // Compute per-source t0 bounds matching CPU pso_fit_single_model_t0 logic
        let mut t0_lo_vec = Vec::with_capacity(n_sources);
        let mut t0_hi_vec = Vec::with_capacity(n_sources);
        for src in sources {
            let n_obs = src.times.len();
            if n_obs > 0 {
                let t_first = src.times.iter().cloned().fold(f64::INFINITY, f64::min);
                let mut peak_time = src.times[0];
                let mut peak_flux = f64::NEG_INFINITY;
                for i in 0..n_obs {
                    if !src.is_upper[i] && src.flux[i] > peak_flux {
                        peak_flux = src.flux[i];
                        peak_time = src.times[i];
                    }
                }
                let t0_lo = (t_first - 30.0).max(lower[t0_idx]);
                let t0_hi = (peak_time + 10.0).min(upper[t0_idx]);
                if t0_lo < t0_hi {
                    t0_lo_vec.push(t0_lo);
                    t0_hi_vec.push(t0_hi);
                } else {
                    t0_lo_vec.push(lower[t0_idx]);
                    t0_hi_vec.push(upper[t0_idx]);
                }
            } else {
                t0_lo_vec.push(lower[t0_idx]);
                t0_hi_vec.push(upper[t0_idx]);
            }
        }

        let d_lower = DevBuf::upload(&lower)?;
        let d_upper = DevBuf::upload(&upper)?;
        let d_t0_lo = DevBuf::upload(&t0_lo_vec)?;
        let d_t0_hi = DevBuf::upload(&t0_hi_vec)?;

        let pop_priors = crate::parametric::population_priors_for_gpu(model);
        let (d_prior_centers, d_prior_widths) = if !pop_priors.is_empty() {
            let centers: Vec<f64> = pop_priors.iter().map(|&(c, _)| c).collect();
            let widths: Vec<f64> = pop_priors.iter().map(|&(_, w)| w).collect();
            (Some(DevBuf::upload(&centers)?), Some(DevBuf::upload(&widths)?))
        } else {
            (None, None)
        };

        let d_gbest_pos = DevBuf::alloc(n_sources * n_params * size_of::<f64>())?;
        let d_gbest_cost = DevBuf::alloc(n_sources * size_of::<f64>())?;

        let pc_ptr = d_prior_centers.as_ref().map_or(ptr::null(), |b| b.ptr as *const f64);
        let pw_ptr = d_prior_widths.as_ref().map_or(ptr::null(), |b| b.ptr as *const f64);

        unsafe {
            if model == GpuModelName::MetzgerKN {
                launch_batch_pso_full_kn(
                    data.d_times.ptr as _,
                    data.d_flux.ptr as _,
                    data.d_obs_var.ptr as _,
                    data.d_is_upper.ptr as _,
                    data.d_upper_flux.ptr as _,
                    data.d_offsets.ptr as _,
                    d_lower.ptr as _,
                    d_upper.ptr as _,
                    pc_ptr,
                    pw_ptr,
                    d_gbest_pos.ptr as _,
                    d_gbest_cost.ptr as _,
                    n_sources as c_int,
                    n_particles as c_int,
                    n_params as c_int,
                    max_iters as c_int,
                    stall_iters as c_int,
                    seed,
                    max_obs as c_int,
                    d_t0_lo.ptr as _,
                    d_t0_hi.ptr as _,
                    t0_idx as c_int,
                );
            } else {
                launch_batch_pso_full_std(
                    data.d_times.ptr as _,
                    data.d_flux.ptr as _,
                    data.d_obs_var.ptr as _,
                    data.d_is_upper.ptr as _,
                    data.d_upper_flux.ptr as _,
                    data.d_offsets.ptr as _,
                    d_lower.ptr as _,
                    d_upper.ptr as _,
                    pc_ptr,
                    pw_ptr,
                    d_gbest_pos.ptr as _,
                    d_gbest_cost.ptr as _,
                    n_sources as c_int,
                    n_particles as c_int,
                    n_params as c_int,
                    model.model_id(),
                    max_iters as c_int,
                    stall_iters as c_int,
                    seed,
                    max_obs as c_int,
                    d_t0_lo.ptr as _,
                    d_t0_hi.ptr as _,
                    t0_idx as c_int,
                );
            }
            cuda_check(cudaGetLastError())?;
            cuda_check(cudaDeviceSynchronize())?;
        }

        let mut h_gbest_pos = vec![0.0f64; n_sources * n_params];
        let mut h_gbest_cost = vec![0.0f64; n_sources];
        d_gbest_pos.download_into(&mut h_gbest_pos)?;
        d_gbest_cost.download_into(&mut h_gbest_cost)?;

        let results: Vec<BatchPsoResult> = (0..n_sources)
            .map(|s| {
                let gb = s * n_params;
                BatchPsoResult {
                    params: h_gbest_pos[gb..gb + n_params].to_vec(),
                    cost: h_gbest_cost[s],
                }
            })
            .collect();

        Ok(results)
    }

    /// Launch PSO for a model on a specific CUDA stream
    /// Does NOT synchronize — caller must sync the stream.
    /// Returns device buffers for gbest_pos and gbest_cost.
    fn batch_pso_on_stream(
        &self,
        model: GpuModelName,
        data: &GpuBatchData,
        n_particles: usize,
        max_iters: usize,
        stall_iters: usize,
        seed: u64,
        stream: &GpuStream,
        d_t0_lo: &DevBuf,
        d_t0_hi: &DevBuf,
        t0_idx: usize,
    ) -> Result<(DevBuf, DevBuf, Vec<f64>, Vec<f64>), String> {
        let n_sources = data.n_sources;
        let n_params = model.n_params();
        let (lower, upper) = model.pso_bounds();
        let max_obs = data.max_obs();

        let d_lower = DevBuf::upload(&lower)?;
        let d_upper = DevBuf::upload(&upper)?;

        let pop_priors = crate::parametric::population_priors_for_gpu(model);
        let (d_prior_centers, d_prior_widths) = if !pop_priors.is_empty() {
            let centers: Vec<f64> = pop_priors.iter().map(|&(c, _)| c).collect();
            let widths: Vec<f64> = pop_priors.iter().map(|&(_, w)| w).collect();
            (Some(DevBuf::upload(&centers)?), Some(DevBuf::upload(&widths)?))
        } else {
            (None, None)
        };

        let d_gbest_pos = DevBuf::alloc(n_sources * n_params * size_of::<f64>())?;
        let d_gbest_cost = DevBuf::alloc(n_sources * size_of::<f64>())?;

        let pc_ptr = d_prior_centers.as_ref().map_or(ptr::null(), |b| b.ptr as *const f64);
        let pw_ptr = d_prior_widths.as_ref().map_or(ptr::null(), |b| b.ptr as *const f64);

        unsafe {
            if model == GpuModelName::MetzgerKN {
                launch_batch_pso_full_kn_stream(
                    data.d_times.ptr as _, data.d_flux.ptr as _,
                    data.d_obs_var.ptr as _, data.d_is_upper.ptr as _,
                    data.d_upper_flux.ptr as _, data.d_offsets.ptr as _,
                    d_lower.ptr as _, d_upper.ptr as _,
                    pc_ptr, pw_ptr,
                    d_gbest_pos.ptr as _, d_gbest_cost.ptr as _,
                    n_sources as c_int, n_particles as c_int, n_params as c_int,
                    max_iters as c_int, stall_iters as c_int, seed,
                    max_obs as c_int, stream.raw,
                    d_t0_lo.ptr as _, d_t0_hi.ptr as _,
                    t0_idx as c_int,
                );
            } else {
                launch_batch_pso_full_std_stream(
                    data.d_times.ptr as _, data.d_flux.ptr as _,
                    data.d_obs_var.ptr as _, data.d_is_upper.ptr as _,
                    data.d_upper_flux.ptr as _, data.d_offsets.ptr as _,
                    d_lower.ptr as _, d_upper.ptr as _,
                    pc_ptr, pw_ptr,
                    d_gbest_pos.ptr as _, d_gbest_cost.ptr as _,
                    n_sources as c_int, n_particles as c_int, n_params as c_int,
                    model.model_id(), max_iters as c_int, stall_iters as c_int, seed,
                    max_obs as c_int, stream.raw,
                    d_t0_lo.ptr as _, d_t0_hi.ptr as _,
                    t0_idx as c_int,
                );
            }
            cuda_check(cudaGetLastError())?;
        }

        Ok((d_gbest_pos, d_gbest_cost, lower, upper))
    }

    /// Download PSO results from device buffers after stream sync.
    fn download_pso_results(
        d_gbest_pos: &DevBuf,
        d_gbest_cost: &DevBuf,
        n_sources: usize,
        n_params: usize,
    ) -> Result<Vec<BatchPsoResult>, String> {
        let mut h_gbest_pos = vec![0.0f64; n_sources * n_params];
        let mut h_gbest_cost = vec![0.0f64; n_sources];
        d_gbest_pos.download_into(&mut h_gbest_pos)?;
        d_gbest_cost.download_into(&mut h_gbest_cost)?;

        Ok((0..n_sources)
            .map(|s| {
                let gb = s * n_params;
                BatchPsoResult {
                    params: h_gbest_pos[gb..gb + n_params].to_vec(),
                    cost: h_gbest_cost[s],
                }
            })
            .collect())
    }

    /// Run PSO model selection across all GPU-supported models for many sources.
    ///
    /// When n_sources <= 32 (GPU underutilized), launches models
    /// on separate CUDA streams for overlapping execution.
    /// When n_sources > 32, runs sequentially (GPU already saturated).
    pub fn batch_model_select(
        &self,
        data: &GpuBatchData,
        sources: &[BatchSource],
        n_particles: usize,
        max_iters: usize,
        stall_iters: usize,
        bazin_good_enough: f64,
    ) -> Result<Vec<(GpuModelName, BatchPsoResult)>, String> {
        let n_sources = data.n_sources;

        let mut best_model = vec![GpuModelName::Bazin; n_sources];
        let mut best_result: Vec<BatchPsoResult> = (0..n_sources)
            .map(|_| BatchPsoResult { params: vec![], cost: f64::INFINITY })
            .collect();

        // Run Bazin first (early-stop gate)
        let bazin_results = self.batch_pso(
            GpuModelName::Bazin, data, sources, n_particles, max_iters, stall_iters, 42,
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

        if !needs_more.iter().any(|&b| b) {
            return Ok(best_model.into_iter().zip(best_result).collect());
        }

        let remaining_models = &ALL_GPU_MODELS[1..];

        // Pre-compute per-source t0 bounds (shared across models, recomputed per-model t0_idx)
        if n_sources <= STREAM_THRESHOLD {
            // Small batch —> launch remaining models on separate streams
            let mut stream_data: Vec<(
                GpuModelName,
                GpuStream,
                DevBuf,  // d_gbest_pos
                DevBuf,  // d_gbest_cost
            )> = Vec::new();

            for &model in remaining_models {
                let stream = GpuStream::new()?;
                let (d_t0_lo, d_t0_hi) = Self::compute_t0_bounds_bufs(model, sources)?;
                let (d_pos, d_cost, _lower, _upper) = self.batch_pso_on_stream(
                    model, data, n_particles, max_iters, stall_iters, 42, &stream,
                    &d_t0_lo, &d_t0_hi, model.t0_idx(),
                )?;
                stream_data.push((model, stream, d_pos, d_cost));
            }

            // Sync all streams and collect results
            for (model, stream, d_pos, d_cost) in &stream_data {
                stream.sync()?;
                let n_params = model.n_params();
                let results = Self::download_pso_results(d_pos, d_cost, n_sources, n_params)?;
                for (s, r) in results.into_iter().enumerate() {
                    if needs_more[s] && r.cost < best_result[s].cost {
                        best_model[s] = *model;
                        best_result[s] = r;
                    }
                }
            }
        } else {
            // Large batch, sequential dispatch
            for &model in remaining_models {
                let results = self.batch_pso(
                    model, data, sources, n_particles, max_iters, stall_iters, 42,
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

    /// Run PSO for every model on all sources, returning all results.
    ///
    /// Uses CUDA streams for small batches.
    pub fn batch_all_models(
        &self,
        data: &GpuBatchData,
        sources: &[BatchSource],
        n_particles: usize,
        max_iters: usize,
        stall_iters: usize,
    ) -> Result<Vec<(GpuModelName, Vec<BatchPsoResult>)>, String> {
        let n_sources = data.n_sources;

        if n_sources <= STREAM_THRESHOLD {
            // Small batch, launch all models on separate streams
            let mut stream_data: Vec<(
                GpuModelName,
                GpuStream,
                DevBuf,
                DevBuf,
            )> = Vec::new();

            for &model in ALL_GPU_MODELS {
                let stream = GpuStream::new()?;
                let (d_t0_lo, d_t0_hi) = Self::compute_t0_bounds_bufs(model, sources)?;
                let (d_pos, d_cost, _lower, _upper) = self.batch_pso_on_stream(
                    model, data, n_particles, max_iters, stall_iters, 42, &stream,
                    &d_t0_lo, &d_t0_hi, model.t0_idx(),
                )?;
                stream_data.push((model, stream, d_pos, d_cost));
            }

            let mut all = Vec::with_capacity(ALL_GPU_MODELS.len());
            for (model, stream, d_pos, d_cost) in &stream_data {
                stream.sync()?;
                let n_params = model.n_params();
                let results = Self::download_pso_results(d_pos, d_cost, n_sources, n_params)?;
                all.push((*model, results));
            }
            Ok(all)
        } else {
            // Large batch, sequential
            let mut all = Vec::with_capacity(ALL_GPU_MODELS.len());
            for &model in ALL_GPU_MODELS {
                let results = self.batch_pso(model, data, sources, n_particles, max_iters, stall_iters, 42)?;
                all.push((model, results));
            }
            Ok(all)
        }
    }

    /// Compute per-source t0 bounds and upload them to GPU device buffers.
    fn compute_t0_bounds_bufs(
        model: GpuModelName,
        sources: &[BatchSource],
    ) -> Result<(DevBuf, DevBuf), String> {
        let t0_idx = model.t0_idx();
        let (lower, upper) = model.pso_bounds();
        let mut t0_lo_vec = Vec::with_capacity(sources.len());
        let mut t0_hi_vec = Vec::with_capacity(sources.len());
        for src in sources {
            let n_obs = src.times.len();
            if n_obs > 0 {
                let t_first = src.times.iter().cloned().fold(f64::INFINITY, f64::min);
                let mut peak_time = src.times[0];
                let mut peak_flux = f64::NEG_INFINITY;
                for i in 0..n_obs {
                    if !src.is_upper[i] && src.flux[i] > peak_flux {
                        peak_flux = src.flux[i];
                        peak_time = src.times[i];
                    }
                }
                let t0_lo = (t_first - 30.0).max(lower[t0_idx]);
                let t0_hi = (peak_time + 10.0).min(upper[t0_idx]);
                if t0_lo < t0_hi {
                    t0_lo_vec.push(t0_lo);
                    t0_hi_vec.push(t0_hi);
                } else {
                    t0_lo_vec.push(lower[t0_idx]);
                    t0_hi_vec.push(upper[t0_idx]);
                }
            } else {
                t0_lo_vec.push(lower[t0_idx]);
                t0_hi_vec.push(upper[t0_idx]);
            }
        }
        Ok((DevBuf::upload(&t0_lo_vec)?, DevBuf::upload(&t0_hi_vec)?))
    }

    // -----------------------------------------------------------------------
    // Batch MultiBazin PSO (GPU-resident)
    // -----------------------------------------------------------------------

    /// Run greedy MultiBazin fitting (K=1..4) across many sources on the GPU.
    ///
    /// Entire K=1..4 loop runs on GPU — single kernel launch,
    /// eliminates ~200 PCIe round-trips from the old CPU-loop approach.
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
        const MB_MAX_PARAMS: usize = 18; // 4*4+2

        let n_sources = data.n_sources;
        assert_eq!(n_sources, sources.len());

        // Compute global time range for bounds
        let t_ranges: Vec<(f64, f64)> = sources
            .iter()
            .map(|s| {
                let tmin = s.times.iter().cloned().fold(f64::INFINITY, f64::min);
                let tmax = s.times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                (tmin, tmax)
            })
            .collect();

        let global_t_min = t_ranges.iter().map(|r| r.0).fold(f64::INFINITY, f64::min) - 30.0;
        let global_t_max = t_ranges.iter().map(|r| r.1).fold(f64::NEG_INFINITY, f64::max) + 30.0;

        let max_obs = data.max_obs();

        // Allocate output buffers
        let d_best_k = DevBuf::alloc(n_sources * size_of::<c_int>())?;
        let d_best_params = DevBuf::alloc(n_sources * MB_MAX_PARAMS * size_of::<f64>())?;
        let d_best_cost = DevBuf::alloc(n_sources * size_of::<f64>())?;
        let d_best_bic = DevBuf::alloc(n_sources * size_of::<f64>())?;
        let d_per_k_cost = DevBuf::alloc(n_sources * MAX_K * size_of::<f64>())?;
        let d_per_k_bic = DevBuf::alloc(n_sources * MAX_K * size_of::<f64>())?;

        unsafe {
            launch_batch_pso_full_multi_bazin(
                data.d_times.ptr as _,
                data.d_flux.ptr as _,
                data.d_obs_var.ptr as _,
                data.d_is_upper.ptr as _,
                data.d_upper_flux.ptr as _,
                data.d_offsets.ptr as _,
                global_t_min,
                global_t_max,
                d_best_k.ptr as _,
                d_best_params.ptr as _,
                d_best_cost.ptr as _,
                d_best_bic.ptr as _,
                d_per_k_cost.ptr as _,
                d_per_k_bic.ptr as _,
                n_sources as c_int,
                n_particles as c_int,
                max_iters as c_int,
                stall_iters as c_int,
                seed,
                max_obs as c_int,
            );
            cuda_check(cudaGetLastError())?;
            cuda_check(cudaDeviceSynchronize())?;
        }

        // Download results
        let mut h_best_k = vec![0i32; n_sources];
        let mut h_best_params = vec![0.0f64; n_sources * MB_MAX_PARAMS];
        let mut h_best_cost = vec![0.0f64; n_sources];
        let mut h_best_bic = vec![0.0f64; n_sources];
        let mut h_per_k_cost = vec![0.0f64; n_sources * MAX_K];
        let mut h_per_k_bic = vec![0.0f64; n_sources * MAX_K];

        d_best_k.download_into(&mut h_best_k)?;
        d_best_params.download_into(&mut h_best_params)?;
        d_best_cost.download_into(&mut h_best_cost)?;
        d_best_bic.download_into(&mut h_best_bic)?;
        d_per_k_cost.download_into(&mut h_per_k_cost)?;
        d_per_k_bic.download_into(&mut h_per_k_bic)?;

        let results: Vec<MultiBazinResult> = (0..n_sources)
            .map(|s| {
                let bk = h_best_k[s] as usize;
                let n_params = 4 * bk + 2;
                let base = s * MB_MAX_PARAMS;
                MultiBazinResult {
                    best_k: bk,
                    params: h_best_params[base..base + n_params].to_vec(),
                    cost: h_best_cost[s],
                    bic: h_best_bic[s],
                    per_k_cost: h_per_k_cost[s * MAX_K..(s + 1) * MAX_K].to_vec(),
                    per_k_bic: h_per_k_bic[s * MAX_K..(s + 1) * MAX_K].to_vec(),
                }
            })
            .collect();

        Ok(results)
    }

    // -----------------------------------------------------------------------
    // Batch GP fitting
    // -----------------------------------------------------------------------

    /// Fit DenseGPs for many bands in parallel on the GPU.
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
        let gp_state_size = 54;

        let d_times = DevBuf::upload(&all_times)?;
        let d_mags = DevBuf::upload(&all_mags)?;
        let d_noise_var = DevBuf::upload(&all_noise_var)?;
        let d_offsets = DevBuf::upload(&offsets)?;
        let d_query = DevBuf::upload(query_times)?;
        let d_amps = DevBuf::upload(amp_candidates)?;
        let d_ls = DevBuf::upload(ls_candidates)?;

        let d_gp_state = DevBuf::alloc(n_bands * gp_state_size * size_of::<f64>())?;
        let d_pred_grid = DevBuf::alloc(n_bands * n_pred * size_of::<f64>())?;
        let d_std_grid = DevBuf::alloc(n_bands * n_pred * size_of::<f64>())?;

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

        let mut h_gp_state = vec![0.0f64; n_bands * gp_state_size];
        let mut h_pred_grid = vec![0.0f64; n_bands * n_pred];
        let mut h_std_grid = vec![0.0f64; n_bands * n_pred];
        let mut h_pred_obs = vec![0.0f64; total_obs];

        d_gp_state.download_into(&mut h_gp_state)?;
        d_pred_grid.download_into(&mut h_pred_grid)?;
        d_std_grid.download_into(&mut h_std_grid)?;
        d_pred_obs.download_into(&mut h_pred_obs)?;

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

            let dense_gp = if m > 0 && m <= 25 {
                let x_train = state[25..25 + m].to_vec();
                let amp = state[50];
                let inv_2ls2 = state[51];

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

        let gp2d_state_size = 125;

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

        let mut h_gp_state = vec![0.0f64; n_sources * gp2d_state_size];
        let mut h_pred_grid = vec![0.0f64; n_sources * n_pred];
        let mut h_std_grid = vec![0.0f64; n_sources * n_pred];

        d_gp_state.download_into(&mut h_gp_state)?;
        d_pred_grid.download_into(&mut h_pred_grid)?;
        d_std_grid.download_into(&mut h_std_grid)?;

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

                let n_train = sources[s].times.len();
                let train_rms = {
                    let alpha = &state[0..m];
                    let x_t = &state[40..40 + m];
                    let x_w = &state[80..80 + m];
                    let y_mean = state[123];
                    let mut rms = 0.0;
                    let n_eval = n_train.min(100);
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

        let max_params: usize = 7;

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

        let d_pso = DevBuf::upload(&h_pso_params)?;
        let d_model_ids = DevBuf::upload(&h_model_ids)?;
        let d_n_params = DevBuf::upload(&h_n_params)?;
        let d_se_idx = DevBuf::upload(&h_se_idx)?;
        let d_prior_centers = DevBuf::upload(&h_prior_centers)?;
        let d_prior_widths = DevBuf::upload(&h_prior_widths)?;

        let d_out_mu = DevBuf::alloc(n_sources * max_params * size_of::<f64>())?;
        let d_out_ls = DevBuf::alloc(n_sources * max_params * size_of::<f64>())?;
        let d_out_elbo = DevBuf::alloc(n_sources * size_of::<f64>())?;

        let max_obs: c_int = data.max_obs() as c_int;

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
                max_obs,
            );
            cuda_check(cudaGetLastError())?;
            cuda_check(cudaDeviceSynchronize())?;
        }

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

/// Input for GPU SVI fit for one source.
pub struct SviBatchInput {
    pub model_id: usize,
    pub pso_params: Vec<f64>,
    pub se_idx: usize,
    pub prior_centers: Vec<f64>,
    pub prior_widths: Vec<f64>,
}

/// Output from GPU SVI fit for one source.
#[derive(Clone, Debug)]
pub struct SviBatchOutput {
    pub mu: Vec<f64>,
    pub log_sigma: Vec<f64>,
    pub elbo: f64,
}

/// Input data for one source's 2D GP fit.
pub struct Gp2dInput {
    pub times: Vec<f64>,
    pub waves: Vec<f64>,
    pub mags: Vec<f64>,
    pub noise_var: Vec<f64>,
}

/// Output from batch 2D GP fitting for one source.
pub struct Gp2dOutput {
    pub pred_grid: Vec<f64>,
    pub std_grid: Vec<f64>,
    pub amp: f64,
    pub ls_time: f64,
    pub ls_wave: f64,
    pub train_rms: f64,
    pub n_train: usize,
    pub success: bool,
}

// ---------------------------------------------------------------------------
// Batch GP types
// ---------------------------------------------------------------------------

/// Input data for one band in batch GP fitting.
pub struct GpBandInput {
    pub times: Vec<f64>,
    pub mags: Vec<f64>,
    pub noise_var: Vec<f64>,
}

/// Output of GPU GP fitting for one band.
pub struct GpBandOutput {
    /// Reconstructed DenseGP for additional predictions (thermal reuse, etc.)
    pub dense_gp: Option<crate::sparse_gp::DenseGP>,
    /// Mean predictions at the shared grid points (50 points).
    pub pred_grid: Vec<f64>,
    /// Std predictions at the shared grid points.
    pub std_grid: Vec<f64>,
    /// Mean predictions at the band's observation points (for chi2).
    pub pred_at_obs: Vec<f64>,
}
