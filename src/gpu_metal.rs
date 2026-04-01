//! Metal backend for GPU-accelerated batch model evaluation and fitting.
//!
//! Uses Apple's Metal API via objc2-metal for compute shader dispatch.
//! All GPU computations use float32; data is converted f64↔f32 at boundaries.

use std::ffi::c_int;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::*;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use super::gpu_types::*;

// ---------------------------------------------------------------------------
// Metal buffer wrapper
// ---------------------------------------------------------------------------

struct MetalBuf {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Number of bytes.
    len: usize,
}

impl MetalBuf {
    /// Allocate an uninitialized float32 buffer of `count` elements.
    fn alloc_f32(device: &ProtocolObject<dyn MTLDevice>, count: usize) -> Result<Self, String> {
        let bytes = count * std::mem::size_of::<f32>();
        let buffer = device
            .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
            .ok_or("Failed to allocate Metal buffer")?;
        Ok(Self { buffer, len: bytes })
    }

    /// Allocate and upload f64 data as f32.
    fn from_f64_as_f32(
        device: &ProtocolObject<dyn MTLDevice>,
        data: &[f64],
    ) -> Result<Self, String> {
        let buf = Self::alloc_f32(device, data.len())?;
        let ptr = buf.buffer.contents().as_ptr() as *mut f32;
        for (i, &v) in data.iter().enumerate() {
            unsafe { ptr.add(i).write(v as f32) };
        }
        Ok(buf)
    }

    /// Allocate and upload i32 data directly.
    fn from_i32(
        device: &ProtocolObject<dyn MTLDevice>,
        data: &[i32],
    ) -> Result<Self, String> {
        let bytes = data.len() * std::mem::size_of::<i32>();
        let buffer = device
            .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
            .ok_or("Failed to allocate Metal buffer")?;
        let ptr = buffer.contents().as_ptr() as *mut i32;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
        Ok(Self { buffer, len: bytes })
    }

    /// Download f32 buffer contents into an f64 host slice.
    fn download_as_f64(&self, host: &mut [f64]) -> Result<(), String> {
        let ptr = self.buffer.contents().as_ptr() as *const f32;
        for (i, slot) in host.iter_mut().enumerate() {
            *slot = unsafe { ptr.add(i).read() } as f64;
        }
        Ok(())
    }

    /// Re-upload f64 data as f32 into an already-allocated buffer.
    fn upload_from_f64(&self, data: &[f64]) -> Result<(), String> {
        let ptr = self.buffer.contents().as_ptr() as *mut f32;
        for (i, &v) in data.iter().enumerate() {
            unsafe { ptr.add(i).write(v as f32) };
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// GpuContext (Metal)
// ---------------------------------------------------------------------------

/// A Metal context for GPU compute.
pub struct GpuContext {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
}

impl GpuContext {
    /// Create a new Metal GPU context.
    ///
    /// The `_device` parameter is ignored — Metal uses the system default device.
    pub fn new(_device: i32) -> Result<Self, String> {
        let device = {
            let ptr = unsafe { MTLCreateSystemDefaultDevice() };
            ptr.ok_or("No Metal device available")?
        };

        let queue = device
            .newCommandQueue()
            .ok_or("Failed to create Metal command queue")?;

        // Load the embedded metallib
        let metallib_bytes: &[u8] =
            include_bytes!(concat!(env!("OUT_DIR"), "/lightcurve.metallib"));

        let data = unsafe {
            objc2_foundation::NSData::dataWithBytes_length(
                metallib_bytes.as_ptr() as *const _,
                metallib_bytes.len(),
            )
        };

        let library = unsafe {
            device
                .newLibraryWithData_error(&data)
                .map_err(|e| format!("Failed to load metallib: {}", e))?
        };

        Ok(Self {
            device,
            queue,
            library,
        })
    }

    /// Get a compute pipeline state for a named kernel function.
    fn pipeline(&self, name: &str) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, String> {
        let ns_name = NSString::from_str(name);
        let function = self
            .library
            .newFunctionWithName(&ns_name)
            .ok_or_else(|| format!("Metal function '{}' not found in library", name))?;

        unsafe {
            self.device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|e| format!("Failed to create pipeline for '{}': {}", name, e))
        }
    }

    /// Create a command buffer, encode a compute pass, and return (encoder, buffer).
    fn begin_compute(
        &self,
    ) -> Result<
        (
            Retained<ProtocolObject<dyn MTLCommandBuffer>>,
            Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
        ),
        String,
    > {
        let cmd_buf = self
            .queue
            .commandBuffer()
            .ok_or("Failed to create command buffer")?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or("Failed to create compute encoder")?;
        Ok((cmd_buf, encoder))
    }

    /// Dispatch and wait.
    fn dispatch_and_wait(
        &self,
        cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        grid_size: usize,
        block_size: usize,
    ) {
        encoder.setComputePipelineState(pipeline);
        let threadgroups = MTLSize {
            width: grid_size as u64,
            height: 1,
            depth: 1,
        };
        let threads_per = MTLSize {
            width: block_size as u64,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per);
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
    }

    // -----------------------------------------------------------------------
    // eval_batch
    // -----------------------------------------------------------------------

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
                params.len(),
                n_draws,
                n_params
            ));
        }

        let kernel_name = match model {
            GpuModelName::Bazin => "bazin_eval",
            GpuModelName::Villar => "villar_eval",
            GpuModelName::Tde => "tde_eval",
            GpuModelName::Arnett => "arnett_eval",
            GpuModelName::Magnetar => "magnetar_eval",
            GpuModelName::ShockCooling => "shock_cooling_eval",
            GpuModelName::Afterglow => "afterglow_eval",
            GpuModelName::MetzgerKN => "metzger_kn_eval",
        };

        let pipeline = self.pipeline(kernel_name)?;

        let out_len = if model == GpuModelName::MetzgerKN {
            n_draws * n_times
        } else {
            n_draws * n_times
        };

        let d_params = MetalBuf::from_f64_as_f32(&self.device, params)?;
        let d_times = MetalBuf::from_f64_as_f32(&self.device, times)?;
        let d_out = MetalBuf::alloc_f32(&self.device, out_len)?;

        // Uniforms
        let n_draws_i = n_draws as i32;
        let n_times_i = n_times as i32;
        let n_params_i = n_params as i32;

        let (cmd_buf, encoder) = self.begin_compute()?;
        encoder.setComputePipelineState(&pipeline);
        encoder.setBuffer_offset_atIndex(Some(&d_params.buffer), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(&d_times.buffer), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&d_out.buffer), 0, 2);
        encoder.setBytes_length_atIndex(
            &n_draws_i as *const i32 as *const _,
            std::mem::size_of::<i32>(),
            3,
        );
        encoder.setBytes_length_atIndex(
            &n_times_i as *const i32 as *const _,
            std::mem::size_of::<i32>(),
            4,
        );
        encoder.setBytes_length_atIndex(
            &n_params_i as *const i32 as *const _,
            std::mem::size_of::<i32>(),
            5,
        );

        let total = if model == GpuModelName::MetzgerKN {
            n_draws
        } else {
            n_draws * n_times
        };
        let block: usize = 256;
        let grid = (total + block - 1) / block;

        let threadgroups = MTLSize { width: grid as u64, height: 1, depth: 1 };
        let threads_per = MTLSize { width: block as u64, height: 1, depth: 1 };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per);
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        let mut output = vec![0.0f64; out_len];
        d_out.download_as_f64(&mut output)?;
        Ok(output)
    }

    // -----------------------------------------------------------------------
    // batch_pso
    // -----------------------------------------------------------------------

    /// Run PSO for a single model across many sources simultaneously.
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

        let w_max = 0.9;
        let w_min = 0.4;
        let c1 = 1.5;
        let c2 = 1.5;
        let inv_max_iters = 1.0 / max_iters as f64;

        let v_max: Vec<f64> = (0..dim).map(|d| 0.5 * (upper[d] - lower[d])).collect();

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

        let pipeline = self.pipeline("batch_pso_cost")?;

        // GPU buffers for positions and costs (reused each iteration)
        let d_positions = MetalBuf::alloc_f32(&self.device, total_particles * dim)?;
        let d_costs = MetalBuf::alloc_f32(&self.device, total_particles)?;
        let mut h_costs = vec![0.0f64; total_particles];

        let model_id_i = model.model_id();
        let n_sources_i = n_sources as i32;
        let n_particles_i = n_particles as i32;
        let n_params_i = n_params as i32;

        let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(n_sources as u64 + 1));

        let mut prev_gbest = vec![f64::INFINITY; n_sources];
        let mut stall_count = vec![0usize; n_sources];
        let mut source_done = vec![false; n_sources];

        let block: usize = 256;

        for iter in 0..max_iters {
            let w = w_max - (w_max - w_min) * (iter as f64) * inv_max_iters;

            d_positions.upload_from_f64(&positions)?;

            let grid = (total_particles + block - 1) / block;
            let (cmd_buf, encoder) = self.begin_compute()?;
            encoder.setComputePipelineState(&pipeline);
            encoder.setBuffer_offset_atIndex(Some(&data.d_times.buffer), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&data.d_flux.buffer), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&data.d_obs_var.buffer), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&data.d_is_upper.buffer), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&data.d_upper_flux.buffer), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(&data.d_offsets.buffer), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(&d_positions.buffer), 0, 6);
            encoder.setBuffer_offset_atIndex(Some(&d_costs.buffer), 0, 7);
            encoder.setBytes_length_atIndex(&n_sources_i as *const _ as *const _, 4, 8);
            encoder.setBytes_length_atIndex(&n_particles_i as *const _ as *const _, 4, 9);
            encoder.setBytes_length_atIndex(&n_params_i as *const _ as *const _, 4, 10);
            encoder.setBytes_length_atIndex(&model_id_i as *const _ as *const _, 4, 11);

            let threadgroups = MTLSize { width: grid as u64, height: 1, depth: 1 };
            let threads_per = MTLSize { width: block as u64, height: 1, depth: 1 };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per);
            encoder.endEncoding();
            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();

            d_costs.download_as_f64(&mut h_costs)?;

            // Add population prior penalty (host-side)
            let pop_priors = crate::parametric::population_priors_for_gpu(model);
            if !pop_priors.is_empty() {
                for idx in 0..total_particles {
                    let s = idx / n_particles;
                    let base = idx * dim;
                    let n_obs = data.n_obs_per_source(s).max(1) as f64;
                    let mut neg_lp = 0.0;
                    for (j, &(center, width)) in pop_priors.iter().enumerate() {
                        if j < dim && width > 0.0 {
                            let z = (positions[base + j] - center) / width;
                            neg_lp += 0.5 * z * z;
                        }
                    }
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

            // Update velocities and positions
            for s in 0..n_sources {
                if source_done[s] { continue; }
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
                        v = v.clamp(-v_max[d], v_max[d]);
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
                if source_done[s] { continue; }
                let improved =
                    prev_gbest[s] - gbest_cost[s] > 0.01 * prev_gbest[s].abs().max(1e-10);
                if improved {
                    stall_count[s] = 0;
                    prev_gbest[s] = gbest_cost[s];
                } else {
                    stall_count[s] += 1;
                    if stall_count[s] >= stall_iters {
                        source_done[s] = true;
                    }
                }
                if !source_done[s] {
                    all_done = false;
                }
            }
            if all_done {
                break;
            }
        }

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

    // -----------------------------------------------------------------------
    // batch_model_select
    // -----------------------------------------------------------------------

    pub fn batch_model_select(
        &self,
        data: &GpuBatchData,
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

        let bazin_results =
            self.batch_pso(GpuModelName::Bazin, data, n_particles, max_iters, stall_iters, 42)?;
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

        if needs_more.iter().any(|&b| b) {
            for &model in &ALL_GPU_MODELS[1..] {
                let results =
                    self.batch_pso(model, data, n_particles, max_iters, stall_iters, 42)?;
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

    // -----------------------------------------------------------------------
    // batch_all_models
    // -----------------------------------------------------------------------

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
    // batch_pso_multi_bazin — multi-component Bazin with BIC model selection
    // -----------------------------------------------------------------------

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

        let pipeline = self.pipeline("batch_pso_cost_multi_bazin")?;

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
            let source_k: Vec<i32> = vec![k as i32; n_sources];

            // GPU buffers
            let d_positions = MetalBuf::alloc_f32(&self.device, total_particles * dim)?;
            let d_costs = MetalBuf::alloc_f32(&self.device, total_particles)?;
            let d_source_k = MetalBuf::from_i32(&self.device, &source_k)?;
            let mut h_costs = vec![0.0f64; total_particles];

            let n_sources_i = n_sources as i32;
            let n_particles_i = n_particles as i32;
            let n_params_i = n_params as i32;

            let block: usize = 256;

            let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(n_sources as u64 + k as u64));
            let mut prev_gbest = vec![f64::INFINITY; n_sources];
            let mut stall_count = vec![0usize; n_sources];
            let mut iter_done = vec![false; n_sources];

            // Velocity clamp for this K
            let v_max_k: Vec<f64> = (0..dim).map(|d| 0.5 * (upper[d] - lower[d])).collect();
            let inv_max_iters_k = 1.0 / max_iters as f64;

            for iter in 0..max_iters {
                let w = w_max_mb - (w_max_mb - w_min_mb) * (iter as f64) * inv_max_iters_k;

                d_positions.upload_from_f64(&positions)?;

                let grid = (total_particles + block - 1) / block;
                let (cmd_buf, encoder) = self.begin_compute()?;
                encoder.setComputePipelineState(&pipeline);
                encoder.setBuffer_offset_atIndex(Some(&data.d_times.buffer), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&data.d_flux.buffer), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&data.d_obs_var.buffer), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&data.d_is_upper.buffer), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(&data.d_upper_flux.buffer), 0, 4);
                encoder.setBuffer_offset_atIndex(Some(&data.d_offsets.buffer), 0, 5);
                encoder.setBuffer_offset_atIndex(Some(&d_positions.buffer), 0, 6);
                encoder.setBuffer_offset_atIndex(Some(&d_costs.buffer), 0, 7);
                encoder.setBuffer_offset_atIndex(Some(&d_source_k.buffer), 0, 8);
                encoder.setBytes_length_atIndex(&n_sources_i as *const _ as *const _, 4, 9);
                encoder.setBytes_length_atIndex(&n_particles_i as *const _ as *const _, 4, 10);
                encoder.setBytes_length_atIndex(&n_params_i as *const _ as *const _, 4, 11);

                let threadgroups = MTLSize { width: grid as u64, height: 1, depth: 1 };
                let threads_per = MTLSize { width: block as u64, height: 1, depth: 1 };
                encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per);
                encoder.endEncoding();
                cmd_buf.commit();
                cmd_buf.waitUntilCompleted();

                d_costs.download_as_f64(&mut h_costs)?;

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
    // batch_gp_fit
    // -----------------------------------------------------------------------

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

        // Pack band data
        let mut all_times = Vec::new();
        let mut all_mags = Vec::new();
        let mut all_noise_var = Vec::new();
        let mut offsets: Vec<i32> = Vec::with_capacity(n_bands + 1);
        let mut obs_to_band: Vec<i32> = Vec::new();
        offsets.push(0);

        for (b, band) in bands.iter().enumerate() {
            all_times.extend_from_slice(&band.times);
            all_mags.extend_from_slice(&band.mags);
            all_noise_var.extend_from_slice(&band.noise_var);
            obs_to_band.extend(std::iter::repeat(b as i32).take(band.times.len()));
            offsets.push(all_times.len() as i32);
        }

        let total_obs = all_times.len();
        let gp_state_size: usize = 54;

        // Upload
        let d_times = MetalBuf::from_f64_as_f32(&self.device, &all_times)?;
        let d_mags = MetalBuf::from_f64_as_f32(&self.device, &all_mags)?;
        let d_noise_var = MetalBuf::from_f64_as_f32(&self.device, &all_noise_var)?;
        let d_offsets = MetalBuf::from_i32(&self.device, &offsets)?;
        let d_query = MetalBuf::from_f64_as_f32(&self.device, query_times)?;
        let d_amps = MetalBuf::from_f64_as_f32(&self.device, amp_candidates)?;
        let d_ls = MetalBuf::from_f64_as_f32(&self.device, ls_candidates)?;

        // Outputs
        let d_gp_state = MetalBuf::alloc_f32(&self.device, n_bands * gp_state_size)?;
        let d_pred_grid = MetalBuf::alloc_f32(&self.device, n_bands * n_pred)?;
        let d_std_grid = MetalBuf::alloc_f32(&self.device, n_bands * n_pred)?;

        let n_hp_total = amp_candidates.len() * ls_candidates.len();
        let n_bands_i = n_bands as i32;
        let n_hp_amp_i = amp_candidates.len() as i32;
        let n_hp_ls_i = ls_candidates.len() as i32;
        let max_subsample_i = max_subsample as i32;

        // Kernel 1: fit + predict at grid
        let pipeline1 = self.pipeline("batch_gp_fit_predict")?;
        let (cmd_buf, encoder) = self.begin_compute()?;
        encoder.setComputePipelineState(&pipeline1);
        encoder.setBuffer_offset_atIndex(Some(&d_times.buffer), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(&d_mags.buffer), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&d_noise_var.buffer), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(&d_offsets.buffer), 0, 3);
        encoder.setBuffer_offset_atIndex(Some(&d_query.buffer), 0, 4);
        encoder.setBuffer_offset_atIndex(Some(&d_amps.buffer), 0, 5);
        encoder.setBuffer_offset_atIndex(Some(&d_ls.buffer), 0, 6);
        encoder.setBuffer_offset_atIndex(Some(&d_gp_state.buffer), 0, 7);
        encoder.setBuffer_offset_atIndex(Some(&d_pred_grid.buffer), 0, 8);
        encoder.setBuffer_offset_atIndex(Some(&d_std_grid.buffer), 0, 9);
        encoder.setBytes_length_atIndex(&n_bands_i as *const _ as *const _, 4, 10);
        encoder.setBytes_length_atIndex(&n_hp_amp_i as *const _ as *const _, 4, 11);
        encoder.setBytes_length_atIndex(&n_hp_ls_i as *const _ as *const _, 4, 12);
        encoder.setBytes_length_atIndex(&max_subsample_i as *const _ as *const _, 4, 13);

        let threadgroups = MTLSize { width: n_bands as u64, height: 1, depth: 1 };
        let threads_per = MTLSize { width: n_hp_total as u64, height: 1, depth: 1 };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per);
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        // Kernel 2: predict at observation points
        let d_obs_to_band = MetalBuf::from_i32(&self.device, &obs_to_band)?;
        let d_pred_obs = MetalBuf::alloc_f32(&self.device, total_obs)?;

        let pipeline2 = self.pipeline("batch_gp_predict_obs")?;
        let (cmd_buf2, encoder2) = self.begin_compute()?;
        encoder2.setComputePipelineState(&pipeline2);
        encoder2.setBuffer_offset_atIndex(Some(&d_gp_state.buffer), 0, 0);
        encoder2.setBuffer_offset_atIndex(Some(&d_times.buffer), 0, 1);
        encoder2.setBuffer_offset_atIndex(Some(&d_obs_to_band.buffer), 0, 2);
        encoder2.setBuffer_offset_atIndex(Some(&d_pred_obs.buffer), 0, 3);
        let total_obs_i = total_obs as i32;
        encoder2.setBytes_length_atIndex(&total_obs_i as *const _ as *const _, 4, 4);

        let block2: usize = 256;
        let grid2 = (total_obs + block2 - 1) / block2;
        let tg2 = MTLSize { width: grid2 as u64, height: 1, depth: 1 };
        let tp2 = MTLSize { width: block2 as u64, height: 1, depth: 1 };
        encoder2.dispatchThreadgroups_threadsPerThreadgroup(tg2, tp2);
        encoder2.endEncoding();
        cmd_buf2.commit();
        cmd_buf2.waitUntilCompleted();

        // Download
        let mut h_gp_state = vec![0.0f64; n_bands * gp_state_size];
        let mut h_pred_grid = vec![0.0f64; n_bands * n_pred];
        let mut h_std_grid = vec![0.0f64; n_bands * n_pred];
        let mut h_pred_obs = vec![0.0f64; total_obs];

        d_gp_state.download_as_f64(&mut h_gp_state)?;
        d_pred_grid.download_as_f64(&mut h_pred_grid)?;
        d_std_grid.download_as_f64(&mut h_std_grid)?;
        d_pred_obs.download_as_f64(&mut h_pred_obs)?;

        // Build output structs (same logic as CUDA)
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
                        (0..m)
                            .map(|i| {
                                let idx = (i as f64 * step + 0.5) as usize;
                                bands[b].noise_var[idx.min(n_obs - 1)]
                            })
                            .collect()
                    }
                };

                DenseGP::fit(
                    &x_train,
                    &{
                        let n_obs = bands[b].times.len();
                        if n_obs <= m {
                            bands[b].mags.clone()
                        } else {
                            let step = (n_obs - 1) as f64 / (m - 1) as f64;
                            (0..m)
                                .map(|i| {
                                    let idx = (i as f64 * step + 0.5) as usize;
                                    bands[b].mags[idx.min(n_obs - 1)]
                                })
                                .collect()
                        }
                    },
                    &sub_nv,
                    amp,
                    (0.5 / inv_2ls2).sqrt(),
                )
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

    // -----------------------------------------------------------------------
    // batch_gp_2d
    // -----------------------------------------------------------------------

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
        assert_eq!(n_pred, query_waves.len());

        let n_hp_total = amp_candidates.len() * lst_candidates.len() * lsw_candidates.len();
        assert!(n_hp_total <= 64, "max 64 hyperparameter combos for 2D GP");

        // Pack
        let mut all_times = Vec::new();
        let mut all_waves = Vec::new();
        let mut all_mags = Vec::new();
        let mut all_noise_var = Vec::new();
        let mut offsets: Vec<i32> = Vec::with_capacity(n_sources + 1);
        offsets.push(0);

        for src in sources {
            all_times.extend_from_slice(&src.times);
            all_waves.extend_from_slice(&src.waves);
            all_mags.extend_from_slice(&src.mags);
            all_noise_var.extend_from_slice(&src.noise_var);
            offsets.push(all_times.len() as i32);
        }

        let gp2d_state_size: usize = 125;

        // Upload
        let d_times = MetalBuf::from_f64_as_f32(&self.device, &all_times)?;
        let d_waves = MetalBuf::from_f64_as_f32(&self.device, &all_waves)?;
        let d_mags = MetalBuf::from_f64_as_f32(&self.device, &all_mags)?;
        let d_noise_var = MetalBuf::from_f64_as_f32(&self.device, &all_noise_var)?;
        let d_offsets = MetalBuf::from_i32(&self.device, &offsets)?;
        let d_query_t = MetalBuf::from_f64_as_f32(&self.device, query_times)?;
        let d_query_w = MetalBuf::from_f64_as_f32(&self.device, query_waves)?;
        let d_amps = MetalBuf::from_f64_as_f32(&self.device, amp_candidates)?;
        let d_lst = MetalBuf::from_f64_as_f32(&self.device, lst_candidates)?;
        let d_lsw = MetalBuf::from_f64_as_f32(&self.device, lsw_candidates)?;

        let d_gp_state = MetalBuf::alloc_f32(&self.device, n_sources * gp2d_state_size)?;
        let d_pred_grid = MetalBuf::alloc_f32(&self.device, n_sources * n_pred)?;
        let d_std_grid = MetalBuf::alloc_f32(&self.device, n_sources * n_pred)?;

        let pipeline = self.pipeline("batch_gp2d_fit_predict")?;

        // GP2D kernel expects a Gp2dParams struct at buffer(13)
        #[repr(C)]
        struct Gp2dParams {
            n_sources: i32,
            n_pred: i32,
            n_hp_amp: i32,
            n_hp_lst: i32,
            n_hp_lsw: i32,
            max_subsample: i32,
        }
        let params_struct = Gp2dParams {
            n_sources: n_sources as i32,
            n_pred: n_pred as i32,
            n_hp_amp: amp_candidates.len() as i32,
            n_hp_lst: lst_candidates.len() as i32,
            n_hp_lsw: lsw_candidates.len() as i32,
            max_subsample: max_subsample as i32,
        };

        let (cmd_buf, encoder) = self.begin_compute()?;
        encoder.setComputePipelineState(&pipeline);
        encoder.setBuffer_offset_atIndex(Some(&d_times.buffer), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(&d_waves.buffer), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&d_mags.buffer), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(&d_noise_var.buffer), 0, 3);
        encoder.setBuffer_offset_atIndex(Some(&d_offsets.buffer), 0, 4);
        encoder.setBuffer_offset_atIndex(Some(&d_query_t.buffer), 0, 5);
        encoder.setBuffer_offset_atIndex(Some(&d_query_w.buffer), 0, 6);
        encoder.setBuffer_offset_atIndex(Some(&d_amps.buffer), 0, 7);
        encoder.setBuffer_offset_atIndex(Some(&d_lst.buffer), 0, 8);
        encoder.setBuffer_offset_atIndex(Some(&d_lsw.buffer), 0, 9);
        encoder.setBuffer_offset_atIndex(Some(&d_gp_state.buffer), 0, 10);
        encoder.setBuffer_offset_atIndex(Some(&d_pred_grid.buffer), 0, 11);
        encoder.setBuffer_offset_atIndex(Some(&d_std_grid.buffer), 0, 12);
        encoder.setBytes_length_atIndex(
            &params_struct as *const Gp2dParams as *const _,
            std::mem::size_of::<Gp2dParams>(),
            13,
        );

        let threadgroups = MTLSize { width: n_sources as u64, height: 1, depth: 1 };
        let threads_per = MTLSize { width: n_hp_total as u64, height: 1, depth: 1 };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per);
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        // Download
        let mut h_gp_state = vec![0.0f64; n_sources * gp2d_state_size];
        let mut h_pred_grid = vec![0.0f64; n_sources * n_pred];
        let mut h_std_grid = vec![0.0f64; n_sources * n_pred];

        d_gp_state.download_as_f64(&mut h_gp_state)?;
        d_pred_grid.download_as_f64(&mut h_pred_grid)?;
        d_std_grid.download_as_f64(&mut h_std_grid)?;

        // Build results (same logic as CUDA)
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
                    let step = if n_eval < n_train {
                        n_train as f64 / n_eval as f64
                    } else {
                        1.0
                    };
                    for i in 0..n_eval {
                        let idx = (i as f64 * step) as usize;
                        let t = sources[s].times[idx];
                        let w = sources[s].waves[idx];
                        let mut pred_val = y_mean;
                        for j in 0..m {
                            let dt = t - x_t[j];
                            let dw = w - x_w[j];
                            pred_val +=
                                amp * (-dt * dt * inv_2lst2 - dw * dw * inv_2lsw2).exp() * alpha[j];
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

    // -----------------------------------------------------------------------
    // batch_svi_fit
    // -----------------------------------------------------------------------

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
        assert_eq!(n_sources, data.n_sources);

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

        let d_pso = MetalBuf::from_f64_as_f32(&self.device, &h_pso_params)?;
        let d_model_ids = MetalBuf::from_i32(&self.device, &h_model_ids)?;
        let d_n_params = MetalBuf::from_i32(&self.device, &h_n_params)?;
        let d_se_idx = MetalBuf::from_i32(&self.device, &h_se_idx)?;
        let d_prior_centers = MetalBuf::from_f64_as_f32(&self.device, &h_prior_centers)?;
        let d_prior_widths = MetalBuf::from_f64_as_f32(&self.device, &h_prior_widths)?;

        let d_out_mu = MetalBuf::alloc_f32(&self.device, n_sources * max_params)?;
        let d_out_ls = MetalBuf::alloc_f32(&self.device, n_sources * max_params)?;
        let d_out_elbo = MetalBuf::alloc_f32(&self.device, n_sources)?;

        let pipeline = self.pipeline("batch_svi_fit")?;

        // SVI kernel expects config packed into buffers 15 and 16
        let config_ints: [i32; 4] = [
            n_sources as i32,
            max_params as i32,
            n_steps as i32,
            n_samples as i32,
        ];
        let config_floats: [f32; 1] = [lr as f32];
        let d_config_ints = MetalBuf::from_i32(&self.device, &config_ints)?;
        let d_config_floats = {
            let buf = MetalBuf::alloc_f32(&self.device, 1)?;
            let ptr = buf.buffer.contents().as_ptr() as *mut f32;
            unsafe { ptr.write(config_floats[0]) };
            buf
        };

        let block: usize = 128;
        let total_threads = n_sources * 32; // one simd group per source
        let grid = (total_threads + block - 1) / block;

        let (cmd_buf, encoder) = self.begin_compute()?;
        encoder.setComputePipelineState(&pipeline);
        encoder.setBuffer_offset_atIndex(Some(&data.d_times.buffer), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(&data.d_flux.buffer), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&data.d_obs_var.buffer), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(&data.d_is_upper.buffer), 0, 3);
        encoder.setBuffer_offset_atIndex(Some(&data.d_upper_flux.buffer), 0, 4);
        encoder.setBuffer_offset_atIndex(Some(&data.d_offsets.buffer), 0, 5);
        encoder.setBuffer_offset_atIndex(Some(&d_pso.buffer), 0, 6);
        encoder.setBuffer_offset_atIndex(Some(&d_model_ids.buffer), 0, 7);
        encoder.setBuffer_offset_atIndex(Some(&d_n_params.buffer), 0, 8);
        encoder.setBuffer_offset_atIndex(Some(&d_se_idx.buffer), 0, 9);
        encoder.setBuffer_offset_atIndex(Some(&d_prior_centers.buffer), 0, 10);
        encoder.setBuffer_offset_atIndex(Some(&d_prior_widths.buffer), 0, 11);
        encoder.setBuffer_offset_atIndex(Some(&d_out_mu.buffer), 0, 12);
        encoder.setBuffer_offset_atIndex(Some(&d_out_ls.buffer), 0, 13);
        encoder.setBuffer_offset_atIndex(Some(&d_out_elbo.buffer), 0, 14);
        encoder.setBuffer_offset_atIndex(Some(&d_config_ints.buffer), 0, 15);
        encoder.setBuffer_offset_atIndex(Some(&d_config_floats.buffer), 0, 16);

        let threadgroups = MTLSize { width: grid as u64, height: 1, depth: 1 };
        let threads_per = MTLSize { width: block as u64, height: 1, depth: 1 };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per);
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        let mut h_mu = vec![0.0f64; n_sources * max_params];
        let mut h_ls = vec![0.0f64; n_sources * max_params];
        let mut h_elbo = vec![0.0f64; n_sources];
        d_out_mu.download_as_f64(&mut h_mu)?;
        d_out_ls.download_as_f64(&mut h_ls)?;
        d_out_elbo.download_as_f64(&mut h_elbo)?;

        let results: Vec<SviBatchOutput> = (0..n_sources)
            .map(|i| {
                let np = inputs[i].pso_params.len().min(max_params);
                let base = i * max_params;
                SviBatchOutput {
                    mu: h_mu[base..base + np].to_vec(),
                    log_sigma: h_ls[base..base + np].to_vec(),
                    elbo: h_elbo[i],
                }
            })
            .collect();

        Ok(results)
    }
}

// ---------------------------------------------------------------------------
// GpuBatchData (Metal)
// ---------------------------------------------------------------------------

/// Packed source data resident on the GPU (Metal backend).
pub struct GpuBatchData {
    d_times: MetalBuf,
    d_flux: MetalBuf,
    d_obs_var: MetalBuf,
    d_is_upper: MetalBuf,
    d_upper_flux: MetalBuf,
    d_offsets: MetalBuf,
    pub n_sources: usize,
    h_n_obs: Vec<usize>,
}

impl GpuBatchData {
    /// Upload packed source data to the GPU.
    ///
    /// Note: Metal version requires a GpuContext reference to access the device.
    /// Callers should use `GpuBatchData::new(sources)` — the Metal feature
    /// changes the signature to include the context.
    pub fn new(sources: &[BatchSource]) -> Result<Self, String> {
        let device = unsafe { MTLCreateSystemDefaultDevice() }
            .ok_or("No Metal device available for GpuBatchData")?;
        Self::new_with_device(sources, &device)
    }

    fn new_with_device(sources: &[BatchSource], device: &ProtocolObject<dyn MTLDevice>) -> Result<Self, String> {
        let n_sources = sources.len();

        let mut all_times = Vec::new();
        let mut all_flux = Vec::new();
        let mut all_obs_var = Vec::new();
        let mut all_is_upper: Vec<i32> = Vec::new();
        let mut all_upper_flux = Vec::new();
        let mut offsets: Vec<i32> = Vec::with_capacity(n_sources + 1);
        offsets.push(0);

        for src in sources {
            all_times.extend_from_slice(&src.times);
            all_flux.extend_from_slice(&src.flux);
            all_obs_var.extend_from_slice(&src.obs_var);
            all_is_upper.extend(src.is_upper.iter().map(|&b| b as i32));
            all_upper_flux.extend_from_slice(&src.upper_flux);
            offsets.push(all_times.len() as i32);
        }

        let h_n_obs: Vec<usize> = sources.iter().map(|s| s.times.len()).collect();

        Ok(Self {
            d_times: MetalBuf::from_f64_as_f32(device, &all_times)?,
            d_flux: MetalBuf::from_f64_as_f32(device, &all_flux)?,
            d_obs_var: MetalBuf::from_f64_as_f32(device, &all_obs_var)?,
            d_is_upper: MetalBuf::from_i32(device, &all_is_upper)?,
            d_upper_flux: MetalBuf::from_f64_as_f32(device, &all_upper_flux)?,
            d_offsets: MetalBuf::from_i32(device, &offsets)?,
            n_sources,
            h_n_obs,
        })
    }

    pub fn n_obs_per_source(&self, source_idx: usize) -> usize {
        self.h_n_obs.get(source_idx).copied().unwrap_or(1)
    }
}
