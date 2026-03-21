// GPU batch SVI (Stochastic Variational Inference) kernel.
//
// One BLOCK (128 threads) per source/band. All 128 threads cooperate on the
// observation loop (the inner hot loop), then thread 0 does the Adam update.
// Uses fused model eval+gradient (finite-diff fallback for MetzgerKN only),
// Reparameterization trick: theta = mu + sigma * eps, eps ~ N(0,1).

#include "models_device.h"

#define SVI_MAX_PARAMS 7
#define SVI_BLOCK_SIZE 128
#define SVI_MAX_CACHED_OBS 128
#define SVI_WARP_COUNT (SVI_BLOCK_SIZE / 32)

// ===========================================================================
// Device-side xorshift64 PRNG
// ===========================================================================

__device__ inline unsigned long long xorshift64(unsigned long long* state) {
    unsigned long long x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

__device__ inline double rand_uniform(unsigned long long* rng) {
    return (double)(xorshift64(rng) & 0x1FFFFFFFFFFFFF) / (double)0x1FFFFFFFFFFFFF;
}

__device__ inline double rand_normal(unsigned long long* rng) {
    double u1 = fmax(rand_uniform(rng), 1e-10);
    double u2 = rand_uniform(rng);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

// ===========================================================================
// Block-level reduction (sum across all 128 threads)
// ===========================================================================

__device__ inline double svi_block_reduce_sum(double val, volatile double* s_warp_buf) {
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    // Intra-warp reduction via shuffles
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    if (lane == 0) s_warp_buf[warp_id] = val;
    __syncthreads();
    // Thread 0 sums across warps
    if (threadIdx.x == 0) {
        double sum = 0.0;
        for (int w = 0; w < SVI_WARP_COUNT; w++) sum += s_warp_buf[w];
        s_warp_buf[0] = sum;
    }
    __syncthreads();
    return s_warp_buf[0]; // broadcast to all threads
}

// ===========================================================================
// Shared memory size helper (host-side)
// ===========================================================================

static size_t svi_smem_bytes(int max_obs) {
    int n_cached = (max_obs < SVI_MAX_CACHED_OBS) ? max_obs : SVI_MAX_CACHED_OBS;
    size_t bytes = 0;
    // SVI state: mu[7] + log_sigma[7] + sigma[7] + theta[7] + eps[7]
    bytes += 5 * SVI_MAX_PARAMS * sizeof(double);  // 280 bytes
    // Warp reduction scratch
    bytes += SVI_WARP_COUNT * sizeof(double);  // 32 bytes
    // Done flag
    bytes += sizeof(int);  // 4 bytes
    // Alignment to 8 bytes
    bytes = (bytes + 7) & ~(size_t)7;
    // Observation cache
    if (n_cached > 0) {
        bytes += 4 * n_cached * sizeof(double);  // times, flux, obs_var, upper_flux
        bytes += n_cached * sizeof(int);          // is_upper
    }
    return bytes;
}

// ===========================================================================
// Batch SVI kernel: one BLOCK per source, block-cooperative observation loop
// ===========================================================================

extern "C" __global__ void batch_svi_fit(
    // Observation data (concatenated, same layout as PSO)
    const double* __restrict__ all_times,
    const double* __restrict__ all_flux,
    const double* __restrict__ all_obs_var,
    const int*    __restrict__ all_is_upper,
    const double* __restrict__ all_upper_flux,
    const int*    __restrict__ source_offsets,
    // Per-source configuration
    const double* __restrict__ pso_params,      // [n_sources * max_params]
    const int*    __restrict__ model_ids,        // [n_sources]
    const int*    __restrict__ n_params_arr,     // [n_sources]
    const int*    __restrict__ se_idx_arr,       // [n_sources]
    // Prior parameters
    const double* __restrict__ prior_centers,    // [n_sources * max_params]
    const double* __restrict__ prior_widths,     // [n_sources * max_params]
    // Output
    double* __restrict__ out_mu,                 // [n_sources * max_params]
    double* __restrict__ out_log_sigma,          // [n_sources * max_params]
    double* __restrict__ out_elbo,               // [n_sources]
    // Config
    int n_sources,
    int max_params,
    int n_steps,
    int n_samples,
    double lr,
    int max_obs)
{
    int src = blockIdx.x;
    int tid = threadIdx.x;

    if (src >= n_sources) return;

    int obs_start = source_offsets[src];
    int obs_end   = source_offsets[src + 1];
    int n_obs     = obs_end - obs_start;
    if (n_obs <= 0) {
        if (tid == 0) out_elbo[src] = -1e99;
        return;
    }

    int model_id = model_ids[src];
    int np       = n_params_arr[src];
    int se_idx   = se_idx_arr[src];

    // Load PSO init and prior params
    const double* pso_base = pso_params + (long long)src * max_params;
    const double* pc_base  = prior_centers + (long long)src * max_params;
    const double* pw_base  = prior_widths + (long long)src * max_params;

    // -----------------------------------------------------------------------
    // Parse shared memory layout
    // -----------------------------------------------------------------------
    extern __shared__ char smem[];
    char* ptr = smem;

    double* s_mu        = (double*)ptr; ptr += SVI_MAX_PARAMS * sizeof(double);
    double* s_log_sigma = (double*)ptr; ptr += SVI_MAX_PARAMS * sizeof(double);
    double* s_sigma     = (double*)ptr; ptr += SVI_MAX_PARAMS * sizeof(double);
    double* s_theta     = (double*)ptr; ptr += SVI_MAX_PARAMS * sizeof(double);
    double* s_eps       = (double*)ptr; ptr += SVI_MAX_PARAMS * sizeof(double);
    volatile double* s_warp_buf = (volatile double*)ptr; ptr += SVI_WARP_COUNT * sizeof(double);
    int* s_done         = (int*)ptr;    ptr += sizeof(int);
    // Align to 8 bytes
    ptr = (char*)(((size_t)ptr + 7) & ~(size_t)7);

    // Observation cache pointers
    int use_cache = (n_obs <= max_obs && n_obs <= SVI_MAX_CACHED_OBS);
    double* s_times      = NULL;
    double* s_flux       = NULL;
    double* s_obs_var    = NULL;
    double* s_upper_flux = NULL;
    int*    s_is_upper   = NULL;

    if (use_cache) {
        s_times      = (double*)ptr; ptr += n_obs * sizeof(double);
        s_flux       = (double*)ptr; ptr += n_obs * sizeof(double);
        s_obs_var    = (double*)ptr; ptr += n_obs * sizeof(double);
        s_upper_flux = (double*)ptr; ptr += n_obs * sizeof(double);
        s_is_upper   = (int*)ptr;
    }

    // Cooperative cache load
    if (use_cache) {
        for (int i = tid; i < n_obs; i += SVI_BLOCK_SIZE) {
            s_times[i]      = all_times[obs_start + i];
            s_flux[i]       = all_flux[obs_start + i];
            s_obs_var[i]    = all_obs_var[obs_start + i];
            s_upper_flux[i] = all_upper_flux[obs_start + i];
            s_is_upper[i]   = all_is_upper[obs_start + i];
        }
    }

    // Set observation pointers (cache or global fallback)
    const double* obs_t  = use_cache ? s_times      : (all_times + obs_start);
    const double* obs_f  = use_cache ? s_flux       : (all_flux + obs_start);
    const double* obs_v  = use_cache ? s_obs_var    : (all_obs_var + obs_start);
    const double* obs_uf = use_cache ? s_upper_flux : (all_upper_flux + obs_start);
    const int*    obs_u  = use_cache ? s_is_upper   : (all_is_upper + obs_start);

    // Thread 0 initializes shared variational state
    if (tid == 0) {
        for (int j = 0; j < np; j++) {
            s_mu[j] = pso_base[j];
            s_log_sigma[j] = -1.0;
        }
        *s_done = 0;
    }

    // Prior params (all threads load into registers)
    double prior_c[SVI_MAX_PARAMS];
    double prior_w_sq[SVI_MAX_PARAMS];
    for (int j = 0; j < np; j++) {
        prior_c[j] = pc_base[j];
        double w = pw_base[j];
        prior_w_sq[j] = w * w;
    }

    __syncthreads();

    // Adam state — only thread 0 uses these
    double adam_m[2 * SVI_MAX_PARAMS];
    double adam_v_arr[2 * SVI_MAX_PARAMS];
    if (tid == 0) {
        for (int j = 0; j < 2 * np; j++) {
            adam_m[j] = 0.0;
            adam_v_arr[j] = 0.0;
        }
    }
    int adam_t = 0;

    // PRNG (thread 0 only)
    unsigned long long rng_state = 42ULL + (unsigned long long)src * 123456789ULL;

    double model_grad[SVI_MAX_PARAMS];
    double final_elbo = -1e300;

    // Early stopping state (thread 0)
    int svi_min_iters = 50;
    int svi_stall_iters = 20;
    double ema_alpha = 0.1;
    double ema_elbo = -1e300;
    double best_ema = -1e300;
    int stall = 0;

    for (int step = 0; step < n_steps; step++) {
        // Cosine annealing LR (matches CPU schedule)
        double step_lr;
        if (step < 50) {
            step_lr = lr * 0.1 + lr * 0.9 * (double)step / 50.0;
        } else {
            int denom = (n_steps - 50) > 1 ? (n_steps - 50) : 1;
            double progress = (double)(step - 50) / (double)denom;
            step_lr = lr * (0.1 + 0.9 * 0.5 * (1.0 + cos(M_PI * progress)));
        }

        // Thread 0: compute sigma from log_sigma
        if (tid == 0) {
            for (int j = 0; j < np; j++) {
                s_sigma[j] = exp(s_log_sigma[j]);
            }
        }
        __syncthreads();

        // Read current variational state into registers (all threads)
        double r_mu[SVI_MAX_PARAMS];
        double r_sigma[SVI_MAX_PARAMS];
        double r_log_sigma[SVI_MAX_PARAMS];
        for (int j = 0; j < np; j++) {
            r_mu[j] = s_mu[j];
            r_sigma[j] = s_sigma[j];
            r_log_sigma[j] = s_log_sigma[j];
        }

        // Per-param gradient accumulators (across samples)
        double grad_mu_acc[SVI_MAX_PARAMS];
        double grad_ls_acc[SVI_MAX_PARAMS];
        for (int j = 0; j < np; j++) {
            grad_mu_acc[j] = 0.0;
            grad_ls_acc[j] = 0.0;
        }
        double elbo_sum = 0.0;

        for (int s = 0; s < n_samples; s++) {
            // Thread 0 generates eps, computes theta, writes to shared
            if (tid == 0) {
                for (int j = 0; j < np; j++) {
                    double eps = rand_normal(&rng_state);
                    s_eps[j] = eps;
                    s_theta[j] = r_mu[j] + r_sigma[j] * eps;
                }
            }
            __syncthreads();

            // All threads read theta and eps from shared memory
            double theta[SVI_MAX_PARAMS];
            double eps_arr[SVI_MAX_PARAMS];
            for (int j = 0; j < np; j++) {
                theta[j] = s_theta[j];
                eps_arr[j] = s_eps[j];
            }

            double sigma_extra = exp(theta[se_idx]);
            double se_sq = sigma_extra * sigma_extra;

            // === Block-parallel observation loop ===
            double local_log_lik = 0.0;
            double local_dll[SVI_MAX_PARAMS];
            for (int j = 0; j < np; j++) local_dll[j] = 0.0;

            for (int i = tid; i < n_obs; i += SVI_BLOCK_SIZE) {
                double t_val = obs_t[i];

                // Fused eval+grad
                bool ok;
                double pred = lc_eval_model_at_and_grad(model_id, theta, t_val, model_grad, &ok);
                if (!ok) {
                    // MetzgerKN finite-diff fallback
                    pred = lc_eval_model_at(model_id, theta, t_val);
                    if (!isfinite(pred)) continue;
                    double theta_pert[SVI_MAX_PARAMS];
                    for (int k = 0; k < np; k++) theta_pert[k] = theta[k];
                    for (int j = 0; j < np; j++) {
                        if (j == se_idx) { model_grad[j] = 0.0; continue; }
                        double h = fmax(1e-5, 1e-5 * fabs(theta[j]));
                        theta_pert[j] = theta[j] + h;
                        double pred_pert = lc_eval_model_at(model_id, theta_pert, t_val);
                        theta_pert[j] = theta[j];
                        model_grad[j] = isfinite(pred_pert) ? (pred_pert - pred) / h : 0.0;
                    }
                    model_grad[se_idx] = 0.0;
                } else {
                    if (!isfinite(pred)) continue;
                }

                double total_var = obs_v[i] + se_sq;
                double inv_sigma = 1.0 / sqrt(total_var);
                double inv_total = inv_sigma * inv_sigma;

                if (obs_u[i]) {
                    double z = (obs_uf[i] - pred) * inv_sigma;
                    local_log_lik += lc_log_normal_cdf_d(z);

                    double phi_z = exp(-0.5 * z * z) / sqrt(2.0 * M_PI);
                    double cdf_z = fmax(0.5 * (1.0 + lc_erf_approx(z * M_SQRT1_2)), 1e-300);
                    double dll_dpred = -phi_z * inv_sigma / cdf_z;

                    for (int j = 0; j < np; j++) {
                        if (j != se_idx && isfinite(model_grad[j]))
                            local_dll[j] += dll_dpred * model_grad[j];
                    }

                    double dz_dlse = -(obs_uf[i] - pred) * se_sq * inv_sigma * inv_total;
                    local_dll[se_idx] += (phi_z / cdf_z) * dz_dlse;
                } else {
                    double residual = obs_f[i] - pred;
                    double r2 = residual * residual;
                    local_log_lik += -0.5 * (r2 * inv_total + log(2.0 * M_PI * total_var));

                    for (int j = 0; j < np; j++) {
                        if (j != se_idx && isfinite(model_grad[j]))
                            local_dll[j] += residual * inv_total * model_grad[j];
                    }

                    local_dll[se_idx] += (r2 * inv_total * inv_total - inv_total) * se_sq;
                }
            }

            // Block-level reduction
            double log_lik = svi_block_reduce_sum(local_log_lik, s_warp_buf);
            double dll_dtheta[SVI_MAX_PARAMS];
            for (int j = 0; j < np; j++) {
                dll_dtheta[j] = svi_block_reduce_sum(local_dll[j], s_warp_buf);
            }

            // Prior gradient + reparameterization gradients (all threads compute, redundant but cheap)
            double log_prior = 0.0;
            for (int j = 0; j < np; j++) {
                double d = theta[j] - prior_c[j];
                log_prior += -0.5 * d * d / prior_w_sq[j];
                double dlp = -d / prior_w_sq[j];
                double df = dll_dtheta[j] + dlp;
                grad_mu_acc[j] += df;
                grad_ls_acc[j] += df * r_sigma[j] * eps_arr[j];
            }

            elbo_sum += log_lik + log_prior;
        }

        double ns = (double)n_samples;
        for (int j = 0; j < np; j++) {
            grad_mu_acc[j] /= ns;
            grad_ls_acc[j] /= ns;
        }
        elbo_sum /= ns;

        // Entropy
        double entropy = 0.0;
        for (int j = 0; j < np; j++) entropy += r_log_sigma[j];
        entropy += 0.5 * (double)np * log(2.0 * M_PI * M_E);
        final_elbo = elbo_sum + entropy;

        // Thread 0: Adam update + early stopping
        if (tid == 0) {
            // Gradient clipping (clip norm to 10 if ||grad||^2 > 100)
            double grad_norm_sq = 0.0;
            for (int j = 0; j < np; j++) {
                double g_mu = -grad_mu_acc[j];
                double g_ls = -(grad_ls_acc[j] + 1.0);
                grad_norm_sq += g_mu * g_mu + g_ls * g_ls;
            }
            if (grad_norm_sq > 100.0) {
                double scale = 10.0 / sqrt(grad_norm_sq);
                for (int j = 0; j < np; j++) {
                    grad_mu_acc[j] *= scale;
                    grad_ls_acc[j] *= scale;
                }
            }

            adam_t++;
            double bc1 = 1.0 - pow(0.9, (double)adam_t);
            double bc2 = 1.0 - pow(0.999, (double)adam_t);
            for (int j = 0; j < np; j++) {
                // mu gradient
                double g_mu = -grad_mu_acc[j];
                if (isfinite(g_mu)) {
                    adam_m[j] = 0.9 * adam_m[j] + 0.1 * g_mu;
                    adam_v_arr[j] = 0.999 * adam_v_arr[j] + 0.001 * g_mu * g_mu;
                    double m_hat = adam_m[j] / bc1;
                    double v_hat = adam_v_arr[j] / bc2;
                    s_mu[j] = r_mu[j] - step_lr * m_hat / (sqrt(v_hat) + 1e-8);
                }
                // log_sigma gradient
                double g_ls = -(grad_ls_acc[j] + 1.0);
                int ls_idx = np + j;
                if (isfinite(g_ls)) {
                    adam_m[ls_idx] = 0.9 * adam_m[ls_idx] + 0.1 * g_ls;
                    adam_v_arr[ls_idx] = 0.999 * adam_v_arr[ls_idx] + 0.001 * g_ls * g_ls;
                    double m_hat = adam_m[ls_idx] / bc1;
                    double v_hat = adam_v_arr[ls_idx] / bc2;
                    s_log_sigma[j] = r_log_sigma[j] - step_lr * m_hat / (sqrt(v_hat) + 1e-8);
                }
            }

            // Clamp log_sigma
            for (int j = 0; j < np; j++) {
                s_log_sigma[j] = fmax(-6.0, fmin(2.0, s_log_sigma[j]));
            }

            // Early stopping
            if (step >= svi_min_iters) {
                if (ema_elbo < -1e290) {
                    ema_elbo = final_elbo;
                } else {
                    ema_elbo = ema_alpha * final_elbo + (1.0 - ema_alpha) * ema_elbo;
                }
                double threshold = 0.01 * fmax(fabs(best_ema), 1e-10);
                if (ema_elbo - best_ema > threshold) {
                    best_ema = ema_elbo;
                    stall = 0;
                } else {
                    stall++;
                    if (stall >= svi_stall_iters) *s_done = 1;
                }
            }
        }
        __syncthreads();
        if (*s_done) break;
    }

    // Write output (thread 0 only)
    if (tid == 0) {
        double* out_mu_base = out_mu + (long long)src * max_params;
        double* out_ls_base = out_log_sigma + (long long)src * max_params;
        for (int j = 0; j < np; j++) {
            out_mu_base[j] = s_mu[j];
            out_ls_base[j] = fmax(-6.0, fmin(2.0, s_log_sigma[j]));
        }
        for (int j = np; j < max_params; j++) {
            out_mu_base[j] = 0.0;
            out_ls_base[j] = 0.0;
        }
        out_elbo[src] = final_elbo;
    }
}

// ===========================================================================
// Host-side launch wrapper
// ===========================================================================

extern "C" void launch_batch_svi_fit(
    const double* all_times,
    const double* all_flux,
    const double* all_obs_var,
    const int*    all_is_upper,
    const double* all_upper_flux,
    const int*    source_offsets,
    const double* pso_params,
    const int*    model_ids,
    const int*    n_params_arr,
    const int*    se_idx_arr,
    const double* prior_centers,
    const double* prior_widths,
    double* out_mu,
    double* out_log_sigma,
    double* out_elbo,
    int n_sources,
    int max_params,
    int n_steps,
    int n_samples,
    double lr,
    int max_obs)
{
    size_t smem = svi_smem_bytes(max_obs);
    batch_svi_fit<<<n_sources, SVI_BLOCK_SIZE, smem>>>(
        all_times, all_flux, all_obs_var, all_is_upper, all_upper_flux,
        source_offsets,
        pso_params, model_ids, n_params_arr, se_idx_arr,
        prior_centers, prior_widths,
        out_mu, out_log_sigma, out_elbo,
        n_sources, max_params, n_steps, n_samples, lr, max_obs);
}
