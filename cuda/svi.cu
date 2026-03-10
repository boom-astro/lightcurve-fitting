// GPU batch SVI (Stochastic Variational Inference) kernel.
//
// One WARP (32 threads) per source/band. The 32 threads cooperate on the
// observation loop (the inner hot loop), then lane 0 does the Adam update.
// Uses analytical model gradients (finite-diff fallback for MetzgerKN only)
// and the reparameterization trick: theta = mu + sigma * eps, eps ~ N(0,1).

#include "models_device.h"

#define SVI_MAX_PARAMS 7
#define WARP_SIZE 32

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

// Warp-level reduction (sum)
__device__ inline double warp_reduce_sum(double val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val; // result valid in lane 0
}

// ===========================================================================
// Batch SVI kernel: one WARP per source, warp-cooperative observation loop
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
    double lr)
{
    // Each warp handles one source
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    if (warp_id >= n_sources) return;

    int obs_start = source_offsets[warp_id];
    int obs_end   = source_offsets[warp_id + 1];
    int n_obs     = obs_end - obs_start;
    if (n_obs <= 0) {
        if (lane == 0) out_elbo[warp_id] = -1e99;
        return;
    }

    int model_id = model_ids[warp_id];
    int np       = n_params_arr[warp_id];
    int se_idx   = se_idx_arr[warp_id];
    bool use_analytical = (model_id != 7);

    // Load PSO init and prior params (all lanes load — cheap and avoids broadcast)
    const double* pso_base = pso_params + (long long)warp_id * max_params;
    const double* pc_base  = prior_centers + (long long)warp_id * max_params;
    const double* pw_base  = prior_widths + (long long)warp_id * max_params;

    // Variational parameters (all lanes keep a copy for model eval)
    double mu[SVI_MAX_PARAMS];
    double log_sigma[SVI_MAX_PARAMS];
    double sigma[SVI_MAX_PARAMS];
    double prior_c[SVI_MAX_PARAMS];
    double prior_w_sq[SVI_MAX_PARAMS];

    for (int j = 0; j < np; j++) {
        mu[j] = pso_base[j];
        log_sigma[j] = -1.0;
        prior_c[j] = pc_base[j];
        double w = pw_base[j];
        prior_w_sq[j] = w * w;
    }

    // Adam state (only lane 0 needs this, but register cost is minimal)
    double adam_m[2 * SVI_MAX_PARAMS];
    double adam_v[2 * SVI_MAX_PARAMS];
    for (int j = 0; j < 2 * np; j++) {
        adam_m[j] = 0.0;
        adam_v[j] = 0.0;
    }
    int adam_t = 0;

    // PRNG (lane 0 only, broadcasts samples)
    unsigned long long rng_state = 42ULL + (unsigned long long)warp_id * 123456789ULL;

    double theta[SVI_MAX_PARAMS];
    double eps_arr[SVI_MAX_PARAMS];
    double model_grad[SVI_MAX_PARAMS];

    double final_elbo = -1e300;

    // Early stopping
    int svi_min_iters = 50;
    int svi_stall_iters = 20;
    double ema_alpha = 0.1;
    double ema_elbo = -1e300;
    double best_ema = -1e300;
    int stall = 0;

    for (int step = 0; step < n_steps; step++) {
        for (int j = 0; j < np; j++) {
            sigma[j] = exp(log_sigma[j]);
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
            // Lane 0 generates samples, broadcast to all lanes
            for (int j = 0; j < np; j++) {
                double eps;
                if (lane == 0) eps = rand_normal(&rng_state);
                eps = __shfl_sync(0xFFFFFFFF, eps, 0);
                eps_arr[j] = eps;
                theta[j] = mu[j] + sigma[j] * eps;
            }

            double sigma_extra = exp(theta[se_idx]);
            double se_sq = sigma_extra * sigma_extra;

            // === Warp-parallel observation loop ===
            // Each lane handles a strided subset of observations
            double local_log_lik = 0.0;
            double local_dll[SVI_MAX_PARAMS];
            for (int j = 0; j < np; j++) local_dll[j] = 0.0;

            for (int i = obs_start + lane; i < obs_end; i += WARP_SIZE) {
                double obs_t = all_times[i];
                double pred = lc_eval_model_at(model_id, theta, obs_t);
                if (!isfinite(pred)) continue;

                if (use_analytical) {
                    lc_eval_model_grad(model_id, theta, obs_t, model_grad);
                } else {
                    double theta_pert[SVI_MAX_PARAMS];
                    for (int k = 0; k < np; k++) theta_pert[k] = theta[k];
                    for (int j = 0; j < np; j++) {
                        if (j == se_idx) { model_grad[j] = 0.0; continue; }
                        double h = fmax(1e-5, 1e-5 * fabs(theta[j]));
                        theta_pert[j] = theta[j] + h;
                        double pred_pert = lc_eval_model_at(model_id, theta_pert, obs_t);
                        theta_pert[j] = theta[j];
                        model_grad[j] = isfinite(pred_pert) ? (pred_pert - pred) / h : 0.0;
                    }
                    model_grad[se_idx] = 0.0;
                }

                double total_var = all_obs_var[i] + se_sq;
                double sigma_total = sqrt(total_var);
                double inv_total = 1.0 / total_var;

                if (all_is_upper[i]) {
                    double z = (all_upper_flux[i] - pred) / sigma_total;
                    local_log_lik += lc_log_normal_cdf_d(z);

                    double phi_z = exp(-0.5 * z * z) / sqrt(2.0 * M_PI);
                    double cdf_z = fmax(0.5 * (1.0 + lc_erf_approx(z * M_SQRT1_2)), 1e-300);
                    double dll_dpred = -phi_z / (cdf_z * sigma_total);

                    for (int j = 0; j < np; j++) {
                        if (j != se_idx && isfinite(model_grad[j]))
                            local_dll[j] += dll_dpred * model_grad[j];
                    }

                    double dz_dlse = -(all_upper_flux[i] - pred) * se_sq / (sigma_total * total_var);
                    local_dll[se_idx] += (phi_z / cdf_z) * dz_dlse;
                } else {
                    double residual = all_flux[i] - pred;
                    double r2 = residual * residual;
                    local_log_lik += -0.5 * (r2 * inv_total + log(2.0 * M_PI * total_var));

                    for (int j = 0; j < np; j++) {
                        if (j != se_idx && isfinite(model_grad[j]))
                            local_dll[j] += residual * inv_total * model_grad[j];
                    }

                    local_dll[se_idx] += (r2 * inv_total * inv_total - inv_total) * se_sq;
                }
            }

            // Warp-reduce the per-observation accumulators
            double log_lik = warp_reduce_sum(local_log_lik);
            double dll_dtheta[SVI_MAX_PARAMS];
            for (int j = 0; j < np; j++) {
                dll_dtheta[j] = warp_reduce_sum(local_dll[j]);
            }

            // Broadcast log_lik to all lanes for ELBO
            log_lik = __shfl_sync(0xFFFFFFFF, log_lik, 0);
            for (int j = 0; j < np; j++) {
                dll_dtheta[j] = __shfl_sync(0xFFFFFFFF, dll_dtheta[j], 0);
            }

            // Prior gradient + reparameterization gradients (all lanes compute, redundant but cheap)
            double log_prior = 0.0;
            for (int j = 0; j < np; j++) {
                double d = theta[j] - prior_c[j];
                log_prior += -0.5 * d * d / prior_w_sq[j];
                double dlp = -d / prior_w_sq[j];
                double df = dll_dtheta[j] + dlp;
                grad_mu_acc[j] += df;
                grad_ls_acc[j] += df * sigma[j] * eps_arr[j];
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
        for (int j = 0; j < np; j++) entropy += log_sigma[j];
        entropy += 0.5 * (double)np * log(2.0 * M_PI * M_E);
        final_elbo = elbo_sum + entropy;

        // Adam update (all lanes compute — keeps mu/log_sigma in sync)
        adam_t++;
        double bc1 = 1.0 - pow(0.9, (double)adam_t);
        double bc2 = 1.0 - pow(0.999, (double)adam_t);
        for (int j = 0; j < np; j++) {
            // mu gradient
            double g_mu = -grad_mu_acc[j];
            if (isfinite(g_mu)) {
                adam_m[j] = 0.9 * adam_m[j] + 0.1 * g_mu;
                adam_v[j] = 0.999 * adam_v[j] + 0.001 * g_mu * g_mu;
                double m_hat = adam_m[j] / bc1;
                double v_hat = adam_v[j] / bc2;
                mu[j] -= lr * m_hat / (sqrt(v_hat) + 1e-8);
            }
            // log_sigma gradient
            double g_ls = -(grad_ls_acc[j] + 1.0);
            int ls_idx = np + j;
            if (isfinite(g_ls)) {
                adam_m[ls_idx] = 0.9 * adam_m[ls_idx] + 0.1 * g_ls;
                adam_v[ls_idx] = 0.999 * adam_v[ls_idx] + 0.001 * g_ls * g_ls;
                double m_hat = adam_m[ls_idx] / bc1;
                double v_hat = adam_v[ls_idx] / bc2;
                log_sigma[j] -= lr * m_hat / (sqrt(v_hat) + 1e-8);
            }
        }

        // Clamp log_sigma using fmin/fmax (nvcc can't optimize these away)
        for (int j = 0; j < np; j++) {
            log_sigma[j] = fmax(-6.0, fmin(2.0, log_sigma[j]));
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
                if (stall >= svi_stall_iters) break;
            }
        }
    }

    // Write output (lane 0 only)
    if (lane == 0) {
        double* out_mu_base = out_mu + (long long)warp_id * max_params;
        double* out_ls_base = out_log_sigma + (long long)warp_id * max_params;
        for (int j = 0; j < np; j++) {
            out_mu_base[j] = mu[j];
            out_ls_base[j] = fmax(-6.0, fmin(2.0, log_sigma[j]));
        }
        for (int j = np; j < max_params; j++) {
            out_mu_base[j] = 0.0;
            out_ls_base[j] = 0.0;
        }
        out_elbo[warp_id] = final_elbo;
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
    int grid,
    int block)
{
    batch_svi_fit<<<grid, block>>>(
        all_times, all_flux, all_obs_var, all_is_upper, all_upper_flux,
        source_offsets,
        pso_params, model_ids, n_params_arr, se_idx_arr,
        prior_centers, prior_widths,
        out_mu, out_log_sigma, out_elbo,
        n_sources, max_params, n_steps, n_samples, lr);
}
