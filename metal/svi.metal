// GPU batch SVI (Stochastic Variational Inference) kernel — Metal port.
//
// One SIMD group (32 threads) per source/band. The 32 threads cooperate on the
// observation loop (the inner hot loop), then lane 0 does the Adam update.
// Uses analytical model gradients (finite-diff fallback for MetzgerKN only)
// and the reparameterization trick: theta = mu + sigma * eps, eps ~ N(0,1).

#include <metal_stdlib>
using namespace metal;

#include "models_device.h"

#define SVI_MAX_PARAMS 7
#define WARP_SIZE 32

// ===========================================================================
// 32-bit xorshift PRNG (replaces 64-bit xorshift64 from CUDA)
// ===========================================================================

inline uint xorshift32(thread uint* state) {
    uint x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

inline float rand_uniform(thread uint* rng) {
    return float(xorshift32(rng) & 0x1FFFFF) / float(0x1FFFFF);
}

inline float rand_normal(thread uint* rng) {
    float u1 = max(rand_uniform(rng), 1e-10f);
    float u2 = rand_uniform(rng);
    return sqrt(-2.0f * log(u1)) * cos(2.0f * M_PI_F * u2);
}

// Simdgroup-level reduction (sum)
inline float warp_reduce_sum(float val) {
    for (ushort offset = 16; offset > 0; offset >>= 1) {
        val += simd_shuffle_down(val, offset);
    }
    return val; // result valid in lane 0
}

// ===========================================================================
// Batch SVI kernel: one simdgroup per source, cooperative observation loop
// ===========================================================================

kernel void batch_svi_fit(
    // Observation data (concatenated, same layout as PSO)
    device const float* all_times        [[buffer(0)]],
    device const float* all_flux         [[buffer(1)]],
    device const float* all_obs_var      [[buffer(2)]],
    device const int*   all_is_upper     [[buffer(3)]],
    device const float* all_upper_flux   [[buffer(4)]],
    device const int*   source_offsets   [[buffer(5)]],
    // Per-source configuration
    device const float* pso_params       [[buffer(6)]],   // [n_sources * max_params]
    device const int*   model_ids        [[buffer(7)]],   // [n_sources]
    device const int*   n_params_arr     [[buffer(8)]],   // [n_sources]
    device const int*   se_idx_arr       [[buffer(9)]],   // [n_sources]
    // Prior parameters
    device const float* prior_centers    [[buffer(10)]],  // [n_sources * max_params]
    device const float* prior_widths     [[buffer(11)]],  // [n_sources * max_params]
    // Output
    device float*       out_mu           [[buffer(12)]],  // [n_sources * max_params]
    device float*       out_log_sigma    [[buffer(13)]],  // [n_sources * max_params]
    device float*       out_elbo         [[buffer(14)]],  // [n_sources]
    // Config (packed into a single buffer)
    device const int*   config_ints      [[buffer(15)]],  // [n_sources, max_params, n_steps, n_samples]
    device const float* config_floats    [[buffer(16)]],  // [lr]
    // Thread indexing
    uint tid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    // Each simdgroup handles one source
    uint warp_id = tid / WARP_SIZE;

    int n_sources  = config_ints[0];
    int max_params = config_ints[1];
    int n_steps    = config_ints[2];
    int n_samples  = config_ints[3];
    float lr       = config_floats[0];

    if ((int)warp_id >= n_sources) return;

    int obs_start = source_offsets[warp_id];
    int obs_end   = source_offsets[warp_id + 1];
    int n_obs     = obs_end - obs_start;
    if (n_obs <= 0) {
        if (lane == 0) out_elbo[warp_id] = -1e30f;
        return;
    }

    int model_id = model_ids[warp_id];
    int np       = n_params_arr[warp_id];
    int se_idx   = se_idx_arr[warp_id];
    bool use_analytical = (model_id != 7);

    // Load PSO init and prior params (all lanes load — cheap and avoids broadcast)
    int base_off = int(warp_id) * max_params;
    device const float* pso_base = pso_params + base_off;
    device const float* pc_base  = prior_centers + base_off;
    device const float* pw_base  = prior_widths + base_off;

    // Variational parameters (all lanes keep a copy for model eval)
    float mu[SVI_MAX_PARAMS];
    float log_sigma[SVI_MAX_PARAMS];
    float sigma[SVI_MAX_PARAMS];
    float prior_c[SVI_MAX_PARAMS];
    float prior_w_sq[SVI_MAX_PARAMS];

    for (int j = 0; j < np; j++) {
        mu[j] = pso_base[j];
        log_sigma[j] = -1.0f;
        prior_c[j] = pc_base[j];
        float w = pw_base[j];
        prior_w_sq[j] = w * w;
    }

    // Adam state (only lane 0 needs this, but register cost is minimal)
    float adam_m[2 * SVI_MAX_PARAMS];
    float adam_v[2 * SVI_MAX_PARAMS];
    for (int j = 0; j < 2 * np; j++) {
        adam_m[j] = 0.0f;
        adam_v[j] = 0.0f;
    }
    int adam_t = 0;

    // PRNG (lane 0 only, broadcasts samples)
    uint rng_state = 42u + uint(warp_id) * 123456789u;

    float theta[SVI_MAX_PARAMS];
    float eps_arr[SVI_MAX_PARAMS];
    float model_grad[SVI_MAX_PARAMS];

    float final_elbo = -1e30f;

    // Early stopping
    int svi_min_iters = 50;
    int svi_stall_iters = 20;
    float ema_alpha = 0.1f;
    float ema_elbo = -1e30f;
    float best_ema = -1e30f;
    int stall = 0;

    for (int step = 0; step < n_steps; step++) {
        for (int j = 0; j < np; j++) {
            sigma[j] = exp(log_sigma[j]);
        }

        // Per-param gradient accumulators (across samples)
        float grad_mu_acc[SVI_MAX_PARAMS];
        float grad_ls_acc[SVI_MAX_PARAMS];
        for (int j = 0; j < np; j++) {
            grad_mu_acc[j] = 0.0f;
            grad_ls_acc[j] = 0.0f;
        }
        float elbo_sum = 0.0f;

        for (int s = 0; s < n_samples; s++) {
            // Lane 0 generates samples, broadcast to all lanes
            for (int j = 0; j < np; j++) {
                float eps;
                if (lane == 0) eps = rand_normal(&rng_state);
                eps = simd_shuffle(eps, ushort(0));
                eps_arr[j] = eps;
                theta[j] = mu[j] + sigma[j] * eps;
            }

            float sigma_extra = exp(theta[se_idx]);
            float se_sq = sigma_extra * sigma_extra;

            // === Simdgroup-parallel observation loop ===
            // Each lane handles a strided subset of observations
            float local_log_lik = 0.0f;
            float local_dll[SVI_MAX_PARAMS];
            for (int j = 0; j < np; j++) local_dll[j] = 0.0f;

            for (int i = obs_start + int(lane); i < obs_end; i += WARP_SIZE) {
                float obs_t = all_times[i];
                float pred = lc_eval_model_at(model_id, theta, obs_t);
                if (!isfinite(pred)) continue;

                if (use_analytical) {
                    lc_eval_model_grad(model_id, theta, obs_t, model_grad);
                } else {
                    float theta_pert[SVI_MAX_PARAMS];
                    for (int k = 0; k < np; k++) theta_pert[k] = theta[k];
                    for (int j = 0; j < np; j++) {
                        if (j == se_idx) { model_grad[j] = 0.0f; continue; }
                        float h = max(1e-5f, 1e-5f * abs(theta[j]));
                        theta_pert[j] = theta[j] + h;
                        float pred_pert = lc_eval_model_at(model_id, theta_pert, obs_t);
                        theta_pert[j] = theta[j];
                        model_grad[j] = isfinite(pred_pert) ? (pred_pert - pred) / h : 0.0f;
                    }
                    model_grad[se_idx] = 0.0f;
                }

                float total_var = all_obs_var[i] + se_sq;
                float sigma_total = sqrt(total_var);
                float inv_total = 1.0f / total_var;

                if (all_is_upper[i]) {
                    float z = (all_upper_flux[i] - pred) / sigma_total;
                    local_log_lik += lc_log_normal_cdf_d(z);

                    float phi_z = exp(-0.5f * z * z) / sqrt(2.0f * M_PI_F);
                    float cdf_z = max(0.5f * (1.0f + lc_erf_approx(z * M_SQRT1_2_F)), 1e-38f);
                    float dll_dpred = -phi_z / (cdf_z * sigma_total);

                    for (int j = 0; j < np; j++) {
                        if (j != se_idx && isfinite(model_grad[j]))
                            local_dll[j] += dll_dpred * model_grad[j];
                    }

                    float dz_dlse = -(all_upper_flux[i] - pred) * se_sq / (sigma_total * total_var);
                    local_dll[se_idx] += (phi_z / cdf_z) * dz_dlse;
                } else {
                    float residual = all_flux[i] - pred;
                    float r2 = residual * residual;
                    local_log_lik += -0.5f * (r2 * inv_total + log(2.0f * M_PI_F * total_var));

                    for (int j = 0; j < np; j++) {
                        if (j != se_idx && isfinite(model_grad[j]))
                            local_dll[j] += residual * inv_total * model_grad[j];
                    }

                    local_dll[se_idx] += (r2 * inv_total * inv_total - inv_total) * se_sq;
                }
            }

            // Simdgroup-reduce the per-observation accumulators
            float log_lik = warp_reduce_sum(local_log_lik);
            float dll_dtheta[SVI_MAX_PARAMS];
            for (int j = 0; j < np; j++) {
                dll_dtheta[j] = warp_reduce_sum(local_dll[j]);
            }

            // Broadcast log_lik to all lanes for ELBO
            log_lik = simd_shuffle(log_lik, ushort(0));
            for (int j = 0; j < np; j++) {
                dll_dtheta[j] = simd_shuffle(dll_dtheta[j], ushort(0));
            }

            // Prior gradient + reparameterization gradients (all lanes compute, redundant but cheap)
            float log_prior = 0.0f;
            for (int j = 0; j < np; j++) {
                float d = theta[j] - prior_c[j];
                log_prior += -0.5f * d * d / prior_w_sq[j];
                float dlp = -d / prior_w_sq[j];
                float df = dll_dtheta[j] + dlp;
                grad_mu_acc[j] += df;
                grad_ls_acc[j] += df * sigma[j] * eps_arr[j];
            }

            elbo_sum += log_lik + log_prior;
        }

        float ns = float(n_samples);
        for (int j = 0; j < np; j++) {
            grad_mu_acc[j] /= ns;
            grad_ls_acc[j] /= ns;
        }
        elbo_sum /= ns;

        // Entropy
        float entropy = 0.0f;
        for (int j = 0; j < np; j++) entropy += log_sigma[j];
        entropy += 0.5f * float(np) * log(2.0f * M_PI_F * M_E_F);
        final_elbo = elbo_sum + entropy;

        // Adam update (all lanes compute — keeps mu/log_sigma in sync)
        adam_t++;
        float bc1 = 1.0f - pow(0.9f, float(adam_t));
        float bc2 = 1.0f - pow(0.999f, float(adam_t));
        for (int j = 0; j < np; j++) {
            // mu gradient
            float g_mu = -grad_mu_acc[j];
            if (isfinite(g_mu)) {
                adam_m[j] = 0.9f * adam_m[j] + 0.1f * g_mu;
                adam_v[j] = 0.999f * adam_v[j] + 0.001f * g_mu * g_mu;
                float m_hat = adam_m[j] / bc1;
                float v_hat = adam_v[j] / bc2;
                mu[j] -= lr * m_hat / (sqrt(v_hat) + 1e-8f);
            }
            // log_sigma gradient
            float g_ls = -(grad_ls_acc[j] + 1.0f);
            int ls_idx = np + j;
            if (isfinite(g_ls)) {
                adam_m[ls_idx] = 0.9f * adam_m[ls_idx] + 0.1f * g_ls;
                adam_v[ls_idx] = 0.999f * adam_v[ls_idx] + 0.001f * g_ls * g_ls;
                float m_hat = adam_m[ls_idx] / bc1;
                float v_hat = adam_v[ls_idx] / bc2;
                log_sigma[j] -= lr * m_hat / (sqrt(v_hat) + 1e-8f);
            }
        }

        // Clamp log_sigma
        for (int j = 0; j < np; j++) {
            log_sigma[j] = max(-6.0f, min(2.0f, log_sigma[j]));
        }

        // Early stopping
        if (step >= svi_min_iters) {
            if (ema_elbo < -1e29f) {
                ema_elbo = final_elbo;
            } else {
                ema_elbo = ema_alpha * final_elbo + (1.0f - ema_alpha) * ema_elbo;
            }
            float threshold = 0.01f * max(abs(best_ema), 1e-10f);
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
        device float* out_mu_base = out_mu + int(warp_id) * max_params;
        device float* out_ls_base = out_log_sigma + int(warp_id) * max_params;
        for (int j = 0; j < np; j++) {
            out_mu_base[j] = mu[j];
            out_ls_base[j] = max(-6.0f, min(2.0f, log_sigma[j]));
        }
        for (int j = np; j < max_params; j++) {
            out_mu_base[j] = 0.0f;
            out_ls_base[j] = 0.0f;
        }
        out_elbo[warp_id] = final_elbo;
    }
}
