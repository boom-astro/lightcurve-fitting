// Metal compute shader port of cuda/models.cu.
//
// Two kernel families:
//   1. *_eval: forward model evaluation (draw x time -> flux)
//   2. batch_pso_cost: fused model eval + likelihood for batch PSO fitting
//      Thread grid: n_sources x n_particles. Each thread loops over its
//      source's observations.

#include <metal_stdlib>
using namespace metal;

#include "models_device.h"

// ===========================================================================
// MultiBazin: sum of K Bazin components + baseline
// ===========================================================================
//
// Params layout for K components:
//   [log_A_1, t0_1, log_tau_rise_1, log_tau_fall_1,   // component 1
//    log_A_2, t0_2, log_tau_rise_2, log_tau_fall_2,   // component 2
//    ...
//    B, log_sigma_extra]                               // shared (last 2)
// Total = 4*K + 2 params.

#define MB_COMP_PARAMS 4

inline float bazin_component_at(const thread float* p, float t) {
    float a        = exp(p[0]);
    float t0       = p[1];
    float tau_rise = exp(p[2]);
    float tau_fall = exp(p[3]);
    float dt       = t - t0;
    return a * exp(-dt / tau_fall) * lc_sigmoid(dt / tau_rise);
}

inline float multi_bazin_at(const thread float* p, int k, float t) {
    float sum = 0.0f;
    for (int c = 0; c < k; c++) {
        sum += bazin_component_at(p + c * MB_COMP_PARAMS, t);
    }
    int b_idx = k * MB_COMP_PARAMS;
    return sum + p[b_idx]; // + B
}

// ===========================================================================
// Forward evaluation kernels (one thread per draw x time)
// ===========================================================================
//
// Buffer layout (consistent across all eval kernels):
//   buffer(0): params  [n_draws * n_params]
//   buffer(1): times   [n_times]
//   buffer(2): out     [n_draws * n_times]
//   buffer(3): n_draws
//   buffer(4): n_times
//   buffer(5): n_params

// Max params across all models (Villar=7, TDE=7)
#define MAX_EVAL_PARAMS 16

kernel void bazin_eval(
    device const float* params   [[buffer(0)]],
    device const float* times    [[buffer(1)]],
    device float*       out      [[buffer(2)]],
    device const int&   n_draws  [[buffer(3)]],
    device const int&   n_times  [[buffer(4)]],
    device const int&   n_params [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= uint(n_draws * n_times)) return;
    int draw = int(gid) / n_times;
    int ti   = int(gid) % n_times;
    float local_p[MAX_EVAL_PARAMS];
    for (int j = 0; j < n_params; j++) local_p[j] = params[draw * n_params + j];
    out[gid] = lc_bazin_at(local_p, times[ti]);
}

kernel void villar_eval(
    device const float* params   [[buffer(0)]],
    device const float* times    [[buffer(1)]],
    device float*       out      [[buffer(2)]],
    device const int&   n_draws  [[buffer(3)]],
    device const int&   n_times  [[buffer(4)]],
    device const int&   n_params [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= uint(n_draws * n_times)) return;
    int draw = int(gid) / n_times;
    int ti   = int(gid) % n_times;
    float local_p[MAX_EVAL_PARAMS];
    for (int j = 0; j < n_params; j++) local_p[j] = params[draw * n_params + j];
    out[gid] = lc_villar_at(local_p, times[ti]);
}

kernel void tde_eval(
    device const float* params   [[buffer(0)]],
    device const float* times    [[buffer(1)]],
    device float*       out      [[buffer(2)]],
    device const int&   n_draws  [[buffer(3)]],
    device const int&   n_times  [[buffer(4)]],
    device const int&   n_params [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= uint(n_draws * n_times)) return;
    int draw = int(gid) / n_times;
    int ti   = int(gid) % n_times;
    float local_p[MAX_EVAL_PARAMS];
    for (int j = 0; j < n_params; j++) local_p[j] = params[draw * n_params + j];
    out[gid] = lc_tde_at(local_p, times[ti]);
}

kernel void arnett_eval(
    device const float* params   [[buffer(0)]],
    device const float* times    [[buffer(1)]],
    device float*       out      [[buffer(2)]],
    device const int&   n_draws  [[buffer(3)]],
    device const int&   n_times  [[buffer(4)]],
    device const int&   n_params [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= uint(n_draws * n_times)) return;
    int draw = int(gid) / n_times;
    int ti   = int(gid) % n_times;
    float local_p[MAX_EVAL_PARAMS];
    for (int j = 0; j < n_params; j++) local_p[j] = params[draw * n_params + j];
    out[gid] = lc_arnett_at(local_p, times[ti]);
}

kernel void magnetar_eval(
    device const float* params   [[buffer(0)]],
    device const float* times    [[buffer(1)]],
    device float*       out      [[buffer(2)]],
    device const int&   n_draws  [[buffer(3)]],
    device const int&   n_times  [[buffer(4)]],
    device const int&   n_params [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= uint(n_draws * n_times)) return;
    int draw = int(gid) / n_times;
    int ti   = int(gid) % n_times;
    float local_p[MAX_EVAL_PARAMS];
    for (int j = 0; j < n_params; j++) local_p[j] = params[draw * n_params + j];
    out[gid] = lc_magnetar_at(local_p, times[ti]);
}

kernel void shock_cooling_eval(
    device const float* params   [[buffer(0)]],
    device const float* times    [[buffer(1)]],
    device float*       out      [[buffer(2)]],
    device const int&   n_draws  [[buffer(3)]],
    device const int&   n_times  [[buffer(4)]],
    device const int&   n_params [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= uint(n_draws * n_times)) return;
    int draw = int(gid) / n_times;
    int ti   = int(gid) % n_times;
    float local_p[MAX_EVAL_PARAMS];
    for (int j = 0; j < n_params; j++) local_p[j] = params[draw * n_params + j];
    out[gid] = lc_shock_cooling_at(local_p, times[ti]);
}

kernel void afterglow_eval(
    device const float* params   [[buffer(0)]],
    device const float* times    [[buffer(1)]],
    device float*       out      [[buffer(2)]],
    device const int&   n_draws  [[buffer(3)]],
    device const int&   n_times  [[buffer(4)]],
    device const int&   n_params [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= uint(n_draws * n_times)) return;
    int draw = int(gid) / n_times;
    int ti   = int(gid) % n_times;
    float local_p[MAX_EVAL_PARAMS];
    for (int j = 0; j < n_params; j++) local_p[j] = params[draw * n_params + j];
    out[gid] = lc_afterglow_at(local_p, times[ti]);
}

// ===========================================================================
// Metzger 1-zone kilonova ODE (sequential -- one thread per draw)
// ===========================================================================

// Physical constants
#define MSUN_CGS  1.989e33f
#define C_CGS_VAL 2.998e10f
#define SECS_DAY  86400.0f
#define METZGER_NGRID 200
#define METZGER_SCALE 1e30f

// Metzger kernel: one thread per draw, loops over all times.
// params layout per draw: [log10_mej, log10_vej, log10_kappa, t0, sigma_extra]
// Output: out[draw * n_times + ti] = normalized flux at times[ti]
//
// Buffer layout:
//   buffer(0): params  [n_draws * n_params]
//   buffer(1): times   [n_times]
//   buffer(2): out     [n_draws * n_times]
//   buffer(3): n_draws
//   buffer(4): n_times
//   buffer(5): n_params

kernel void metzger_kn_eval(
    device const float* params   [[buffer(0)]],
    device const float* times    [[buffer(1)]],
    device float*       out      [[buffer(2)]],
    device const int&   n_draws  [[buffer(3)]],
    device const int&   n_times  [[buffer(4)]],
    device const int&   n_params [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    int draw = int(gid);
    if (draw >= n_draws) return;

    device const float* p_dev = params + draw * n_params;
    float m_ej    = pow(10.0f, p_dev[0]) * MSUN_CGS;
    float v_ej    = pow(10.0f, p_dev[1]) * C_CGS_VAL;
    float kappa_r = pow(10.0f, p_dev[2]);
    float t0      = p_dev[3];

    // Find max phase needed
    float phase_max = 0.01f;
    for (int i = 0; i < n_times; i++) {
        float ph = times[i] - t0;
        if (ph > phase_max) phase_max = ph;
    }

    if (phase_max <= 0.01f) {
        for (int i = 0; i < n_times; i++)
            out[draw * n_times + i] = 0.0f;
        return;
    }

    // Build log-spaced time grid
    float log_t_min = log(0.01f);
    float log_t_max = log(phase_max * 1.05f);

    // ODE state
    float ye = 0.1f;
    float xn0 = 1.0f - 2.0f * ye;
    float e0 = 0.5f * m_ej * v_ej * v_ej;
    float e_th = e0 / METZGER_SCALE;
    float e_kin = e0 / METZGER_SCALE;
    float v = v_ej;

    float grid_t_prev = exp(log_t_min);
    float r = grid_t_prev * SECS_DAY * v;

    // Store grid values in local arrays (stack-allocated, 200 entries)
    float grid_t[METZGER_NGRID];
    float grid_lrad[METZGER_NGRID];

    for (int i = 0; i < METZGER_NGRID; i++) {
        float t_day = exp(log_t_min + (log_t_max - log_t_min) * float(i) / float(METZGER_NGRID - 1));
        grid_t[i] = t_day;
        float t_sec = t_day * SECS_DAY;

        // Thermalization efficiency
        float eth_factor = 0.34f * pow(t_day, 0.74f);
        float eth_log_term = (eth_factor > 1e-10f) ? log(1.0f + eth_factor) / eth_factor : 1.0f;
        float eth = 0.36f * (exp(-0.56f * t_day) + eth_log_term);

        // Heating rate
        float xn = xn0 * exp(-t_sec / 900.0f);
        float eps_neutron = 3.2e14f * xn;
        float time_term = 0.5f - atan((t_sec - 1.3f) / 0.11f) / M_PI_F;
        if (time_term < 1e-30f) time_term = 1e-30f;
        float eps_rp = 2e18f * eth * pow(time_term, 1.3f);
        float l_heat = m_ej * (eps_neutron + eps_rp) / METZGER_SCALE;

        // Opacity
        float xr = 1.0f - xn0;
        float xn_decayed = xn0 - xn;
        float kappa_eff = 0.4f * xn_decayed + kappa_r * xr;

        // Diffusion time
        float t_diff = 3.0f * kappa_eff * m_ej / (4.0f * M_PI_F * C_CGS_VAL * v * t_sec) + r / C_CGS_VAL;

        // Radiative luminosity
        float l_rad = (e_th > 0.0f && t_diff > 0.0f) ? e_th / t_diff : 0.0f;
        grid_lrad[i] = l_rad;

        // PdV work
        float l_pdv = (r > 0.0f) ? e_th * v / r : 0.0f;

        // Euler step
        if (i < METZGER_NGRID - 1) {
            float t_next = exp(log_t_min + (log_t_max - log_t_min) * float(i + 1) / float(METZGER_NGRID - 1));
            float dt_sec = (t_next - t_day) * SECS_DAY;
            e_th += (l_heat - l_pdv - l_rad) * dt_sec;
            if (e_th < 0.0f) e_th = 0.0f;
            e_kin += l_pdv * dt_sec;
            v = sqrt(2.0f * e_kin * METZGER_SCALE / m_ej);
            if (v > C_CGS_VAL) v = C_CGS_VAL;
            r += v * dt_sec;
        }
    }

    // Find peak luminosity for normalization
    float l_peak = -1e38f;
    for (int i = 0; i < METZGER_NGRID; i++) {
        if (grid_lrad[i] > l_peak) l_peak = grid_lrad[i];
    }

    if (l_peak <= 0.0f || !isfinite(l_peak)) {
        for (int i = 0; i < n_times; i++)
            out[draw * n_times + i] = 0.0f;
        return;
    }

    // Normalize
    for (int i = 0; i < METZGER_NGRID; i++) {
        grid_lrad[i] /= l_peak;
    }

    // Interpolate for each observation time
    for (int ti = 0; ti < n_times; ti++) {
        float phase = times[ti] - t0;
        float val = 0.0f;

        if (phase <= 0.0f) {
            val = 0.0f;
        } else if (phase <= grid_t[0]) {
            val = grid_lrad[0];
        } else if (phase >= grid_t[METZGER_NGRID - 1]) {
            val = grid_lrad[METZGER_NGRID - 1];
        } else {
            // Binary search for interval
            int lo = 0, hi = METZGER_NGRID - 1;
            while (hi - lo > 1) {
                int mid = (lo + hi) / 2;
                if (grid_t[mid] < phase) lo = mid;
                else hi = mid;
            }
            float frac = (phase - grid_t[lo]) / (grid_t[hi] - grid_t[lo]);
            val = grid_lrad[lo] + frac * (grid_lrad[hi] - grid_lrad[lo]);
        }

        out[draw * n_times + ti] = val;
    }
}

// ===========================================================================
// Batch PSO cost kernel
// ===========================================================================
//
// One thread per (source, particle) pair. Each thread loops over its
// source's observations and computes the negative log-likelihood (divided
// by n_obs) -- matching PsoCost::cost on the CPU side.
//
// Buffer layout:
//   buffer(0):  all_times      [total_obs]
//   buffer(1):  all_flux       [total_obs]
//   buffer(2):  all_obs_var    [total_obs]
//   buffer(3):  all_is_upper   [total_obs]  (int 0/1)
//   buffer(4):  all_upper_flux [total_obs]
//   buffer(5):  source_offsets [n_sources + 1]
//   buffer(6):  positions      [n_sources * n_particles * n_params]
//   buffer(7):  costs          [n_sources * n_particles]
//   buffer(8):  n_sources
//   buffer(9):  n_particles
//   buffer(10): n_params
//   buffer(11): model_id

// Max params for thread-local copy (enough for all models)
#define MAX_PSO_PARAMS 16

kernel void batch_pso_cost(
    device const float* all_times      [[buffer(0)]],
    device const float* all_flux       [[buffer(1)]],
    device const float* all_obs_var    [[buffer(2)]],
    device const int*   all_is_upper   [[buffer(3)]],
    device const float* all_upper_flux [[buffer(4)]],
    device const int*   source_offsets [[buffer(5)]],
    device const float* positions      [[buffer(6)]],
    device float*       costs          [[buffer(7)]],
    device const int&   n_sources      [[buffer(8)]],
    device const int&   n_particles    [[buffer(9)]],
    device const int&   n_params       [[buffer(10)]],
    device const int&   model_id       [[buffer(11)]],
    uint gid [[thread_position_in_grid]])
{
    int idx = int(gid);
    if (idx >= n_sources * n_particles) return;

    int src = idx / n_particles;
    int pid = idx % n_particles;

    int obs_start = source_offsets[src];
    int obs_end   = source_offsets[src + 1];
    int n_obs     = obs_end - obs_start;
    if (n_obs <= 0) { costs[idx] = 1e30f; return; }

    // Copy params to thread-local storage
    int p_offset = (src * n_particles + pid) * n_params;
    float local_p[MAX_PSO_PARAMS];
    for (int j = 0; j < n_params && j < MAX_PSO_PARAMS; j++)
        local_p[j] = positions[p_offset + j];

    int se_idx = n_params - 1; // sigma_extra is always the last parameter
    float sigma_extra = exp(local_p[se_idx]);
    float sigma_extra_sq = sigma_extra * sigma_extra;

    float neg_ll = 0.0f;

    for (int i = obs_start; i < obs_end; i++) {
        float pred = lc_eval_model_at(model_id, local_p, all_times[i]);
        if (!isfinite(pred)) { costs[idx] = 1e30f; return; }

        float total_var = all_obs_var[i] + sigma_extra_sq;

        if (all_is_upper[i]) {
            float z = (all_upper_flux[i] - pred) / sqrt(total_var);
            neg_ll -= lc_log_normal_cdf_d(z);
        } else {
            float diff = pred - all_flux[i];
            neg_ll += diff * diff / total_var + log(total_var);
        }
    }

    costs[idx] = neg_ll / float(n_obs);
}

// ===========================================================================
// Batch PSO cost kernel for MultiBazin
// ===========================================================================
//
// Thread grid: n_sources x n_particles.
// Each source can have a different K, passed via source_k[].
// n_params must equal 4*max_k + 2 (positions are padded).
//
// Buffer layout:
//   buffer(0):  all_times      [total_obs]
//   buffer(1):  all_flux       [total_obs]
//   buffer(2):  all_obs_var    [total_obs]
//   buffer(3):  all_is_upper   [total_obs]  (int 0/1)
//   buffer(4):  all_upper_flux [total_obs]
//   buffer(5):  source_offsets [n_sources + 1]
//   buffer(6):  positions      [n_sources * n_particles * n_params]
//   buffer(7):  costs          [n_sources * n_particles]
//   buffer(8):  source_k       [n_sources]
//   buffer(9):  n_sources
//   buffer(10): n_particles
//   buffer(11): n_params

// Max params for MultiBazin: 4*max_k + 2; support up to k=5 => 22
#define MAX_MB_PARAMS 24

kernel void batch_pso_cost_multi_bazin(
    device const float* all_times      [[buffer(0)]],
    device const float* all_flux       [[buffer(1)]],
    device const float* all_obs_var    [[buffer(2)]],
    device const int*   all_is_upper   [[buffer(3)]],
    device const float* all_upper_flux [[buffer(4)]],
    device const int*   source_offsets [[buffer(5)]],
    device const float* positions      [[buffer(6)]],
    device float*       costs          [[buffer(7)]],
    device const int*   source_k       [[buffer(8)]],
    device const int&   n_sources      [[buffer(9)]],
    device const int&   n_particles    [[buffer(10)]],
    device const int&   n_params       [[buffer(11)]],
    uint gid [[thread_position_in_grid]])
{
    int idx = int(gid);
    if (idx >= n_sources * n_particles) return;

    int src = idx / n_particles;
    int pid = idx % n_particles;

    int obs_start = source_offsets[src];
    int obs_end   = source_offsets[src + 1];
    int n_obs     = obs_end - obs_start;
    if (n_obs <= 0) { costs[idx] = 1e30f; return; }

    int k = source_k[src];

    // Copy params to thread-local storage
    int p_offset = (src * n_particles + pid) * n_params;
    float local_p[MAX_MB_PARAMS];
    for (int j = 0; j < n_params && j < MAX_MB_PARAMS; j++)
        local_p[j] = positions[p_offset + j];

    int se_idx = k * MB_COMP_PARAMS + 1; // sigma_extra is after B
    float sigma_extra = exp(local_p[se_idx]);
    float sigma_extra_sq = sigma_extra * sigma_extra;

    float neg_ll = 0.0f;
    for (int i = obs_start; i < obs_end; i++) {
        float pred = multi_bazin_at(local_p, k, all_times[i]);
        if (!isfinite(pred)) { costs[idx] = 1e30f; return; }

        float total_var = all_obs_var[i] + sigma_extra_sq;

        if (all_is_upper[i]) {
            float z = (all_upper_flux[i] - pred) / sqrt(total_var);
            neg_ll -= lc_log_normal_cdf_d(z);
        } else {
            float diff = pred - all_flux[i];
            neg_ll += diff * diff / total_var + log(total_var);
        }
    }

    costs[idx] = neg_ll / float(n_obs);
}
