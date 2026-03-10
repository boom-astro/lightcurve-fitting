// CUDA kernels for parametric lightcurve model evaluation and batch PSO cost.
//
// Two kernel families:
//   1. *_eval: forward model evaluation (draw × time → flux)
//   2. batch_pso_cost: fused model eval + likelihood for batch PSO fitting
//      Thread grid: n_sources × n_particles. Each thread loops over its
//      source's observations.

#include <math.h>

// ===========================================================================
// Device helpers
// ===========================================================================

__device__ inline double softplus(double x) {
    return log(1.0 + exp(x)) + 1e-6;
}

__device__ inline double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

__device__ inline double log_normal_cdf_d(double x) {
    if (x > 8.0) return 0.0;
    if (x < -30.0) return -0.5 * x * x - 0.5 * log(2.0 * M_PI) - log(-x);
    double z = -x * M_SQRT1_2;
    double az = fabs(z);
    double t = 1.0 / (1.0 + 0.3275911 * az);
    double poly = t * (0.254829592
        + t * (-0.284496736
        + t * (1.421413741
        + t * (-1.453152027
        + t * 1.061405429))));
    double erfc_z = poly * exp(-z * z);
    double phi = (z >= 0.0) ? 0.5 * erfc_z : 1.0 - 0.5 * erfc_z;
    return log(fmax(phi, 1e-300));
}

// ===========================================================================
// Device model evaluation functions
// ===========================================================================

// Model IDs (must match GpuModelName::model_id() in Rust):
//   0=Bazin, 1=Villar, 2=TDE, 3=Arnett, 4=Magnetar, 5=ShockCooling, 6=Afterglow, 7=MetzgerKN

__device__ inline double bazin_at(const double* p, double t) {
    double a        = exp(p[0]);
    double b        = p[1];
    double t0       = p[2];
    double tau_rise = exp(p[3]);
    double tau_fall = exp(p[4]);
    double dt       = t - t0;
    return a * exp(-dt / tau_fall) * sigmoid(dt / tau_rise) + b;
}

__device__ inline double villar_at(const double* p, double t) {
    double a        = exp(p[0]);
    double beta     = p[1];
    double gamma    = exp(p[2]);
    double t0       = p[3];
    double tau_rise = exp(p[4]);
    double tau_fall = exp(p[5]);
    double phase    = t - t0;
    double sig_rise = sigmoid(phase / tau_rise);
    double w        = sigmoid(10.0 * (phase - gamma));
    double piece_left  = 1.0 - beta * phase;
    double piece_right = (1.0 - beta * gamma) * exp((gamma - phase) / tau_fall);
    return a * sig_rise * ((1.0 - w) * piece_left + w * piece_right);
}

__device__ inline double tde_at(const double* p, double t) {
    double a        = exp(p[0]);
    double b        = p[1];
    double t0       = p[2];
    double tau_rise = exp(p[3]);
    double tau_fall = exp(p[4]);
    double alpha    = p[5];
    double phase    = t - t0;
    double sig      = sigmoid(phase / tau_rise);
    double ps       = softplus(phase);
    return a * sig * pow(1.0 + ps / tau_fall, -alpha) + b;
}

__device__ inline double arnett_at(const double* p, double t) {
    double a       = exp(p[0]);
    double t0      = p[1];
    double tau_m   = exp(p[2]);
    double logit_f = p[3];
    double ps      = softplus(t - t0);
    double f       = sigmoid(logit_f);
    double e_ni    = exp(-ps / 8.8);
    double e_co    = exp(-ps / 111.3);
    double heat    = f * e_ni + (1.0 - f) * e_co;
    double x       = ps / tau_m;
    return a * heat * (1.0 - exp(-x * x));
}

__device__ inline double magnetar_at(const double* p, double t) {
    double a        = exp(p[0]);
    double t0       = p[1];
    double tau_sd   = exp(p[2]);
    double tau_diff = exp(p[3]);
    double ps       = softplus(t - t0);
    double w        = 1.0 + ps / tau_sd;
    double x        = ps / tau_diff;
    return a * (1.0 / (w * w)) * (1.0 - exp(-x * x));
}

__device__ inline double shock_cooling_at(const double* p, double t) {
    double a      = exp(p[0]);
    double t0     = p[1];
    double n_exp  = p[2];
    double tau_tr = exp(p[3]);
    double phase  = t - t0;
    double ps     = softplus(phase);
    double ratio  = ps / tau_tr;
    return a * sigmoid(phase * 5.0) * pow(ps, -n_exp) * exp(-ratio * ratio);
}

__device__ inline double afterglow_at(const double* p, double t) {
    double a      = exp(p[0]);
    double t0     = p[1];
    double t_b    = exp(p[2]);
    double alpha1 = p[3];
    double alpha2 = p[4];
    double ps     = softplus(t - t0);
    double ln_r   = log(ps / t_b);
    double u1     = exp(2.0 * alpha1 * ln_r);
    double u2     = exp(2.0 * alpha2 * ln_r);
    return a * pow(u1 + u2, -0.5);
}

// ===========================================================================
// Metzger 1-zone kilonova ODE (sequential — one thread per draw)
// ===========================================================================

// Physical constants
#define MSUN_CGS  1.989e33
#define C_CGS_VAL 2.998e10
#define SECS_DAY  86400.0
#define METZGER_NGRID 200

// Metzger kernel: one thread per draw, loops over all times.
// params layout per draw: [log10_mej, log10_vej, log10_kappa, t0, sigma_extra]
// Output: out[draw * n_times + ti] = normalized flux at times[ti]
extern "C" __global__ void metzger_kn_eval(
    const double* __restrict__ params,
    const double* __restrict__ times,
    double* __restrict__ out,
    int n_draws, int n_times, int n_params)
{
    int draw = blockIdx.x * blockDim.x + threadIdx.x;
    if (draw >= n_draws) return;

    const double* p = params + draw * n_params;
    double m_ej   = pow(10.0, p[0]) * MSUN_CGS;
    double v_ej   = pow(10.0, p[1]) * C_CGS_VAL;
    double kappa_r = pow(10.0, p[2]);
    double t0     = p[3];

    // Find max phase needed
    double phase_max = 0.01;
    for (int i = 0; i < n_times; i++) {
        double ph = times[i] - t0;
        if (ph > phase_max) phase_max = ph;
    }

    if (phase_max <= 0.01) {
        for (int i = 0; i < n_times; i++)
            out[draw * n_times + i] = 0.0;
        return;
    }

    // Build log-spaced time grid
    double log_t_min = log(0.01);
    double log_t_max = log(phase_max * 1.05);

    // ODE state
    double ye = 0.1;
    double xn0 = 1.0 - 2.0 * ye;
    double scale = 1e40;
    double e0 = 0.5 * m_ej * v_ej * v_ej;
    double e_th = e0 / scale;
    double e_kin = e0 / scale;
    double v = v_ej;

    double grid_t_prev = exp(log_t_min);
    double r = grid_t_prev * SECS_DAY * v;

    // Store grid values in local arrays (stack-allocated, 200 entries)
    double grid_t[METZGER_NGRID];
    double grid_lrad[METZGER_NGRID];

    for (int i = 0; i < METZGER_NGRID; i++) {
        double t_day = exp(log_t_min + (log_t_max - log_t_min) * (double)i / (double)(METZGER_NGRID - 1));
        grid_t[i] = t_day;
        double t_sec = t_day * SECS_DAY;

        // Thermalization efficiency
        double eth_factor = 0.34 * pow(t_day, 0.74);
        double eth_log_term = (eth_factor > 1e-10) ? log(1.0 + eth_factor) / eth_factor : 1.0;
        double eth = 0.36 * (exp(-0.56 * t_day) + eth_log_term);

        // Heating rate
        double xn = xn0 * exp(-t_sec / 900.0);
        double eps_neutron = 3.2e14 * xn;
        double time_term = 0.5 - atan((t_sec - 1.3) / 0.11) / M_PI;
        if (time_term < 1e-30) time_term = 1e-30;
        double eps_rp = 2e18 * eth * pow(time_term, 1.3);
        double l_heat = m_ej * (eps_neutron + eps_rp) / scale;

        // Opacity
        double xr = 1.0 - xn0;
        double xn_decayed = xn0 - xn;
        double kappa_eff = 0.4 * xn_decayed + kappa_r * xr;

        // Diffusion time
        double t_diff = 3.0 * kappa_eff * m_ej / (4.0 * M_PI * C_CGS_VAL * v * t_sec) + r / C_CGS_VAL;

        // Radiative luminosity
        double l_rad = (e_th > 0.0 && t_diff > 0.0) ? e_th / t_diff : 0.0;
        grid_lrad[i] = l_rad;

        // PdV work
        double l_pdv = (r > 0.0) ? e_th * v / r : 0.0;

        // Euler step
        if (i < METZGER_NGRID - 1) {
            double t_next = exp(log_t_min + (log_t_max - log_t_min) * (double)(i + 1) / (double)(METZGER_NGRID - 1));
            double dt_sec = (t_next - t_day) * SECS_DAY;
            e_th += (l_heat - l_pdv - l_rad) * dt_sec;
            if (e_th < 0.0) e_th = 0.0;
            e_kin += l_pdv * dt_sec;
            v = sqrt(2.0 * e_kin * scale / m_ej);
            if (v > C_CGS_VAL) v = C_CGS_VAL;
            r += v * dt_sec;
        }
    }

    // Find peak luminosity for normalization
    double l_peak = -1e300;
    for (int i = 0; i < METZGER_NGRID; i++) {
        if (grid_lrad[i] > l_peak) l_peak = grid_lrad[i];
    }

    if (l_peak <= 0.0 || !isfinite(l_peak)) {
        for (int i = 0; i < n_times; i++)
            out[draw * n_times + i] = 0.0;
        return;
    }

    // Normalize
    for (int i = 0; i < METZGER_NGRID; i++) {
        grid_lrad[i] /= l_peak;
    }

    // Interpolate for each observation time
    for (int ti = 0; ti < n_times; ti++) {
        double phase = times[ti] - t0;
        double val = 0.0;

        if (phase <= 0.0) {
            val = 0.0;
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
            double frac = (phase - grid_t[lo]) / (grid_t[hi] - grid_t[lo]);
            val = grid_lrad[lo] + frac * (grid_lrad[hi] - grid_lrad[lo]);
        }

        out[draw * n_times + ti] = val;
    }
}

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

__device__ inline double bazin_component_at(const double* p, double t) {
    double a        = exp(p[0]);
    double t0       = p[1];
    double tau_rise = exp(p[2]);
    double tau_fall = exp(p[3]);
    double dt       = t - t0;
    return a * exp(-dt / tau_fall) * sigmoid(dt / tau_rise);
}

__device__ inline double multi_bazin_at(const double* p, int k, double t) {
    double sum = 0.0;
    for (int c = 0; c < k; c++) {
        sum += bazin_component_at(p + c * MB_COMP_PARAMS, t);
    }
    int b_idx = k * MB_COMP_PARAMS;
    return sum + p[b_idx]; // + B
}

// Batch PSO cost kernel for MultiBazin.
// Thread grid: n_sources × n_particles.
// Each source can have a different K, passed via source_k[].
// n_params must equal 4*max_k + 2 (positions are padded).
extern "C" __global__ void batch_pso_cost_multi_bazin(
    const double* __restrict__ all_times,
    const double* __restrict__ all_flux,
    const double* __restrict__ all_obs_var,
    const int*    __restrict__ all_is_upper,
    const double* __restrict__ all_upper_flux,
    const int*    __restrict__ source_offsets,
    const double* __restrict__ positions,
    double* __restrict__ costs,
    const int*    __restrict__ source_k,
    int n_sources,
    int n_particles,
    int n_params)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_sources * n_particles) return;

    int src = idx / n_particles;
    int pid = idx % n_particles;

    int obs_start = source_offsets[src];
    int obs_end   = source_offsets[src + 1];
    int n_obs     = obs_end - obs_start;
    if (n_obs <= 0) { costs[idx] = 1e99; return; }

    int k = source_k[src];
    const double* p = positions + (long long)(src * n_particles + pid) * n_params;
    int se_idx = k * MB_COMP_PARAMS + 1;  // sigma_extra is after B
    double sigma_extra = exp(p[se_idx]);
    double sigma_extra_sq = sigma_extra * sigma_extra;

    double neg_ll = 0.0;
    for (int i = obs_start; i < obs_end; i++) {
        double pred = multi_bazin_at(p, k, all_times[i]);
        if (!isfinite(pred)) { costs[idx] = 1e99; return; }

        double total_var = all_obs_var[i] + sigma_extra_sq;

        if (all_is_upper[i]) {
            double z = (all_upper_flux[i] - pred) / sqrt(total_var);
            neg_ll -= log_normal_cdf_d(z);
        } else {
            double diff = pred - all_flux[i];
            neg_ll += diff * diff / total_var + log(total_var);
        }
    }

    costs[idx] = neg_ll / (double)n_obs;
}

__device__ inline double eval_model_at(int model_id, const double* p, double t) {
    switch (model_id) {
        case 0: return bazin_at(p, t);
        case 1: return villar_at(p, t);
        case 2: return tde_at(p, t);
        case 3: return arnett_at(p, t);
        case 4: return magnetar_at(p, t);
        case 5: return shock_cooling_at(p, t);
        case 6: return afterglow_at(p, t);
        // case 7 (MetzgerKN) handled by separate kernel
        default: return 0.0;
    }
}

// ===========================================================================
// Forward evaluation kernels (one thread per draw×time)
// ===========================================================================

#define DEFINE_EVAL_KERNEL(name, model_fn)                                    \
extern "C" __global__ void name(                                             \
    const double* __restrict__ params,                                       \
    const double* __restrict__ times,                                        \
    double* __restrict__ out,                                                \
    int n_draws, int n_times, int n_params)                                  \
{                                                                            \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                         \
    if (idx >= n_draws * n_times) return;                                    \
    int draw = idx / n_times;                                                \
    int ti   = idx % n_times;                                                \
    out[idx] = model_fn(params + draw * n_params, times[ti]);                \
}

DEFINE_EVAL_KERNEL(bazin_eval,          bazin_at)
DEFINE_EVAL_KERNEL(villar_eval,         villar_at)
DEFINE_EVAL_KERNEL(tde_eval,            tde_at)
DEFINE_EVAL_KERNEL(arnett_eval,         arnett_at)
DEFINE_EVAL_KERNEL(magnetar_eval,       magnetar_at)
DEFINE_EVAL_KERNEL(shock_cooling_eval,  shock_cooling_at)
DEFINE_EVAL_KERNEL(afterglow_eval,      afterglow_at)

// ===========================================================================
// Batch PSO cost kernel
// ===========================================================================
//
// One thread per (source, particle) pair.  Each thread loops over its
// source's observations and computes the negative log-likelihood (divided
// by n_obs) — matching PsoCost::cost on the CPU side.
//
// Data layout:
//   all_times, all_flux, all_obs_var, all_upper_flux: concatenated across
//       sources.  source_offsets[src] .. source_offsets[src+1] gives the
//       range for source `src`.
//   all_is_upper: int (0/1) array, same layout.
//   positions: [n_sources × n_particles × n_params], row-major.
//   costs (output): [n_sources × n_particles].

// Device function: solve MetzgerKN ODE and evaluate at a single time.
// This is used by batch_pso_cost for model_id=7.
// It solves the full ODE each time (inefficient but correct for PSO where
// each particle has different params).
__device__ double metzger_kn_at_single(const double* p, double obs_time) {
    double m_ej   = pow(10.0, p[0]) * MSUN_CGS;
    double v_ej   = pow(10.0, p[1]) * C_CGS_VAL;
    double kappa_r = pow(10.0, p[2]);
    double t0     = p[3];
    double phase  = obs_time - t0;

    if (phase <= 0.0) return 0.0;

    // Build grid up to this phase
    double phase_max = phase * 1.05;
    if (phase_max < 0.02) phase_max = 0.02;
    double log_t_min = log(0.01);
    double log_t_max = log(phase_max);
    int ngrid = 100; // Reduced grid for PSO (speed vs accuracy tradeoff)

    double ye = 0.1;
    double xn0 = 1.0 - 2.0 * ye;
    double scale = 1e40;
    double e0 = 0.5 * m_ej * v_ej * v_ej;
    double e_th = e0 / scale;
    double e_kin = e0 / scale;
    double v = v_ej;

    double grid_t_0 = exp(log_t_min);
    double r = grid_t_0 * SECS_DAY * v;

    double l_peak = 0.0;
    double l_at_phase = 0.0;
    double prev_t = grid_t_0;
    double prev_lrad = 0.0;

    for (int i = 0; i < ngrid; i++) {
        double t_day = exp(log_t_min + (log_t_max - log_t_min) * (double)i / (double)(ngrid - 1));
        double t_sec = t_day * SECS_DAY;

        double eth_factor = 0.34 * pow(t_day, 0.74);
        double eth_log_term = (eth_factor > 1e-10) ? log(1.0 + eth_factor) / eth_factor : 1.0;
        double eth = 0.36 * (exp(-0.56 * t_day) + eth_log_term);

        double xn = xn0 * exp(-t_sec / 900.0);
        double eps_neutron = 3.2e14 * xn;
        double time_term = 0.5 - atan((t_sec - 1.3) / 0.11) / M_PI;
        if (time_term < 1e-30) time_term = 1e-30;
        double eps_rp = 2e18 * eth * pow(time_term, 1.3);
        double l_heat = m_ej * (eps_neutron + eps_rp) / scale;

        double xr = 1.0 - xn0;
        double xn_decayed = xn0 - xn;
        double kappa_eff = 0.4 * xn_decayed + kappa_r * xr;
        double t_diff = 3.0 * kappa_eff * m_ej / (4.0 * M_PI * C_CGS_VAL * v * t_sec) + r / C_CGS_VAL;

        double l_rad = (e_th > 0.0 && t_diff > 0.0) ? e_th / t_diff : 0.0;
        if (l_rad > l_peak) l_peak = l_rad;

        // Interpolate if phase falls in this interval
        if (i > 0 && phase >= prev_t && phase <= t_day) {
            double frac = (phase - prev_t) / (t_day - prev_t);
            l_at_phase = prev_lrad + frac * (l_rad - prev_lrad);
        }
        if (i == ngrid - 1 && phase >= t_day) {
            l_at_phase = l_rad;
        }
        if (i == 0 && phase <= t_day) {
            l_at_phase = l_rad;
        }

        prev_t = t_day;
        prev_lrad = l_rad;

        double l_pdv = (r > 0.0) ? e_th * v / r : 0.0;
        if (i < ngrid - 1) {
            double t_next = exp(log_t_min + (log_t_max - log_t_min) * (double)(i + 1) / (double)(ngrid - 1));
            double dt_sec = (t_next - t_day) * SECS_DAY;
            e_th += (l_heat - l_pdv - l_rad) * dt_sec;
            if (e_th < 0.0) e_th = 0.0;
            e_kin += l_pdv * dt_sec;
            v = sqrt(2.0 * e_kin * scale / m_ej);
            if (v > C_CGS_VAL) v = C_CGS_VAL;
            r += v * dt_sec;
        }
    }

    if (l_peak <= 0.0 || !isfinite(l_peak)) return 0.0;
    return l_at_phase / l_peak;
}

extern "C" __global__ void batch_pso_cost(
    const double* __restrict__ all_times,
    const double* __restrict__ all_flux,
    const double* __restrict__ all_obs_var,
    const int*    __restrict__ all_is_upper,
    const double* __restrict__ all_upper_flux,
    const int*    __restrict__ source_offsets,
    const double* __restrict__ positions,
    double* __restrict__ costs,
    int n_sources,
    int n_particles,
    int n_params,
    int model_id)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_sources * n_particles) return;

    int src = idx / n_particles;
    int pid = idx % n_particles;

    int obs_start = source_offsets[src];
    int obs_end   = source_offsets[src + 1];
    int n_obs     = obs_end - obs_start;
    if (n_obs <= 0) { costs[idx] = 1e99; return; }

    const double* p = positions + (long long)(src * n_particles + pid) * n_params;
    int se_idx = n_params - 1;  // sigma_extra is always the last parameter
    double sigma_extra = exp(p[se_idx]);
    double sigma_extra_sq = sigma_extra * sigma_extra;

    double neg_ll = 0.0;

    if (model_id == 7) {
        // MetzgerKN: solve ODE once, evaluate at all observation times
        // For PSO cost, we solve per-observation (less efficient but avoids
        // large stack allocation). Each observation gets its own ODE solve.
        for (int i = obs_start; i < obs_end; i++) {
            double pred = metzger_kn_at_single(p, all_times[i]);
            if (!isfinite(pred)) { costs[idx] = 1e99; return; }

            double total_var = all_obs_var[i] + sigma_extra_sq;

            if (all_is_upper[i]) {
                double z = (all_upper_flux[i] - pred) / sqrt(total_var);
                neg_ll -= log_normal_cdf_d(z);
            } else {
                double diff = pred - all_flux[i];
                neg_ll += diff * diff / total_var + log(total_var);
            }
        }
    } else {
        for (int i = obs_start; i < obs_end; i++) {
            double pred = eval_model_at(model_id, p, all_times[i]);
            if (!isfinite(pred)) { costs[idx] = 1e99; return; }

            double total_var = all_obs_var[i] + sigma_extra_sq;

            if (all_is_upper[i]) {
                double z = (all_upper_flux[i] - pred) / sqrt(total_var);
                neg_ll -= log_normal_cdf_d(z);
            } else {
                double diff = pred - all_flux[i];
                neg_ll += diff * diff / total_var + log(total_var);
            }
        }
    }

    costs[idx] = neg_ll / (double)n_obs;
}

// ===========================================================================
// Host-side launch wrappers (callable from Rust via FFI)
// ===========================================================================

#define DEFINE_LAUNCHER(name, kernel)                                         \
extern "C" void name(                                                        \
    const double* params, const double* times, double* out,                  \
    int n_draws, int n_times, int n_params, int grid, int block)             \
{                                                                            \
    kernel<<<grid, block>>>(params, times, out, n_draws, n_times, n_params); \
}

DEFINE_LAUNCHER(launch_bazin,          bazin_eval)
DEFINE_LAUNCHER(launch_villar,         villar_eval)
DEFINE_LAUNCHER(launch_tde,            tde_eval)
DEFINE_LAUNCHER(launch_arnett,         arnett_eval)
DEFINE_LAUNCHER(launch_magnetar,       magnetar_eval)
DEFINE_LAUNCHER(launch_shock_cooling,  shock_cooling_eval)
DEFINE_LAUNCHER(launch_afterglow,      afterglow_eval)

// MetzgerKN uses a different grid: one thread per draw (not per draw×time)
extern "C" void launch_metzger_kn(
    const double* params, const double* times, double* out,
    int n_draws, int n_times, int n_params, int /*grid*/, int block)
{
    // Recompute grid for n_draws (not n_draws*n_times)
    int g = (n_draws + block - 1) / block;
    metzger_kn_eval<<<g, block>>>(params, times, out, n_draws, n_times, n_params);
}

extern "C" void launch_batch_pso_cost(
    const double* all_times,
    const double* all_flux,
    const double* all_obs_var,
    const int*    all_is_upper,
    const double* all_upper_flux,
    const int*    source_offsets,
    const double* positions,
    double* costs,
    int n_sources,
    int n_particles,
    int n_params,
    int model_id,
    int grid,
    int block)
{
    batch_pso_cost<<<grid, block>>>(
        all_times, all_flux, all_obs_var, all_is_upper, all_upper_flux,
        source_offsets, positions, costs,
        n_sources, n_particles, n_params, model_id);
}

extern "C" void launch_batch_pso_cost_multi_bazin(
    const double* all_times,
    const double* all_flux,
    const double* all_obs_var,
    const int*    all_is_upper,
    const double* all_upper_flux,
    const int*    source_offsets,
    const double* positions,
    double* costs,
    const int*    source_k,
    int n_sources,
    int n_particles,
    int n_params,
    int grid,
    int block)
{
    batch_pso_cost_multi_bazin<<<grid, block>>>(
        all_times, all_flux, all_obs_var, all_is_upper, all_upper_flux,
        source_offsets, positions, costs, source_k,
        n_sources, n_particles, n_params);
}
