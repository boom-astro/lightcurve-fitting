// CUDA kernels for parametric lightcurve model evaluation and batch PSO cost.
//
// Three kernel families:
//   1. *_eval: forward model evaluation (draw × time → flux)
//   2. batch_pso_cost: fused model eval + likelihood for batch PSO fitting
//      Thread grid: n_sources × n_particles. Each thread loops over its
//      source's observations.
//   3. batch_pso_full: GPU-resident PSO — entire optimization on GPU,
//      one block per source, one thread per particle.

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
#define METZGER_PSO_NGRID 100

// Solve MetzgerKN ODE once and store normalized luminosity grid.
// Returns l_peak for normalization check. grid_t[] and grid_lrad[] are
// populated with METZGER_PSO_NGRID entries (normalized by l_peak if > 0).
__device__ double metzger_kn_solve_grid(
    const double* p, double phase_max,
    double* grid_t, double* grid_lrad)
{
    double m_ej    = pow(10.0, p[0]) * MSUN_CGS;
    double v_ej    = pow(10.0, p[1]) * C_CGS_VAL;
    double kappa_r = pow(10.0, p[2]);

    if (phase_max < 0.02) phase_max = 0.02;
    double log_t_min = log(0.01);
    double log_t_max = log(phase_max);

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

    for (int i = 0; i < METZGER_PSO_NGRID; i++) {
        double t_day = exp(log_t_min + (log_t_max - log_t_min) * (double)i / (double)(METZGER_PSO_NGRID - 1));
        double t_sec = t_day * SECS_DAY;
        grid_t[i] = t_day;

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
        grid_lrad[i] = l_rad;
        if (l_rad > l_peak) l_peak = l_rad;

        double l_pdv = (r > 0.0) ? e_th * v / r : 0.0;
        if (i < METZGER_PSO_NGRID - 1) {
            double t_next = exp(log_t_min + (log_t_max - log_t_min) * (double)(i + 1) / (double)(METZGER_PSO_NGRID - 1));
            double dt_sec = (t_next - t_day) * SECS_DAY;
            e_th += (l_heat - l_pdv - l_rad) * dt_sec;
            if (e_th < 0.0) e_th = 0.0;
            e_kin += l_pdv * dt_sec;
            v = sqrt(2.0 * e_kin * scale / m_ej);
            if (v > C_CGS_VAL) v = C_CGS_VAL;
            r += v * dt_sec;
        }
    }

    // Normalize grid
    if (l_peak > 0.0 && isfinite(l_peak)) {
        for (int i = 0; i < METZGER_PSO_NGRID; i++)
            grid_lrad[i] /= l_peak;
    }

    return l_peak;
}

// Interpolate a phase value from the pre-solved ODE grid.
__device__ inline double metzger_kn_interp(
    const double* grid_t, const double* grid_lrad, double phase)
{
    if (phase <= 0.0) return 0.0;
    if (phase <= grid_t[0]) return grid_lrad[0];
    if (phase >= grid_t[METZGER_PSO_NGRID - 1]) return grid_lrad[METZGER_PSO_NGRID - 1];

    // Binary search
    int lo = 0, hi = METZGER_PSO_NGRID - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (grid_t[mid] < phase) lo = mid;
        else hi = mid;
    }
    double frac = (phase - grid_t[lo]) / (grid_t[hi] - grid_t[lo]);
    return grid_lrad[lo] + frac * (grid_lrad[hi] - grid_lrad[lo]);
}

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
    double grid_t[METZGER_NGRID];
    double grid_lrad[METZGER_NGRID];

    double m_ej   = pow(10.0, p[0]) * MSUN_CGS;
    double v_ej   = pow(10.0, p[1]) * C_CGS_VAL;
    double kappa_r = pow(10.0, p[2]);

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
// Batch PSO cost kernel (used by MultiBazin CPU-loop PSO)
// ===========================================================================

extern "C" __global__ void batch_pso_cost(
    const double* __restrict__ all_times,
    const double* __restrict__ all_flux,
    const double* __restrict__ all_obs_var,
    const int*    __restrict__ all_is_upper,
    const double* __restrict__ all_upper_flux,
    const int*    __restrict__ source_offsets,
    const double* __restrict__ positions,
    double* __restrict__ costs,
    const double* __restrict__ prior_centers,
    const double* __restrict__ prior_widths,
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
    int se_idx = n_params - 1;
    double sigma_extra = exp(p[se_idx]);
    double sigma_extra_sq = sigma_extra * sigma_extra;

    double neg_ll = 0.0;

    if (model_id == 7) {
        // MetzgerKN: solve ODE once, then interpolate for all observations
        double t0 = p[3];
        double phase_max = 0.01;
        for (int i = obs_start; i < obs_end; i++) {
            double ph = all_times[i] - t0;
            if (ph > phase_max) phase_max = ph;
        }
        double grid_t[METZGER_PSO_NGRID];
        double grid_lrad[METZGER_PSO_NGRID];
        double l_peak = metzger_kn_solve_grid(p, phase_max * 1.05, grid_t, grid_lrad);
        if (l_peak <= 0.0 || !isfinite(l_peak)) { costs[idx] = 1e99; return; }

        for (int i = obs_start; i < obs_end; i++) {
            double phase = all_times[i] - t0;
            double pred = metzger_kn_interp(grid_t, grid_lrad, phase);
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

    // Add population prior penalty (fused, avoids CPU round-trip)
    if (prior_centers != nullptr && prior_widths != nullptr) {
        double neg_lp = 0.0;
        for (int j = 0; j < n_params; j++) {
            double w = prior_widths[j];
            if (w > 0.0) {
                double z = (p[j] - prior_centers[j]) / w;
                neg_lp += 0.5 * z * z;
            }
        }
        neg_ll += neg_lp / ((double)n_obs * (double)n_obs);
    }

    costs[idx] = neg_ll / (double)n_obs;
}

// ===========================================================================
// GPU-resident PSO kernel: entire optimization runs on GPU
// ===========================================================================
//
// One thread block per source. blockDim.x = n_particles.
// Each thread owns one particle and loops over all PSO iterations.
// Eliminates per-iteration host-device transfers.

#define PSO_MAX_PARAMS 7

// xorshift64 PRNG for velocity updates
__device__ inline unsigned long long pso_xorshift64(unsigned long long* state) {
    unsigned long long x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

__device__ inline double pso_rand_uniform(unsigned long long* rng) {
    return (double)(pso_xorshift64(rng) & 0x1FFFFFFFFFFFFF) / (double)0x1FFFFFFFFFFFFF;
}

extern "C" __global__ void batch_pso_full(
    const double* __restrict__ all_times,
    const double* __restrict__ all_flux,
    const double* __restrict__ all_obs_var,
    const int*    __restrict__ all_is_upper,
    const double* __restrict__ all_upper_flux,
    const int*    __restrict__ source_offsets,
    const double* __restrict__ bounds_lower,
    const double* __restrict__ bounds_upper,
    const double* __restrict__ prior_centers,
    const double* __restrict__ prior_widths,
    double* __restrict__ out_gbest_pos,
    double* __restrict__ out_gbest_cost,
    int n_particles,
    int n_params,
    int model_id,
    int max_iters,
    int stall_iters,
    unsigned long long base_seed)
{
    int src = blockIdx.x;
    int pid = threadIdx.x;
    if (pid >= n_particles) return;

    int obs_start = source_offsets[src];
    int obs_end   = source_offsets[src + 1];
    int n_obs     = obs_end - obs_start;
    if (n_obs <= 0) {
        if (pid == 0) out_gbest_cost[src] = 1e99;
        return;
    }

    // Load bounds into registers
    double lower[PSO_MAX_PARAMS], upper[PSO_MAX_PARAMS], v_max[PSO_MAX_PARAMS];
    for (int d = 0; d < n_params; d++) {
        lower[d] = bounds_lower[d];
        upper[d] = bounds_upper[d];
        v_max[d] = 0.5 * (upper[d] - lower[d]);
    }

    // Per-thread RNG
    unsigned long long rng = base_seed + (unsigned long long)src * 10000ULL + (unsigned long long)pid * 137ULL + 1ULL;
    pso_xorshift64(&rng);
    pso_xorshift64(&rng);

    // Initialize position and velocity
    double pos[PSO_MAX_PARAMS], vel[PSO_MAX_PARAMS], pbest_p[PSO_MAX_PARAMS];
    for (int d = 0; d < n_params; d++) {
        pos[d] = lower[d] + pso_rand_uniform(&rng) * (upper[d] - lower[d]);
        vel[d] = v_max[d] * 0.2 * (2.0 * pso_rand_uniform(&rng) - 1.0);
        pbest_p[d] = pos[d];
    }
    double pbest_cost = 1e99;

    // Shared memory layout
    extern __shared__ char smem[];
    double* s_gbest_pos  = (double*)smem;
    double* s_gbest_cost = s_gbest_pos + n_params;
    double* s_costs      = s_gbest_cost + 1;
    double* s_positions  = s_costs + n_particles;
    int*    s_stall      = (int*)(s_positions + n_particles * n_params);
    int*    s_done       = s_stall + 1;

    if (pid == 0) {
        *s_gbest_cost = 1e99;
        *s_stall = 0;
        *s_done = 0;
    }
    __syncthreads();

    double w_max_pso = 0.9, w_min_pso = 0.4, c1 = 1.5, c2 = 1.5;
    double inv_max_iters = 1.0 / (double)max_iters;
    double prev_gbest_val = 1e99;

    for (int iter = 0; iter < max_iters; iter++) {
        // All threads must reach __syncthreads() together, so check
        // s_done uniformly after a barrier rather than breaking early.
        __syncthreads();
        if (*s_done) break;

        double w = w_max_pso - (w_max_pso - w_min_pso) * (double)iter * inv_max_iters;

        // === Evaluate cost for this particle ===
        double neg_ll = 0.0;

        if (model_id == 7) {
            double t0 = pos[3];
            double phase_max = 0.01;
            for (int i = obs_start; i < obs_end; i++) {
                double ph = all_times[i] - t0;
                if (ph > phase_max) phase_max = ph;
            }
            double grid_t[METZGER_PSO_NGRID], grid_lrad[METZGER_PSO_NGRID];
            double l_peak = metzger_kn_solve_grid(pos, phase_max * 1.05, grid_t, grid_lrad);
            if (l_peak <= 0.0 || !isfinite(l_peak)) {
                neg_ll = 1e99;
            } else {
                double se = exp(pos[n_params - 1]);
                double se_sq = se * se;
                for (int i = obs_start; i < obs_end; i++) {
                    double phase = all_times[i] - t0;
                    double pred = metzger_kn_interp(grid_t, grid_lrad, phase);
                    if (!isfinite(pred)) { neg_ll = 1e99; break; }
                    double total_var = all_obs_var[i] + se_sq;
                    if (all_is_upper[i]) {
                        double z = (all_upper_flux[i] - pred) / sqrt(total_var);
                        neg_ll -= log_normal_cdf_d(z);
                    } else {
                        double diff = pred - all_flux[i];
                        neg_ll += diff * diff / total_var + log(total_var);
                    }
                }
            }
        } else {
            double se = exp(pos[n_params - 1]);
            double se_sq = se * se;
            for (int i = obs_start; i < obs_end; i++) {
                double pred = eval_model_at(model_id, pos, all_times[i]);
                if (!isfinite(pred)) { neg_ll = 1e99; break; }
                double total_var = all_obs_var[i] + se_sq;
                if (all_is_upper[i]) {
                    double z = (all_upper_flux[i] - pred) / sqrt(total_var);
                    neg_ll -= log_normal_cdf_d(z);
                } else {
                    double diff = pred - all_flux[i];
                    neg_ll += diff * diff / total_var + log(total_var);
                }
            }
        }

        double cost = neg_ll / (double)n_obs;

        // Add population prior penalty
        if (prior_centers != nullptr && prior_widths != nullptr) {
            double neg_lp = 0.0;
            for (int j = 0; j < n_params; j++) {
                double pw = prior_widths[j];
                if (pw > 0.0) {
                    double z = (pos[j] - prior_centers[j]) / pw;
                    neg_lp += 0.5 * z * z;
                }
            }
            cost += neg_lp / ((double)n_obs * (double)n_obs);
        }

        // Update personal best
        if (cost < pbest_cost) {
            pbest_cost = cost;
            for (int d = 0; d < n_params; d++) pbest_p[d] = pos[d];
        }

        // === Block-level gbest reduction via shared memory ===
        s_costs[pid] = pbest_cost;
        for (int d = 0; d < n_params; d++)
            s_positions[pid * n_params + d] = pbest_p[d];
        __syncthreads();

        // Thread 0 finds global best
        if (pid == 0) {
            for (int p = 0; p < n_particles; p++) {
                if (s_costs[p] < *s_gbest_cost) {
                    *s_gbest_cost = s_costs[p];
                    for (int d = 0; d < n_params; d++)
                        s_gbest_pos[d] = s_positions[p * n_params + d];
                }
            }

            // Stall detection
            double improved_threshold = 0.01 * fmax(fabs(prev_gbest_val), 1e-10);
            if (prev_gbest_val - *s_gbest_cost > improved_threshold) {
                *s_stall = 0;
                prev_gbest_val = *s_gbest_cost;
            } else {
                *s_stall = *s_stall + 1;
                if (*s_stall >= stall_iters) *s_done = 1;
            }
        }
        __syncthreads();

        // Read gbest into registers for velocity update
        double gbest[PSO_MAX_PARAMS];
        for (int d = 0; d < n_params; d++) gbest[d] = s_gbest_pos[d];

        // === Update velocity and position ===
        for (int d = 0; d < n_params; d++) {
            double r1 = pso_rand_uniform(&rng);
            double r2 = pso_rand_uniform(&rng);
            double v_new = w * vel[d]
                + c1 * r1 * (pbest_p[d] - pos[d])
                + c2 * r2 * (gbest[d] - pos[d]);

            if (v_new > v_max[d]) v_new = v_max[d];
            if (v_new < -v_max[d]) v_new = -v_max[d];

            double new_pos = pos[d] + v_new;

            if (new_pos <= lower[d]) {
                pos[d] = lower[d];
                vel[d] = 0.0;
            } else if (new_pos >= upper[d]) {
                pos[d] = upper[d];
                vel[d] = 0.0;
            } else {
                pos[d] = new_pos;
                vel[d] = v_new;
            }
        }
        __syncthreads();
    }

    // Write final gbest (thread 0 only)
    if (pid == 0) {
        for (int d = 0; d < n_params; d++)
            out_gbest_pos[src * n_params + d] = s_gbest_pos[d];
        out_gbest_cost[src] = *s_gbest_cost;
    }
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
    const double* prior_centers,
    const double* prior_widths,
    int n_sources,
    int n_particles,
    int n_params,
    int model_id,
    int grid,
    int block)
{
    batch_pso_cost<<<grid, block>>>(
        all_times, all_flux, all_obs_var, all_is_upper, all_upper_flux,
        source_offsets, positions, costs, prior_centers, prior_widths,
        n_sources, n_particles, n_params, model_id);
}

extern "C" void launch_batch_pso_full(
    const double* all_times,
    const double* all_flux,
    const double* all_obs_var,
    const int*    all_is_upper,
    const double* all_upper_flux,
    const int*    source_offsets,
    const double* bounds_lower,
    const double* bounds_upper,
    const double* prior_centers,
    const double* prior_widths,
    double* out_gbest_pos,
    double* out_gbest_cost,
    int n_sources,
    int n_particles,
    int n_params,
    int model_id,
    int max_iters,
    int stall_iters,
    unsigned long long base_seed)
{
    size_t smem_bytes = (n_params + 1 + n_particles + n_particles * n_params) * sizeof(double) + 2 * sizeof(int);
    batch_pso_full<<<n_sources, n_particles, smem_bytes>>>(
        all_times, all_flux, all_obs_var, all_is_upper, all_upper_flux,
        source_offsets, bounds_lower, bounds_upper, prior_centers, prior_widths,
        out_gbest_pos, out_gbest_cost,
        n_particles, n_params, model_id, max_iters, stall_iters, base_seed);
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
