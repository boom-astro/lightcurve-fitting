#pragma once
// Shared device-side model evaluation and gradient functions (float32).
// Metal port of cuda/models_device.h.
// Included by models.metal and svi.metal.

#include <metal_stdlib>
using namespace metal;

// ===========================================================================
// Device helpers
// ===========================================================================

inline float lc_softplus(float x) {
    return log(1.0f + exp(x)) + 1e-6f;
}

inline float lc_softplus_grad(float x) {
    // d/dx softplus(x) = sigmoid(x) = exp(x)/(1+exp(x))
    return exp(x) / (1.0f + exp(x));
}

inline float lc_sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

inline float lc_log_normal_cdf_d(float x) {
    if (x > 8.0f) return 0.0f;
    if (x < -20.0f) return -0.5f * x * x - 0.5f * log(2.0f * M_PI_F) - log(-x);
    float z = -x * M_SQRT1_2_F;
    float az = abs(z);
    float t = 1.0f / (1.0f + 0.3275911f * az);
    float poly = t * (0.254829592f
        + t * (-0.284496736f
        + t * (1.421413741f
        + t * (-1.453152027f
        + t * 1.061405429f))));
    float erfc_z = poly * exp(-z * z);
    float phi = (z >= 0.0f) ? 0.5f * erfc_z : 1.0f - 0.5f * erfc_z;
    return log(max(phi, 1e-38f));
}

// erf approximation for CDF gradient
inline float lc_erf_approx(float x) {
    float ax = abs(x);
    float t = 1.0f / (1.0f + 0.3275911f * ax);
    float poly = t * (0.254829592f
        + t * (-0.284496736f
        + t * (1.421413741f
        + t * (-1.453152027f
        + t * 1.061405429f))));
    float val = 1.0f - poly * exp(-ax * ax);
    return (x >= 0.0f) ? val : -val;
}

// ===========================================================================
// Model evaluation functions
// ===========================================================================

// Model IDs: 0=Bazin, 1=Villar, 2=TDE, 3=Arnett, 4=Magnetar,
//            5=ShockCooling, 6=Afterglow, 7=MetzgerKN

inline float lc_bazin_at(const thread float* p, float t) {
    float a        = exp(p[0]);
    float b        = p[1];
    float t0       = p[2];
    float tau_rise = exp(p[3]);
    float tau_fall = exp(p[4]);
    float dt       = t - t0;
    return a * exp(-dt / tau_fall) * lc_sigmoid(dt / tau_rise) + b;
}

inline float lc_villar_at(const thread float* p, float t) {
    float a        = exp(p[0]);
    float beta     = p[1];
    float gamma    = exp(p[2]);
    float t0       = p[3];
    float tau_rise = exp(p[4]);
    float tau_fall = exp(p[5]);
    float phase    = t - t0;
    float sig_rise = lc_sigmoid(phase / tau_rise);
    float w        = lc_sigmoid(10.0f * (phase - gamma));
    float piece_left  = 1.0f - beta * phase;
    float piece_right = (1.0f - beta * gamma) * exp((gamma - phase) / tau_fall);
    return a * sig_rise * ((1.0f - w) * piece_left + w * piece_right);
}

inline float lc_tde_at(const thread float* p, float t) {
    float a        = exp(p[0]);
    float b        = p[1];
    float t0       = p[2];
    float tau_rise = exp(p[3]);
    float tau_fall = exp(p[4]);
    float alpha    = p[5];
    float phase    = t - t0;
    float sig      = lc_sigmoid(phase / tau_rise);
    float ps       = lc_softplus(phase);
    return a * sig * pow(1.0f + ps / tau_fall, -alpha) + b;
}

inline float lc_arnett_at(const thread float* p, float t) {
    float a       = exp(p[0]);
    float t0      = p[1];
    float tau_m   = exp(p[2]);
    float logit_f = p[3];
    float ps      = lc_softplus(t - t0);
    float f       = lc_sigmoid(logit_f);
    float e_ni    = exp(-ps / 8.8f);
    float e_co    = exp(-ps / 111.3f);
    float heat    = f * e_ni + (1.0f - f) * e_co;
    float x       = ps / tau_m;
    return a * heat * (1.0f - exp(-x * x));
}

inline float lc_magnetar_at(const thread float* p, float t) {
    float a        = exp(p[0]);
    float t0       = p[1];
    float tau_sd   = exp(p[2]);
    float tau_diff = exp(p[3]);
    float ps       = lc_softplus(t - t0);
    float w        = 1.0f + ps / tau_sd;
    float x        = ps / tau_diff;
    return a * (1.0f / (w * w)) * (1.0f - exp(-x * x));
}

inline float lc_shock_cooling_at(const thread float* p, float t) {
    float a      = exp(p[0]);
    float t0     = p[1];
    float n_exp  = p[2];
    float tau_tr = exp(p[3]);
    float phase  = t - t0;
    float ps     = lc_softplus(phase);
    float ratio  = ps / tau_tr;
    return a * lc_sigmoid(phase * 5.0f) * pow(ps, -n_exp) * exp(-ratio * ratio);
}

inline float lc_afterglow_at(const thread float* p, float t) {
    float a      = exp(p[0]);
    float t0     = p[1];
    float t_b    = exp(p[2]);
    float alpha1 = p[3];
    float alpha2 = p[4];
    float ps     = lc_softplus(t - t0);
    float ln_r   = log(ps / t_b);
    float u1     = exp(2.0f * alpha1 * ln_r);
    float u2     = exp(2.0f * alpha2 * ln_r);
    return a * pow(u1 + u2, -0.5f);
}

// ===========================================================================
// Analytical gradient functions (d(flux)/d(param_j))
// ===========================================================================

// Bazin: params = [log_a, b, t0, log_tau_rise, log_tau_fall, log_sigma_extra]
inline void lc_bazin_grad(const thread float* p, float t, thread float* out) {
    float a        = exp(p[0]);
    float t0       = p[2];
    float tau_rise = exp(p[3]);
    float tau_fall = exp(p[4]);
    float dt       = t - t0;
    float e_fall   = exp(-dt / tau_fall);
    float sig      = lc_sigmoid(dt / tau_rise);
    float base     = a * e_fall * sig;
    out[0] = base;                                         // d/d(log_a)
    out[1] = 1.0f;                                         // d/d(b)
    out[2] = base * (1.0f/tau_fall - (1.0f-sig)/tau_rise); // d/d(t0)
    out[3] = -base * (1.0f - sig) * dt / tau_rise;        // d/d(log_tau_rise)
    out[4] = base * dt / tau_fall;                         // d/d(log_tau_fall)
    out[5] = 0.0f;                                         // d/d(log_sigma_extra)
}

// Villar: params = [log_a, beta, log_gamma, t0, log_tau_rise, log_tau_fall, log_sigma_extra]
inline void lc_villar_grad(const thread float* p, float t, thread float* out) {
    float a        = exp(p[0]);
    float beta     = p[1];
    float gamma    = exp(p[2]);
    float t0       = p[3];
    float tau_rise = exp(p[4]);
    float tau_fall = exp(p[5]);
    float phase    = t - t0;
    float k        = 10.0f;
    float sig_rise = lc_sigmoid(phase / tau_rise);
    float w        = lc_sigmoid(k * (phase - gamma));
    float piece_left  = 1.0f - beta * phase;
    float e_decay     = exp((gamma - phase) / tau_fall);
    float piece_right = (1.0f - beta * gamma) * e_decay;
    float piece       = (1.0f - w) * piece_left + w * piece_right;
    float flux        = a * sig_rise * piece;

    out[0] = flux;

    float d_pl_dbeta = -phase;
    float d_pr_dbeta = -gamma * e_decay;
    out[1] = a * sig_rise * ((1.0f - w) * d_pl_dbeta + w * d_pr_dbeta);

    float dw_dgamma = -k * w * (1.0f - w);
    float dw_dloggamma = dw_dgamma * gamma;
    float dpr_dgamma = e_decay * (-beta + (1.0f - beta * gamma) / tau_fall);
    float dpr_dloggamma = dpr_dgamma * gamma;
    float d_piece_dloggamma = dw_dloggamma * (piece_right - piece_left) + w * dpr_dloggamma;
    out[2] = a * sig_rise * d_piece_dloggamma;

    float dsig_dphase = sig_rise * (1.0f - sig_rise) / tau_rise;
    float dsig_dt0    = -dsig_dphase;
    float dw_dphase   = k * w * (1.0f - w);
    float dw_dt0      = -dw_dphase;
    float dpl_dt0     = beta;
    float dpr_dt0     = (1.0f - beta * gamma) * e_decay / tau_fall;
    float d_piece_dt0 = dw_dt0 * (piece_right - piece_left) + (1.0f - w) * dpl_dt0 + w * dpr_dt0;
    out[3] = a * (dsig_dt0 * piece + sig_rise * d_piece_dt0);

    out[4] = a * piece * sig_rise * (1.0f - sig_rise) * (-phase / tau_rise);

    float d_pr_dlogtf = piece_right * (phase - gamma) / tau_fall;
    out[5] = a * sig_rise * w * d_pr_dlogtf;

    out[6] = 0.0f;
}

// TDE: params = [log_a, b, t0, log_tau_rise, log_tau_fall, alpha, log_sigma_extra]
inline void lc_tde_grad(const thread float* p, float t, thread float* out) {
    float a        = exp(p[0]);
    float t0       = p[2];
    float tau_rise = exp(p[3]);
    float tau_fall = exp(p[4]);
    float alpha    = p[5];
    float phase    = t - t0;
    float sig      = lc_sigmoid(phase / tau_rise);
    float ps       = lc_softplus(phase);
    float sig_p    = lc_softplus_grad(phase);
    float w        = 1.0f + ps / tau_fall;
    float decay    = pow(w, -alpha);
    float base     = a * sig * decay;

    out[0] = base;
    out[1] = 1.0f;
    float dsig_dt0   = -sig * (1.0f - sig) / tau_rise;
    float ddecay_dt0 = alpha * pow(w, -alpha - 1.0f) * sig_p / tau_fall;
    out[2] = a * (dsig_dt0 * decay + sig * ddecay_dt0);
    out[3] = a * decay * (-sig * (1.0f - sig) * phase / tau_rise);
    out[4] = a * sig * alpha * pow(w, -alpha - 1.0f) * ps / tau_fall;
    out[5] = a * sig * (-log(w)) * decay;
    out[6] = 0.0f;
}

// Arnett: params = [log_a, t0, log_tau_m, logit_f, log_sigma_extra]
inline void lc_arnett_grad(const thread float* p, float t, thread float* out) {
    float a       = exp(p[0]);
    float t0      = p[1];
    float tau_m   = exp(p[2]);
    float logit_f = p[3];
    float phase   = t - t0;
    float ps      = lc_softplus(phase);
    float sig_p   = lc_softplus_grad(phase);
    float f       = lc_sigmoid(logit_f);
    float e_ni    = exp(-ps / 8.8f);
    float e_co    = exp(-ps / 111.3f);
    float heat    = f * e_ni + (1.0f - f) * e_co;
    float x       = ps / tau_m;
    float exp_x2  = exp(-x * x);
    float trap    = 1.0f - exp_x2;
    float flux    = a * heat * trap;

    out[0] = flux;
    float dheat_dps = -f * e_ni / 8.8f - (1.0f - f) * e_co / 111.3f;
    float dtrap_dps = 2.0f * ps * exp_x2 / (tau_m * tau_m);
    out[1] = a * (-sig_p) * (dheat_dps * trap + heat * dtrap_dps);
    out[2] = -2.0f * a * heat * exp_x2 * x * x;
    out[3] = a * trap * (e_ni - e_co) * f * (1.0f - f);
    out[4] = 0.0f;
}

// Magnetar: params = [log_a, t0, log_tau_sd, log_tau_diff, log_sigma_extra]
inline void lc_magnetar_grad(const thread float* p, float t, thread float* out) {
    float a        = exp(p[0]);
    float t0       = p[1];
    float tau_sd   = exp(p[2]);
    float tau_diff = exp(p[3]);
    float phase    = t - t0;
    float ps       = lc_softplus(phase);
    float sig_p    = lc_softplus_grad(phase);
    float w        = 1.0f + ps / tau_sd;
    float w3       = w * w * w;
    float spindown = 1.0f / (w * w);
    float x        = ps / tau_diff;
    float exp_x2   = exp(-x * x);
    float trap     = 1.0f - exp_x2;
    float flux     = a * spindown * trap;

    out[0] = flux;
    float dspindown_dps = -2.0f / (w3 * tau_sd);
    float dtrap_dps     = 2.0f * ps * exp_x2 / (tau_diff * tau_diff);
    out[1] = a * (-sig_p) * (dspindown_dps * trap + spindown * dtrap_dps);
    out[2] = a * trap * 2.0f * ps / (w3 * tau_sd);
    out[3] = -2.0f * a * spindown * exp_x2 * x * x;
    out[4] = 0.0f;
}

// ShockCooling: params = [log_a, t0, n, log_tau_tr, log_sigma_extra]
inline void lc_shock_cooling_grad(const thread float* p, float t, thread float* out) {
    float a      = exp(p[0]);
    float t0     = p[1];
    float n      = p[2];
    float tau_tr = exp(p[3]);
    float phase  = t - t0;
    float sig5   = lc_sigmoid(phase * 5.0f);
    float ps     = lc_softplus(phase);
    float sig_p  = lc_softplus_grad(phase);
    float cooling = pow(ps, -n);
    float ratio   = ps / tau_tr;
    float cutoff  = exp(-ratio * ratio);
    float base    = cooling * cutoff;
    float flux    = a * sig5 * base;

    out[0] = flux;
    out[1] = a * base * (-5.0f * sig5 * (1.0f - sig5)
             + sig5 * sig_p * (n / ps + 2.0f * ps / (tau_tr * tau_tr)));
    out[2] = -flux * log(ps);
    out[3] = flux * 2.0f * ratio * ratio;
    out[4] = 0.0f;
}

// Afterglow: params = [log_a, t0, log_t_b, alpha1, alpha2, log_sigma_extra]
inline void lc_afterglow_grad(const thread float* p, float t, thread float* out) {
    float a      = exp(p[0]);
    float t0     = p[1];
    float t_b    = exp(p[2]);
    float alpha1 = p[3];
    float alpha2 = p[4];
    float phase  = t - t0;
    float ps     = lc_softplus(phase);
    float sig_p  = lc_softplus_grad(phase);
    float r      = ps / t_b;
    float ln_r   = log(r);
    float u1     = exp(2.0f * alpha1 * ln_r);
    float u2     = exp(2.0f * alpha2 * ln_r);
    float u      = u1 + u2;
    float flux   = a * pow(u, -0.5f);
    float u15    = pow(u, -1.5f);

    out[0] = flux;
    float du_dps   = (2.0f * alpha1 * u1 + 2.0f * alpha2 * u2) / ps;
    float dflux_dps = a * (-0.5f) * u15 * du_dps;
    out[1] = dflux_dps * (-sig_p);
    float du_dlog_tb = -(2.0f * alpha1 * u1 + 2.0f * alpha2 * u2);
    out[2] = a * (-0.5f) * u15 * du_dlog_tb;
    out[3] = a * (-0.5f) * u15 * 2.0f * ln_r * u1;
    out[4] = a * (-0.5f) * u15 * 2.0f * ln_r * u2;
    out[5] = 0.0f;
}

// ===========================================================================
// Gradient dispatch
// ===========================================================================

inline bool lc_eval_model_grad(int model_id, const thread float* p, float t, thread float* out) {
    switch (model_id) {
        case 0: lc_bazin_grad(p, t, out); return true;
        case 1: lc_villar_grad(p, t, out); return true;
        case 2: lc_tde_grad(p, t, out); return true;
        case 3: lc_arnett_grad(p, t, out); return true;
        case 4: lc_magnetar_grad(p, t, out); return true;
        case 5: lc_shock_cooling_grad(p, t, out); return true;
        case 6: lc_afterglow_grad(p, t, out); return true;
        default: return false; // MetzgerKN or unknown — use finite diff
    }
}

// ===========================================================================
// MetzgerKN (ODE-based, no analytical gradient)
// ===========================================================================

// Note: scale reduced from 1e40 (CUDA/float64) to 1e30 for float32 range.
#define LC_MSUN_CGS  1.989e33f
#define LC_C_CGS_VAL 2.998e10f
#define LC_SECS_DAY  86400.0f
#define LC_SCALE     1e30f

inline float lc_metzger_kn_at_single(const thread float* p, float obs_time) {
    float m_ej    = pow(10.0f, p[0]) * LC_MSUN_CGS;
    float v_ej    = pow(10.0f, p[1]) * LC_C_CGS_VAL;
    float kappa_r = pow(10.0f, p[2]);
    float t0      = p[3];
    float phase   = obs_time - t0;

    if (phase <= 0.0f) return 0.0f;

    float phase_max = phase * 1.05f;
    if (phase_max < 0.02f) phase_max = 0.02f;
    float log_t_min = log(0.01f);
    float log_t_max = log(phase_max);
    int ngrid = 100;

    float ye = 0.1f;
    float xn0 = 1.0f - 2.0f * ye;
    float e0 = 0.5f * m_ej * v_ej * v_ej;
    float e_th = e0 / LC_SCALE;
    float e_kin = e0 / LC_SCALE;
    float v = v_ej;

    float grid_t_0 = exp(log_t_min);
    float r = grid_t_0 * LC_SECS_DAY * v;

    float l_peak = 0.0f;
    float l_at_phase = 0.0f;
    float prev_t = grid_t_0;
    float prev_lrad = 0.0f;

    for (int i = 0; i < ngrid; i++) {
        float t_day = exp(log_t_min + (log_t_max - log_t_min) * float(i) / float(ngrid - 1));
        float t_sec = t_day * LC_SECS_DAY;

        float eth_factor = 0.34f * pow(t_day, 0.74f);
        float eth_log_term = (eth_factor > 1e-10f) ? log(1.0f + eth_factor) / eth_factor : 1.0f;
        float eth = 0.36f * (exp(-0.56f * t_day) + eth_log_term);

        float xn = xn0 * exp(-t_sec / 900.0f);
        float eps_neutron = 3.2e14f * xn;
        float time_term = 0.5f - atan((t_sec - 1.3f) / 0.11f) / M_PI_F;
        if (time_term < 1e-30f) time_term = 1e-30f;
        float eps_rp = 2e18f * eth * pow(time_term, 1.3f);
        float l_heat = m_ej * (eps_neutron + eps_rp) / LC_SCALE;

        float xr = 1.0f - xn0;
        float xn_decayed = xn0 - xn;
        float kappa_eff = 0.4f * xn_decayed + kappa_r * xr;
        float t_diff = 3.0f * kappa_eff * m_ej / (4.0f * M_PI_F * LC_C_CGS_VAL * v * t_sec) + r / LC_C_CGS_VAL;

        float l_rad = (e_th > 0.0f && t_diff > 0.0f) ? e_th / t_diff : 0.0f;
        if (l_rad > l_peak) l_peak = l_rad;

        if (i > 0 && phase >= prev_t && phase <= t_day) {
            float frac = (phase - prev_t) / (t_day - prev_t);
            l_at_phase = prev_lrad + frac * (l_rad - prev_lrad);
        }
        if (i == ngrid - 1 && phase >= t_day) l_at_phase = l_rad;
        if (i == 0 && phase <= t_day) l_at_phase = l_rad;

        prev_t = t_day;
        prev_lrad = l_rad;

        float l_pdv = (r > 0.0f) ? e_th * v / r : 0.0f;
        if (i < ngrid - 1) {
            float t_next = exp(log_t_min + (log_t_max - log_t_min) * float(i + 1) / float(ngrid - 1));
            float dt_sec = (t_next - t_day) * LC_SECS_DAY;
            e_th += (l_heat - l_pdv - l_rad) * dt_sec;
            if (e_th < 0.0f) e_th = 0.0f;
            e_kin += l_pdv * dt_sec;
            v = sqrt(2.0f * e_kin * LC_SCALE / m_ej);
            if (v > LC_C_CGS_VAL) v = LC_C_CGS_VAL;
            r += v * dt_sec;
        }
    }

    if (l_peak <= 0.0f || !isfinite(l_peak)) return 0.0f;
    return l_at_phase / l_peak;
}

// Unified model dispatch
inline float lc_eval_model_at(int model_id, const thread float* p, float t) {
    switch (model_id) {
        case 0: return lc_bazin_at(p, t);
        case 1: return lc_villar_at(p, t);
        case 2: return lc_tde_at(p, t);
        case 3: return lc_arnett_at(p, t);
        case 4: return lc_magnetar_at(p, t);
        case 5: return lc_shock_cooling_at(p, t);
        case 6: return lc_afterglow_at(p, t);
        case 7: return lc_metzger_kn_at_single(p, t);
        default: return 0.0f;
    }
}
