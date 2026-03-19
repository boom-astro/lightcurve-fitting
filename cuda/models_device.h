#pragma once
// Shared device-side model evaluation and gradient functions.
// Included by models.cu and svi.cu.

#include <math.h>

// ===========================================================================
// Device helpers
// ===========================================================================

__device__ inline double lc_softplus(double x) {
    return log(1.0 + exp(x)) + 1e-6;
}

__device__ inline double lc_softplus_grad(double x) {
    // d/dx softplus(x) = sigmoid(x) = exp(x)/(1+exp(x))
    return exp(x) / (1.0 + exp(x));
}

// Fused softplus + sigmoid sharing one exp(x) call.
// sp = softplus(x), sg = sigmoid(x) = softplus_grad(x).
__device__ inline void lc_softplus_both(double x, double* sp, double* sg) {
    double ex = exp(x);
    *sp = log(1.0 + ex) + 1e-6;
    *sg = ex / (1.0 + ex);
}

__device__ inline double lc_sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

__device__ inline double lc_log_normal_cdf_d(double x) {
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

// erf approximation for CDF gradient
__device__ inline double lc_erf_approx(double x) {
    double ax = fabs(x);
    double t = 1.0 / (1.0 + 0.3275911 * ax);
    double poly = t * (0.254829592
        + t * (-0.284496736
        + t * (1.421413741
        + t * (-1.453152027
        + t * 1.061405429))));
    double val = 1.0 - poly * exp(-ax * ax);
    return (x >= 0.0) ? val : -val;
}

// ===========================================================================
// Model evaluation functions
// ===========================================================================

// Model IDs: 0=Bazin, 1=Villar, 2=TDE, 3=Arnett, 4=Magnetar,
//            5=ShockCooling, 6=Afterglow, 7=MetzgerKN

__device__ inline double lc_bazin_at(const double* p, double t) {
    double a        = exp(p[0]);
    double b        = p[1];
    double t0       = p[2];
    double tau_rise = exp(p[3]);
    double tau_fall = exp(p[4]);
    double dt       = t - t0;
    return a * exp(-dt / tau_fall) * lc_sigmoid(dt / tau_rise) + b;
}

__device__ inline double lc_villar_at(const double* p, double t) {
    double a        = exp(p[0]);
    double beta     = p[1];
    double gamma    = exp(p[2]);
    double t0       = p[3];
    double tau_rise = exp(p[4]);
    double tau_fall = exp(p[5]);
    double phase    = t - t0;
    double sig_rise = lc_sigmoid(phase / tau_rise);
    double w        = lc_sigmoid(10.0 * (phase - gamma));
    double piece_left  = 1.0 - beta * phase;
    double piece_right = (1.0 - beta * gamma) * exp((gamma - phase) / tau_fall);
    return a * sig_rise * ((1.0 - w) * piece_left + w * piece_right);
}

__device__ inline double lc_tde_at(const double* p, double t) {
    double a        = exp(p[0]);
    double b        = p[1];
    double t0       = p[2];
    double tau_rise = exp(p[3]);
    double tau_fall = exp(p[4]);
    double alpha    = p[5];
    double phase    = t - t0;
    double sig      = lc_sigmoid(phase / tau_rise);
    double ps       = lc_softplus(phase);
    return a * sig * pow(1.0 + ps / tau_fall, -alpha) + b;
}

__device__ inline double lc_arnett_at(const double* p, double t) {
    double a       = exp(p[0]);
    double t0      = p[1];
    double tau_m   = exp(p[2]);
    double logit_f = p[3];
    double ps      = lc_softplus(t - t0);
    double f       = lc_sigmoid(logit_f);
    double e_ni    = exp(-ps / 8.8);
    double e_co    = exp(-ps / 111.3);
    double heat    = f * e_ni + (1.0 - f) * e_co;
    double x       = ps / tau_m;
    return a * heat * (1.0 - exp(-x * x));
}

__device__ inline double lc_magnetar_at(const double* p, double t) {
    double a        = exp(p[0]);
    double t0       = p[1];
    double tau_sd   = exp(p[2]);
    double tau_diff = exp(p[3]);
    double ps       = lc_softplus(t - t0);
    double w        = 1.0 + ps / tau_sd;
    double x        = ps / tau_diff;
    return a * (1.0 / (w * w)) * (1.0 - exp(-x * x));
}

__device__ inline double lc_shock_cooling_at(const double* p, double t) {
    double a      = exp(p[0]);
    double t0     = p[1];
    double n_exp  = p[2];
    double tau_tr = exp(p[3]);
    double phase  = t - t0;
    double ps     = lc_softplus(phase);
    double ratio  = ps / tau_tr;
    return a * lc_sigmoid(phase * 5.0) * pow(ps, -n_exp) * exp(-ratio * ratio);
}

__device__ inline double lc_afterglow_at(const double* p, double t) {
    double a      = exp(p[0]);
    double t0     = p[1];
    double t_b    = exp(p[2]);
    double alpha1 = p[3];
    double alpha2 = p[4];
    double ps     = lc_softplus(t - t0);
    double ln_r   = log(ps / t_b);
    double u1     = exp(2.0 * alpha1 * ln_r);
    double u2     = exp(2.0 * alpha2 * ln_r);
    return a * pow(u1 + u2, -0.5);
}

// ===========================================================================
// Analytical gradient functions (d(flux)/d(param_j))
// ===========================================================================

// Bazin: params = [log_a, b, t0, log_tau_rise, log_tau_fall, log_sigma_extra]
__device__ inline void lc_bazin_grad(const double* p, double t, double* out) {
    double a        = exp(p[0]);
    double t0       = p[2];
    double tau_rise = exp(p[3]);
    double tau_fall = exp(p[4]);
    double dt       = t - t0;
    double e_fall   = exp(-dt / tau_fall);
    double sig      = lc_sigmoid(dt / tau_rise);
    double base     = a * e_fall * sig;
    out[0] = base;                                         // d/d(log_a)
    out[1] = 1.0;                                          // d/d(b)
    out[2] = base * (1.0/tau_fall - (1.0-sig)/tau_rise);   // d/d(t0)
    out[3] = -base * (1.0 - sig) * dt / tau_rise;         // d/d(log_tau_rise)
    out[4] = base * dt / tau_fall;                         // d/d(log_tau_fall)
    out[5] = 0.0;                                          // d/d(log_sigma_extra)
}

// Villar: params = [log_a, beta, log_gamma, t0, log_tau_rise, log_tau_fall, log_sigma_extra]
__device__ inline void lc_villar_grad(const double* p, double t, double* out) {
    double a        = exp(p[0]);
    double beta     = p[1];
    double gamma    = exp(p[2]);
    double t0       = p[3];
    double tau_rise = exp(p[4]);
    double tau_fall = exp(p[5]);
    double phase    = t - t0;
    double k        = 10.0;
    double sig_rise = lc_sigmoid(phase / tau_rise);
    double w        = lc_sigmoid(k * (phase - gamma));
    double piece_left  = 1.0 - beta * phase;
    double e_decay     = exp((gamma - phase) / tau_fall);
    double piece_right = (1.0 - beta * gamma) * e_decay;
    double piece       = (1.0 - w) * piece_left + w * piece_right;
    double flux        = a * sig_rise * piece;

    out[0] = flux;

    double d_pl_dbeta = -phase;
    double d_pr_dbeta = -gamma * e_decay;
    out[1] = a * sig_rise * ((1.0 - w) * d_pl_dbeta + w * d_pr_dbeta);

    double dw_dgamma = -k * w * (1.0 - w);
    double dw_dloggamma = dw_dgamma * gamma;
    double dpr_dgamma = e_decay * (-beta + (1.0 - beta * gamma) / tau_fall);
    double dpr_dloggamma = dpr_dgamma * gamma;
    double d_piece_dloggamma = dw_dloggamma * (piece_right - piece_left) + w * dpr_dloggamma;
    out[2] = a * sig_rise * d_piece_dloggamma;

    double dsig_dphase = sig_rise * (1.0 - sig_rise) / tau_rise;
    double dsig_dt0    = -dsig_dphase;
    double dw_dphase   = k * w * (1.0 - w);
    double dw_dt0      = -dw_dphase;
    double dpl_dt0     = beta;
    double dpr_dt0     = (1.0 - beta * gamma) * e_decay / tau_fall;
    double d_piece_dt0 = dw_dt0 * (piece_right - piece_left) + (1.0 - w) * dpl_dt0 + w * dpr_dt0;
    out[3] = a * (dsig_dt0 * piece + sig_rise * d_piece_dt0);

    out[4] = a * piece * sig_rise * (1.0 - sig_rise) * (-phase / tau_rise);

    double d_pr_dlogtf = piece_right * (phase - gamma) / tau_fall;
    out[5] = a * sig_rise * w * d_pr_dlogtf;

    out[6] = 0.0;
}

// TDE: params = [log_a, b, t0, log_tau_rise, log_tau_fall, alpha, log_sigma_extra]
__device__ inline void lc_tde_grad(const double* p, double t, double* out) {
    double a        = exp(p[0]);
    double t0       = p[2];
    double tau_rise = exp(p[3]);
    double tau_fall = exp(p[4]);
    double alpha    = p[5];
    double phase    = t - t0;
    double sig      = lc_sigmoid(phase / tau_rise);
    double ps, sig_p;
    lc_softplus_both(phase, &ps, &sig_p);
    double w        = 1.0 + ps / tau_fall;
    double decay    = pow(w, -alpha);
    double base     = a * sig * decay;

    out[0] = base;
    out[1] = 1.0;
    double dsig_dt0   = -sig * (1.0 - sig) / tau_rise;
    double ddecay_dt0 = alpha * pow(w, -alpha - 1.0) * sig_p / tau_fall;
    out[2] = a * (dsig_dt0 * decay + sig * ddecay_dt0);
    out[3] = a * decay * (-sig * (1.0 - sig) * phase / tau_rise);
    out[4] = a * sig * alpha * pow(w, -alpha - 1.0) * ps / tau_fall;
    out[5] = a * sig * (-log(w)) * decay;
    out[6] = 0.0;
}

// Arnett: params = [log_a, t0, log_tau_m, logit_f, log_sigma_extra]
__device__ inline void lc_arnett_grad(const double* p, double t, double* out) {
    double a       = exp(p[0]);
    double t0      = p[1];
    double tau_m   = exp(p[2]);
    double logit_f = p[3];
    double phase   = t - t0;
    double ps, sig_p;
    lc_softplus_both(phase, &ps, &sig_p);
    double f       = lc_sigmoid(logit_f);
    double e_ni    = exp(-ps / 8.8);
    double e_co    = exp(-ps / 111.3);
    double heat    = f * e_ni + (1.0 - f) * e_co;
    double x       = ps / tau_m;
    double exp_x2  = exp(-x * x);
    double trap    = 1.0 - exp_x2;
    double flux    = a * heat * trap;

    out[0] = flux;
    double dheat_dps = -f * e_ni / 8.8 - (1.0 - f) * e_co / 111.3;
    double dtrap_dps = 2.0 * ps * exp_x2 / (tau_m * tau_m);
    out[1] = a * (-sig_p) * (dheat_dps * trap + heat * dtrap_dps);
    out[2] = -2.0 * a * heat * exp_x2 * x * x;
    out[3] = a * trap * (e_ni - e_co) * f * (1.0 - f);
    out[4] = 0.0;
}

// Magnetar: params = [log_a, t0, log_tau_sd, log_tau_diff, log_sigma_extra]
__device__ inline void lc_magnetar_grad(const double* p, double t, double* out) {
    double a        = exp(p[0]);
    double t0       = p[1];
    double tau_sd   = exp(p[2]);
    double tau_diff = exp(p[3]);
    double phase    = t - t0;
    double ps, sig_p;
    lc_softplus_both(phase, &ps, &sig_p);
    double w        = 1.0 + ps / tau_sd;
    double w3       = w * w * w;
    double spindown = 1.0 / (w * w);
    double x        = ps / tau_diff;
    double exp_x2   = exp(-x * x);
    double trap     = 1.0 - exp_x2;
    double flux     = a * spindown * trap;

    out[0] = flux;
    double dspindown_dps = -2.0 / (w3 * tau_sd);
    double dtrap_dps     = 2.0 * ps * exp_x2 / (tau_diff * tau_diff);
    out[1] = a * (-sig_p) * (dspindown_dps * trap + spindown * dtrap_dps);
    out[2] = a * trap * 2.0 * ps / (w3 * tau_sd);
    out[3] = -2.0 * a * spindown * exp_x2 * x * x;
    out[4] = 0.0;
}

// ShockCooling: params = [log_a, t0, n, log_tau_tr, log_sigma_extra]
__device__ inline void lc_shock_cooling_grad(const double* p, double t, double* out) {
    double a      = exp(p[0]);
    double t0     = p[1];
    double n      = p[2];
    double tau_tr = exp(p[3]);
    double phase  = t - t0;
    double sig5   = lc_sigmoid(phase * 5.0);
    double ps, sig_p;
    lc_softplus_both(phase, &ps, &sig_p);
    double cooling = pow(ps, -n);
    double ratio   = ps / tau_tr;
    double cutoff  = exp(-ratio * ratio);
    double base    = cooling * cutoff;
    double flux    = a * sig5 * base;

    out[0] = flux;
    out[1] = a * base * (-5.0 * sig5 * (1.0 - sig5)
             + sig5 * sig_p * (n / ps + 2.0 * ps / (tau_tr * tau_tr)));
    out[2] = -flux * log(ps);
    out[3] = flux * 2.0 * ratio * ratio;
    out[4] = 0.0;
}

// Afterglow: params = [log_a, t0, log_t_b, alpha1, alpha2, log_sigma_extra]
__device__ inline void lc_afterglow_grad(const double* p, double t, double* out) {
    double a      = exp(p[0]);
    double t0     = p[1];
    double t_b    = exp(p[2]);
    double alpha1 = p[3];
    double alpha2 = p[4];
    double phase  = t - t0;
    double ps, sig_p;
    lc_softplus_both(phase, &ps, &sig_p);
    double r      = ps / t_b;
    double ln_r   = log(r);
    double u1     = exp(2.0 * alpha1 * ln_r);
    double u2     = exp(2.0 * alpha2 * ln_r);
    double u      = u1 + u2;
    double flux   = a * pow(u, -0.5);
    double u15    = pow(u, -1.5);

    out[0] = flux;
    double du_dps   = (2.0 * alpha1 * u1 + 2.0 * alpha2 * u2) / ps;
    double dflux_dps = a * (-0.5) * u15 * du_dps;
    out[1] = dflux_dps * (-sig_p);
    double du_dlog_tb = -(2.0 * alpha1 * u1 + 2.0 * alpha2 * u2);
    out[2] = a * (-0.5) * u15 * du_dlog_tb;
    out[3] = a * (-0.5) * u15 * 2.0 * ln_r * u1;
    out[4] = a * (-0.5) * u15 * 2.0 * ln_r * u2;
    out[5] = 0.0;
}

// ===========================================================================
// Fused eval+grad functions (return flux, write gradient into out[])
// ===========================================================================

// Bazin: returns flux = a * exp(-dt/tau_fall) * sigmoid(dt/tau_rise) + b
__device__ inline double lc_bazin_at_and_grad(const double* p, double t, double* out) {
    double a        = exp(p[0]);
    double t0       = p[2];
    double tau_rise = exp(p[3]);
    double tau_fall = exp(p[4]);
    double dt       = t - t0;
    double e_fall   = exp(-dt / tau_fall);
    double sig      = lc_sigmoid(dt / tau_rise);
    double base     = a * e_fall * sig;
    out[0] = base;                                         // d/d(log_a)
    out[1] = 1.0;                                          // d/d(b)
    out[2] = base * (1.0/tau_fall - (1.0-sig)/tau_rise);   // d/d(t0)
    out[3] = -base * (1.0 - sig) * dt / tau_rise;         // d/d(log_tau_rise)
    out[4] = base * dt / tau_fall;                         // d/d(log_tau_fall)
    out[5] = 0.0;                                          // d/d(log_sigma_extra)
    return base + p[1];
}

// Villar: returns flux = a * sig_rise * piece
__device__ inline double lc_villar_at_and_grad(const double* p, double t, double* out) {
    double a        = exp(p[0]);
    double beta     = p[1];
    double gamma    = exp(p[2]);
    double t0       = p[3];
    double tau_rise = exp(p[4]);
    double tau_fall = exp(p[5]);
    double phase    = t - t0;
    double k        = 10.0;
    double sig_rise = lc_sigmoid(phase / tau_rise);
    double w        = lc_sigmoid(k * (phase - gamma));
    double piece_left  = 1.0 - beta * phase;
    double e_decay     = exp((gamma - phase) / tau_fall);
    double piece_right = (1.0 - beta * gamma) * e_decay;
    double piece       = (1.0 - w) * piece_left + w * piece_right;
    double flux        = a * sig_rise * piece;

    out[0] = flux;

    double d_pl_dbeta = -phase;
    double d_pr_dbeta = -gamma * e_decay;
    out[1] = a * sig_rise * ((1.0 - w) * d_pl_dbeta + w * d_pr_dbeta);

    double dw_dgamma = -k * w * (1.0 - w);
    double dw_dloggamma = dw_dgamma * gamma;
    double dpr_dgamma = e_decay * (-beta + (1.0 - beta * gamma) / tau_fall);
    double dpr_dloggamma = dpr_dgamma * gamma;
    double d_piece_dloggamma = dw_dloggamma * (piece_right - piece_left) + w * dpr_dloggamma;
    out[2] = a * sig_rise * d_piece_dloggamma;

    double dsig_dphase = sig_rise * (1.0 - sig_rise) / tau_rise;
    double dsig_dt0    = -dsig_dphase;
    double dw_dphase   = k * w * (1.0 - w);
    double dw_dt0      = -dw_dphase;
    double dpl_dt0     = beta;
    double dpr_dt0     = (1.0 - beta * gamma) * e_decay / tau_fall;
    double d_piece_dt0 = dw_dt0 * (piece_right - piece_left) + (1.0 - w) * dpl_dt0 + w * dpr_dt0;
    out[3] = a * (dsig_dt0 * piece + sig_rise * d_piece_dt0);

    out[4] = a * piece * sig_rise * (1.0 - sig_rise) * (-phase / tau_rise);

    double d_pr_dlogtf = piece_right * (phase - gamma) / tau_fall;
    out[5] = a * sig_rise * w * d_pr_dlogtf;

    out[6] = 0.0;
    return flux;
}

// TDE: returns flux = a * sig * decay + b
__device__ inline double lc_tde_at_and_grad(const double* p, double t, double* out) {
    double a        = exp(p[0]);
    double t0       = p[2];
    double tau_rise = exp(p[3]);
    double tau_fall = exp(p[4]);
    double alpha    = p[5];
    double phase    = t - t0;
    double sig      = lc_sigmoid(phase / tau_rise);
    double ps, sig_p;
    lc_softplus_both(phase, &ps, &sig_p);
    double w        = 1.0 + ps / tau_fall;
    double decay    = pow(w, -alpha);
    double base     = a * sig * decay;

    out[0] = base;
    out[1] = 1.0;
    double dsig_dt0   = -sig * (1.0 - sig) / tau_rise;
    double ddecay_dt0 = alpha * pow(w, -alpha - 1.0) * sig_p / tau_fall;
    out[2] = a * (dsig_dt0 * decay + sig * ddecay_dt0);
    out[3] = a * decay * (-sig * (1.0 - sig) * phase / tau_rise);
    out[4] = a * sig * alpha * pow(w, -alpha - 1.0) * ps / tau_fall;
    out[5] = a * sig * (-log(w)) * decay;
    out[6] = 0.0;
    return base + p[1];
}

// Arnett: returns flux = a * heat * trap
__device__ inline double lc_arnett_at_and_grad(const double* p, double t, double* out) {
    double a       = exp(p[0]);
    double t0      = p[1];
    double tau_m   = exp(p[2]);
    double logit_f = p[3];
    double phase   = t - t0;
    double ps, sig_p;
    lc_softplus_both(phase, &ps, &sig_p);
    double f       = lc_sigmoid(logit_f);
    double e_ni    = exp(-ps / 8.8);
    double e_co    = exp(-ps / 111.3);
    double heat    = f * e_ni + (1.0 - f) * e_co;
    double x       = ps / tau_m;
    double exp_x2  = exp(-x * x);
    double trap    = 1.0 - exp_x2;
    double flux    = a * heat * trap;

    out[0] = flux;
    double dheat_dps = -f * e_ni / 8.8 - (1.0 - f) * e_co / 111.3;
    double dtrap_dps = 2.0 * ps * exp_x2 / (tau_m * tau_m);
    out[1] = a * (-sig_p) * (dheat_dps * trap + heat * dtrap_dps);
    out[2] = -2.0 * a * heat * exp_x2 * x * x;
    out[3] = a * trap * (e_ni - e_co) * f * (1.0 - f);
    out[4] = 0.0;
    return flux;
}

// Magnetar: returns flux = a * spindown * trap
__device__ inline double lc_magnetar_at_and_grad(const double* p, double t, double* out) {
    double a        = exp(p[0]);
    double t0       = p[1];
    double tau_sd   = exp(p[2]);
    double tau_diff = exp(p[3]);
    double phase    = t - t0;
    double ps, sig_p;
    lc_softplus_both(phase, &ps, &sig_p);
    double w        = 1.0 + ps / tau_sd;
    double w3       = w * w * w;
    double spindown = 1.0 / (w * w);
    double x        = ps / tau_diff;
    double exp_x2   = exp(-x * x);
    double trap     = 1.0 - exp_x2;
    double flux     = a * spindown * trap;

    out[0] = flux;
    double dspindown_dps = -2.0 / (w3 * tau_sd);
    double dtrap_dps     = 2.0 * ps * exp_x2 / (tau_diff * tau_diff);
    out[1] = a * (-sig_p) * (dspindown_dps * trap + spindown * dtrap_dps);
    out[2] = a * trap * 2.0 * ps / (w3 * tau_sd);
    out[3] = -2.0 * a * spindown * exp_x2 * x * x;
    out[4] = 0.0;
    return flux;
}

// ShockCooling: returns flux = a * sig5 * base
__device__ inline double lc_shock_cooling_at_and_grad(const double* p, double t, double* out) {
    double a      = exp(p[0]);
    double t0     = p[1];
    double n      = p[2];
    double tau_tr = exp(p[3]);
    double phase  = t - t0;
    double sig5   = lc_sigmoid(phase * 5.0);
    double ps, sig_p;
    lc_softplus_both(phase, &ps, &sig_p);
    double cooling = pow(ps, -n);
    double ratio   = ps / tau_tr;
    double cutoff  = exp(-ratio * ratio);
    double base    = cooling * cutoff;
    double flux    = a * sig5 * base;

    out[0] = flux;
    out[1] = a * base * (-5.0 * sig5 * (1.0 - sig5)
             + sig5 * sig_p * (n / ps + 2.0 * ps / (tau_tr * tau_tr)));
    out[2] = -flux * log(ps);
    out[3] = flux * 2.0 * ratio * ratio;
    out[4] = 0.0;
    return flux;
}

// Afterglow: returns flux = a * pow(u1+u2, -0.5)
__device__ inline double lc_afterglow_at_and_grad(const double* p, double t, double* out) {
    double a      = exp(p[0]);
    double t0     = p[1];
    double t_b    = exp(p[2]);
    double alpha1 = p[3];
    double alpha2 = p[4];
    double phase  = t - t0;
    double ps, sig_p;
    lc_softplus_both(phase, &ps, &sig_p);
    double r      = ps / t_b;
    double ln_r   = log(r);
    double u1     = exp(2.0 * alpha1 * ln_r);
    double u2     = exp(2.0 * alpha2 * ln_r);
    double u      = u1 + u2;
    double flux   = a * pow(u, -0.5);
    double u15    = pow(u, -1.5);

    out[0] = flux;
    double du_dps   = (2.0 * alpha1 * u1 + 2.0 * alpha2 * u2) / ps;
    double dflux_dps = a * (-0.5) * u15 * du_dps;
    out[1] = dflux_dps * (-sig_p);
    double du_dlog_tb = -(2.0 * alpha1 * u1 + 2.0 * alpha2 * u2);
    out[2] = a * (-0.5) * u15 * du_dlog_tb;
    out[3] = a * (-0.5) * u15 * 2.0 * ln_r * u1;
    out[4] = a * (-0.5) * u15 * 2.0 * ln_r * u2;
    out[5] = 0.0;
    return flux;
}

// ===========================================================================
// Fused eval+grad dispatch
// ===========================================================================

// Returns predicted flux and writes gradient into grad[0..n_params-1].
// For MetzgerKN (model 7), sets *ok = false; caller should use finite-diff.
__device__ inline double lc_eval_model_at_and_grad(
    int model_id, const double* p, double t, double* grad, bool* ok) {
    *ok = true;
    switch (model_id) {
        case 0: return lc_bazin_at_and_grad(p, t, grad);
        case 1: return lc_villar_at_and_grad(p, t, grad);
        case 2: return lc_tde_at_and_grad(p, t, grad);
        case 3: return lc_arnett_at_and_grad(p, t, grad);
        case 4: return lc_magnetar_at_and_grad(p, t, grad);
        case 5: return lc_shock_cooling_at_and_grad(p, t, grad);
        case 6: return lc_afterglow_at_and_grad(p, t, grad);
        default: *ok = false; return 0.0;
    }
}

// ===========================================================================
// Gradient dispatch
// ===========================================================================

// Writes gradient into out[0..n_params-1]. For MetzgerKN, returns false
// to signal caller should use finite-diff.
__device__ inline bool lc_eval_model_grad(int model_id, const double* p, double t, double* out) {
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

#define LC_MSUN_CGS  1.989e33
#define LC_C_CGS_VAL 2.998e10
#define LC_SECS_DAY  86400.0

__device__ inline double lc_metzger_kn_at_single(const double* p, double obs_time) {
    double m_ej    = pow(10.0, p[0]) * LC_MSUN_CGS;
    double v_ej    = pow(10.0, p[1]) * LC_C_CGS_VAL;
    double kappa_r = pow(10.0, p[2]);
    double t0      = p[3];
    double phase   = obs_time - t0;

    if (phase <= 0.0) return 0.0;

    double phase_max = phase * 1.05;
    if (phase_max < 0.02) phase_max = 0.02;
    double log_t_min = log(0.01);
    double log_t_max = log(phase_max);
    int ngrid = 100;

    double ye = 0.1;
    double xn0 = 1.0 - 2.0 * ye;
    double scale = 1e40;
    double e0 = 0.5 * m_ej * v_ej * v_ej;
    double e_th = e0 / scale;
    double e_kin = e0 / scale;
    double v = v_ej;

    double grid_t_0 = exp(log_t_min);
    double r = grid_t_0 * LC_SECS_DAY * v;

    double l_peak = 0.0;
    double l_at_phase = 0.0;
    double prev_t = grid_t_0;
    double prev_lrad = 0.0;

    for (int i = 0; i < ngrid; i++) {
        double t_day = exp(log_t_min + (log_t_max - log_t_min) * (double)i / (double)(ngrid - 1));
        double t_sec = t_day * LC_SECS_DAY;

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
        double t_diff = 3.0 * kappa_eff * m_ej / (4.0 * M_PI * LC_C_CGS_VAL * v * t_sec) + r / LC_C_CGS_VAL;

        double l_rad = (e_th > 0.0 && t_diff > 0.0) ? e_th / t_diff : 0.0;
        if (l_rad > l_peak) l_peak = l_rad;

        if (i > 0 && phase >= prev_t && phase <= t_day) {
            double frac = (phase - prev_t) / (t_day - prev_t);
            l_at_phase = prev_lrad + frac * (l_rad - prev_lrad);
        }
        if (i == ngrid - 1 && phase >= t_day) l_at_phase = l_rad;
        if (i == 0 && phase <= t_day) l_at_phase = l_rad;

        prev_t = t_day;
        prev_lrad = l_rad;

        double l_pdv = (r > 0.0) ? e_th * v / r : 0.0;
        if (i < ngrid - 1) {
            double t_next = exp(log_t_min + (log_t_max - log_t_min) * (double)(i + 1) / (double)(ngrid - 1));
            double dt_sec = (t_next - t_day) * LC_SECS_DAY;
            e_th += (l_heat - l_pdv - l_rad) * dt_sec;
            if (e_th < 0.0) e_th = 0.0;
            e_kin += l_pdv * dt_sec;
            v = sqrt(2.0 * e_kin * scale / m_ej);
            if (v > LC_C_CGS_VAL) v = LC_C_CGS_VAL;
            r += v * dt_sec;
        }
    }

    if (l_peak <= 0.0 || !isfinite(l_peak)) return 0.0;
    return l_at_phase / l_peak;
}

// Unified model dispatch
__device__ inline double lc_eval_model_at(int model_id, const double* p, double t) {
    switch (model_id) {
        case 0: return lc_bazin_at(p, t);
        case 1: return lc_villar_at(p, t);
        case 2: return lc_tde_at(p, t);
        case 3: return lc_arnett_at(p, t);
        case 4: return lc_magnetar_at(p, t);
        case 5: return lc_shock_cooling_at(p, t);
        case 6: return lc_afterglow_at(p, t);
        case 7: return lc_metzger_kn_at_single(p, t);
        default: return 0.0;
    }
}
