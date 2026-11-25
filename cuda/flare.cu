// Batch stellar flare search (Davenport et al. 2014 template) — CUDA.
//
// One warp (32 lanes) per window. Lanes split the per-point sums; the small
// linear solves and the Nelder-Mead logic run redundantly on every lane from
// butterfly all-reduce results, so all lanes follow the same control flow.
//
// Same algorithm as metal/flare.metal and the CPU search in src/flare.rs, in
// double precision. Inputs are pre-scaled per window on the host (times in
// cadence units, flux in units of the median error).

#include <math_constants.h>

#define FL_MAX_K 6
#define FL_MAX_NB 2
#define FL_MAX_P (FL_MAX_K + FL_MAX_NB)
#define FL_MAX_DIM (2 * FL_MAX_K)
#define FL_N_TP 11
#define FL_N_TH 20
#define FL_LANES 32

__constant__ double FL_RISE[5] = {1.0, 1.94053, -0.175084, -2.24588, -1.12498};
__constant__ double FL_DAMP[2] = {0.689008, 0.302963};
__constant__ double FL_DRATE[2] = {-1.60053, -0.278318};

// ---------------------------------------------------------------------------
// Warp reductions (result valid in every lane)
// ---------------------------------------------------------------------------

static __device__ inline double fl_warp_sum(double v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

static __device__ inline double fl_warp_max(double v) {
    for (int o = 16; o > 0; o >>= 1) v = fmax(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}

static __device__ inline int fl_warp_min(int v) {
    for (int o = 16; o > 0; o >>= 1) v = min(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}

static __device__ inline double fl_clamp(double x, double lo, double hi) {
    return fmin(fmax(x, lo), hi);
}

// ---------------------------------------------------------------------------
// Template
// ---------------------------------------------------------------------------

static __device__ inline double fl_template(double x) {
    if (x < -1.0) return 0.0;
    if (x <= 0.0)
        return FL_RISE[0] + x * (FL_RISE[1] + x * (FL_RISE[2] + x * (FL_RISE[3] + x * FL_RISE[4])));
    return FL_DAMP[0] * exp(FL_DRATE[0] * x) + FL_DAMP[1] * exp(FL_DRATE[1] * x);
}

// ∫_a^b x^n dx = (b − a)·h_n(a, b)/(n + 1) with h_n = a·h_{n−1} + bⁿ.
static __device__ inline double fl_rise_integral(double a, double b) {
    double h = 1.0, bn = 1.0, s = FL_RISE[0];
    for (int n = 1; n < 5; n++) {
        bn *= b;
        h = a * h + bn;
        s += FL_RISE[n] * h / (double)(n + 1);
    }
    return (b - a) * s;
}

static __device__ inline double fl_template_integral(double a, double b) {
    if (!(b > a)) return 0.0;
    double total = 0.0;
    double lo = fmax(a, -1.0), hi = fmin(b, 0.0);
    if (hi > lo) total += fl_rise_integral(lo, hi);
    lo = fmax(a, 0.0);
    if (b > lo) {
        for (int j = 0; j < 2; j++) {
            double r = FL_DRATE[j];
            total += FL_DAMP[j] / r * exp(r * lo) * expm1(r * (b - lo));
        }
    }
    return total;
}

static __device__ inline double fl_template_binned(double x, double hw) {
    if (hw > 1e-9) {
        if (x + hw <= -1.0) return 0.0;
        return fl_template_integral(x - hw, x + hw) / (2.0 * hw);
    }
    return fl_template(x);
}

// ---------------------------------------------------------------------------
// Per-window problem
// ---------------------------------------------------------------------------

struct FlareCtx {
    const double* t;
    const double* y;
    const double* w;
    const double* s;
    int n;
    int nb;
    double half_exp, tp_lo, tp_hi, lt_lo, lt_hi, cadence;
    double area, min_cov;
    int lane;
};

// Cholesky solve of an m×m SPD system (row stride m) in place; false if not PD.
static __device__ bool fl_cholesky_solve(double* a, double* b, int m) {
    for (int j = 0; j < m; j++) {
        double d = a[j * m + j];
        for (int k = 0; k < j; k++) d -= a[j * m + k] * a[j * m + k];
        if (!(d > 1e-12 * fmax(fabs(a[j * m + j]), 1e-300))) return false;
        d = sqrt(d);
        a[j * m + j] = d;
        for (int i = j + 1; i < m; i++) {
            double s = a[i * m + j];
            for (int k = 0; k < j; k++) s -= a[i * m + k] * a[j * m + k];
            a[i * m + j] = s / d;
        }
    }
    for (int i = 0; i < m; i++) {
        double s = b[i];
        for (int k = 0; k < i; k++) s -= a[i * m + k] * b[k];
        b[i] = s / a[i * m + i];
    }
    for (int i = m - 1; i >= 0; i--) {
        double s = b[i];
        for (int k = i + 1; k < m; k++) s -= a[k * m + i] * b[k];
        b[i] = s / a[i * m + i];
    }
    return true;
}

// Columns at one point: k flare templates then the baseline powers of t.
static __device__ inline void fl_columns(const FlareCtx& c, double ti, int k,
                                         const double* tp, const double* inv_th, const double* hw,
                                         double* col) {
    for (int j = 0; j < k; j++) col[j] = fl_template_binned((ti - tp[j]) * inv_th[j], hw[j]);
    double pw = 1.0;
    for (int b = 0; b < c.nb; b++) { col[k + b] = pw; pw *= ti; }
}

// Profiled χ² for θ = [t_peak_1, ln t_half_1, …]: non-negative amplitudes and a
// free baseline by weighted least squares. Components with less than
// min_cov of their area on observed points are held at zero.
static __device__ double fl_profile(const FlareCtx& c, const double* theta, int k, double* coef) {
    const int p = k + c.nb;
    double tp[FL_MAX_K], inv_th[FL_MAX_K], hw[FL_MAX_K], th[FL_MAX_K];
    for (int j = 0; j < k; j++) {
        th[j] = exp(theta[2 * j + 1]);
        inv_th[j] = 1.0 / th[j];
        hw[j] = c.half_exp * inv_th[j];
        tp[j] = theta[2 * j];
    }

    double gram[FL_MAX_P * FL_MAX_P];
    double rhs[FL_MAX_P];
    double cov[FL_MAX_K];
    for (int a = 0; a < p * p; a++) gram[a] = 0.0;
    for (int a = 0; a < p; a++) rhs[a] = 0.0;
    for (int j = 0; j < k; j++) cov[j] = 0.0;

    for (int i = c.lane; i < c.n; i += FL_LANES) {
        double col[FL_MAX_P];
        fl_columns(c, c.t[i], k, tp, inv_th, hw, col);
        double wi = c.w[i], yi = c.y[i];
        for (int a = 0; a < p; a++) {
            double wa = wi * col[a];
            rhs[a] += wa * yi;
            for (int b = a; b < p; b++) gram[a * p + b] += wa * col[b];
        }
        for (int j = 0; j < k; j++) cov[j] += col[j] * c.s[i];
    }
    for (int a = 0; a < p; a++) {
        rhs[a] = fl_warp_sum(rhs[a]);
        for (int b = a; b < p; b++) {
            double v = fl_warp_sum(gram[a * p + b]);
            gram[a * p + b] = v;
            gram[b * p + a] = v;
        }
    }
    for (int j = 0; j < k; j++) cov[j] = fl_warp_sum(cov[j]);

    bool active[FL_MAX_P];
    for (int j = 0; j < p; j++) active[j] = (j >= k) || (cov[j] >= c.min_cov * th[j] * c.area);
    for (int j = 0; j < p; j++) coef[j] = 0.0;

    // Active set: drop the most negative (or degenerate) flare and re-solve.
    for (int it = 0; it <= k; it++) {
        int idx[FL_MAX_P];
        int m = 0;
        for (int j = 0; j < p; j++) if (active[j]) idx[m++] = j;
        double g[FL_MAX_P * FL_MAX_P];
        double r[FL_MAX_P];
        for (int a = 0; a < m; a++) {
            r[a] = rhs[idx[a]];
            for (int b = 0; b < m; b++) g[a * m + b] = gram[idx[a] * p + idx[b]];
        }
        for (int j = 0; j < p; j++) coef[j] = 0.0;
        if (!fl_cholesky_solve(g, r, m)) {
            int drop = -1;
            for (int j = 0; j < k; j++)
                if (active[j] && (drop < 0 || gram[j * p + j] < gram[drop * p + drop])) drop = j;
            if (drop < 0) break;
            active[drop] = false;
            continue;
        }
        for (int a = 0; a < m; a++) coef[idx[a]] = r[a];
        int drop = -1;
        for (int j = 0; j < k; j++)
            if (active[j] && coef[j] < 0.0 && (drop < 0 || coef[j] < coef[drop])) drop = j;
        if (drop < 0) break;
        active[drop] = false;
    }

    double chi2 = 0.0;
    for (int i = c.lane; i < c.n; i += FL_LANES) {
        double col[FL_MAX_P];
        fl_columns(c, c.t[i], k, tp, inv_th, hw, col);
        double m = 0.0;
        for (int a = 0; a < p; a++) m += coef[a] * col[a];
        double r = c.y[i] - m;
        chi2 += c.w[i] * r * r;
    }
    return fl_warp_sum(chi2);
}

// χ² at θ clamped into bounds, plus a penalty on the squared violation.
static __device__ double fl_cost(const FlareCtx& c, const double* theta, int k) {
    double clamped[FL_MAX_DIM];
    double excess = 0.0;
    double wtp = c.tp_hi - c.tp_lo, wlt = c.lt_hi - c.lt_lo;
    for (int j = 0; j < k; j++) {
        double tp = theta[2 * j], lt = theta[2 * j + 1];
        double etp = (fmax(c.tp_lo - tp, 0.0) + fmax(tp - c.tp_hi, 0.0)) / wtp;
        double elt = (fmax(c.lt_lo - lt, 0.0) + fmax(lt - c.lt_hi, 0.0)) / wlt;
        excess += etp * etp + elt * elt;
        clamped[2 * j] = fl_clamp(tp, c.tp_lo, c.tp_hi);
        clamped[2 * j + 1] = fl_clamp(lt, c.lt_lo, c.lt_hi);
    }
    double coef[FL_MAX_P];
    return fl_profile(c, clamped, k, coef) + 1e6 * excess;
}

static __device__ void fl_clamp_theta(const FlareCtx& c, double* theta, int k) {
    for (int j = 0; j < k; j++) {
        theta[2 * j] = fl_clamp(theta[2 * j], c.tp_lo, c.tp_hi);
        theta[2 * j + 1] = fl_clamp(theta[2 * j + 1], c.lt_lo, c.lt_hi);
    }
}

// Grid-search component k (0-based) on top of theta[0..2k), centred on the
// largest positive residual; writes theta[2k], theta[2k+1].
static __device__ void fl_seed(const FlareCtx& c, double* theta, int k) {
    double coef[FL_MAX_P];
    fl_profile(c, theta, k, coef);
    double tp[FL_MAX_K], inv_th[FL_MAX_K], hw[FL_MAX_K];
    for (int j = 0; j < k; j++) {
        inv_th[j] = exp(-theta[2 * j + 1]);
        hw[j] = c.half_exp * inv_th[j];
        tp[j] = theta[2 * j];
    }
    double best_s = -CUDART_INF;
    int best_i = 0x7fffffff;
    for (int i = c.lane; i < c.n; i += FL_LANES) {
        double col[FL_MAX_P];
        fl_columns(c, c.t[i], k, tp, inv_th, hw, col);
        double m = 0.0;
        for (int a = 0; a < k + c.nb; a++) m += coef[a] * col[a];
        double s = (c.y[i] - m) * sqrt(c.w[i]);
        if (s > best_s) { best_s = s; best_i = i; }
    }
    double smax = fl_warp_max(best_s);
    int i0 = fl_warp_min(best_s == smax ? best_i : 0x7fffffff);
    if (i0 == 0x7fffffff) i0 = 0;

    double t0 = c.t[i0];
    double dt = c.cadence;
    double best = CUDART_INF, best_tp = t0, best_lt = c.lt_lo;
    for (int a = 0; a < FL_N_TP; a++) {
        double tpa = fl_clamp(t0 - 1.5 * dt + 2.5 * dt * a / (double)(FL_N_TP - 1), c.tp_lo, c.tp_hi);
        for (int b = 0; b < FL_N_TH; b++) {
            double lt = c.lt_lo + (c.lt_hi - c.lt_lo) * b / (double)(FL_N_TH - 1);
            theta[2 * k] = tpa;
            theta[2 * k + 1] = lt;
            double chi2 = fl_profile(c, theta, k + 1, coef);
            if (chi2 < best) { best = chi2; best_tp = tpa; best_lt = lt; }
        }
    }
    theta[2 * k] = best_tp;
    theta[2 * k + 1] = best_lt;
}

// Nelder-Mead on fl_cost from x (in place); returns the cost at the clamped
// result, or leaves x unchanged if that is no better than the start.
static __device__ double fl_nelder_mead(const FlareCtx& c, double* x, int k, int max_iters,
                                        double tol_abs, double tol_rel) {
    const int d = 2 * k;
    double v[(FL_MAX_DIM + 1) * FL_MAX_DIM];
    double f[FL_MAX_DIM + 1];
    double f_start = fl_cost(c, x, k);
    for (int i = 0; i <= d; i++) {
        for (int j = 0; j < d; j++) v[i * d + j] = x[j];
        if (i > 0) v[i * d + (i - 1)] += ((i - 1) % 2 == 0) ? 0.5 * c.cadence : 0.3;
        f[i] = (i == 0) ? f_start : fl_cost(c, &v[i * d], k);
    }
    double cen[FL_MAX_DIM], xr[FL_MAX_DIM], xt[FL_MAX_DIM];
    for (int iter = 0; iter < max_iters; iter++) {
        for (int i = 1; i <= d; i++) {
            for (int j = i; j > 0 && f[j] < f[j - 1]; j--) {
                double tf = f[j]; f[j] = f[j - 1]; f[j - 1] = tf;
                for (int q = 0; q < d; q++) {
                    double tv = v[j * d + q]; v[j * d + q] = v[(j - 1) * d + q]; v[(j - 1) * d + q] = tv;
                }
            }
        }
        double mean = 0.0;
        for (int i = 0; i <= d; i++) mean += f[i];
        mean /= (double)(d + 1);
        double var = 0.0;
        for (int i = 0; i <= d; i++) var += (f[i] - mean) * (f[i] - mean);
        if (sqrt(var / (double)(d + 1)) < tol_abs + tol_rel * fabs(f[0])) break;

        for (int q = 0; q < d; q++) {
            double s = 0.0;
            for (int i = 0; i < d; i++) s += v[i * d + q];
            cen[q] = s / (double)d;
        }
        double* worst = &v[d * d];
        for (int q = 0; q < d; q++) xr[q] = 2.0 * cen[q] - worst[q];
        double fr = fl_cost(c, xr, k);
        if (fr < f[0]) {
            for (int q = 0; q < d; q++) xt[q] = cen[q] + 2.0 * (xr[q] - cen[q]);
            double fe = fl_cost(c, xt, k);
            if (fe < fr) { for (int q = 0; q < d; q++) worst[q] = xt[q]; f[d] = fe; }
            else { for (int q = 0; q < d; q++) worst[q] = xr[q]; f[d] = fr; }
        } else if (fr < f[d - 1]) {
            for (int q = 0; q < d; q++) worst[q] = xr[q];
            f[d] = fr;
        } else {
            bool outside = fr < f[d];
            for (int q = 0; q < d; q++)
                xt[q] = cen[q] + 0.5 * ((outside ? xr[q] : worst[q]) - cen[q]);
            double fc = fl_cost(c, xt, k);
            if (outside ? (fc <= fr) : (fc < f[d])) {
                for (int q = 0; q < d; q++) worst[q] = xt[q];
                f[d] = fc;
            } else {
                for (int i = 1; i <= d; i++) {
                    for (int q = 0; q < d; q++) v[i * d + q] = v[q] + 0.5 * (v[i * d + q] - v[q]);
                    f[i] = fl_cost(c, &v[i * d], k);
                }
            }
        }
    }
    int best = 0;
    for (int i = 1; i <= d; i++) if (f[i] < f[best]) best = i;
    double xb[FL_MAX_DIM];
    for (int q = 0; q < d; q++) xb[q] = v[best * d + q];
    fl_clamp_theta(c, xb, k);
    double fb = fl_cost(c, xb, k);
    if (fb <= f_start) {
        for (int q = 0; q < d; q++) x[q] = xb[q];
        return fb;
    }
    return f_start;
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

extern "C" __global__ void batch_flare_fit(
    const double* __restrict__ all_t,
    const double* __restrict__ all_y,
    const double* __restrict__ all_w,
    const double* __restrict__ all_s,
    const int*    __restrict__ offsets,
    const double* __restrict__ win_params,   // [n_windows * 6]
    double* out_theta,                       // [n_windows * FL_MAX_DIM]
    int*    out_k,                           // [n_windows]
    double* out_bic,                         // [n_windows * (FL_MAX_K + 1)]
    int*    out_nbic,                        // [n_windows]
    int n_windows, int max_k, int nb, int iters_per_dim,
    double min_dbic, double area, double min_cov, double tol_abs, double tol_rel)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int win = tid / FL_LANES;
    if (win >= n_windows) return;
    max_k = min(max_k, FL_MAX_K);

    int start = offsets[win];
    FlareCtx c;
    c.t = all_t + start;
    c.y = all_y + start;
    c.w = all_w + start;
    c.s = all_s + start;
    c.n = offsets[win + 1] - start;
    c.nb = nb;
    const double* wp = win_params + win * 6;
    c.half_exp = wp[0];
    c.tp_lo = wp[1];
    c.tp_hi = wp[2];
    c.lt_lo = wp[3];
    c.lt_hi = wp[4];
    c.cadence = wp[5];
    c.area = area;
    c.min_cov = min_cov;
    c.lane = threadIdx.x & (FL_LANES - 1);

    double ln_n = log((double)c.n);
    double best_theta[FL_MAX_DIM];
    double theta[FL_MAX_DIM];
    for (int q = 0; q < FL_MAX_DIM; q++) { best_theta[q] = 0.0; theta[q] = 0.0; }
    double coef[FL_MAX_P];
    double best_bic = fl_profile(c, best_theta, 0, coef) + nb * ln_n;
    double bics[FL_MAX_K + 1];
    bics[0] = best_bic;
    int nbic = 1, best_k = 0;

    for (int k = 1; k <= max_k; k++) {
        if (c.n < 3 * k + nb + 1) break;
        for (int q = 0; q < 2 * (k - 1); q++) theta[q] = best_theta[q];
        fl_seed(c, theta, k - 1);
        int iters = iters_per_dim * 2 * k;
        fl_nelder_mead(c, theta, k, iters, tol_abs, tol_rel);
        double chi2 = fl_nelder_mead(c, theta, k, iters, tol_abs, tol_rel);
        double kb = chi2 + (3 * k + nb) * ln_n;
        bics[nbic++] = kb;
        if (kb < best_bic - min_dbic) {
            best_bic = kb;
            best_k = k;
            for (int q = 0; q < 2 * k; q++) best_theta[q] = theta[q];
        } else {
            break;
        }
    }

    if (c.lane == 0) {
        for (int q = 0; q < FL_MAX_DIM; q++) out_theta[win * FL_MAX_DIM + q] = best_theta[q];
        out_k[win] = best_k;
        for (int q = 0; q <= FL_MAX_K; q++)
            out_bic[win * (FL_MAX_K + 1) + q] = q < nbic ? bics[q] : CUDART_NAN;
        out_nbic[win] = nbic;
    }
}

// ===========================================================================
// Host-side launch wrapper
// ===========================================================================

extern "C" void launch_batch_flare_fit(
    const double* all_t, const double* all_y, const double* all_w, const double* all_s,
    const int* offsets, const double* win_params,
    double* out_theta, int* out_k, double* out_bic, int* out_nbic,
    int n_windows, int max_k, int nb, int iters_per_dim,
    double min_dbic, double area, double min_cov, double tol_abs, double tol_rel,
    int grid, int block)
{
    batch_flare_fit<<<grid, block>>>(
        all_t, all_y, all_w, all_s, offsets, win_params,
        out_theta, out_k, out_bic, out_nbic,
        n_windows, max_k, nb, iters_per_dim,
        min_dbic, area, min_cov, tol_abs, tol_rel);
}
