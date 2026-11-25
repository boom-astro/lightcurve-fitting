// Batch stellar flare search (Davenport et al. 2014 template) — Metal.
//
// One SIMD group (32 lanes) per window. Lanes split the per-point sums; the
// small linear solves and the Nelder-Mead logic run redundantly on every lane
// from simd_sum results, so all lanes follow the same control flow.
//
// Mirrors the CPU search in src/flare.rs: baseline-only fit, then for each K a
// residual-seeded grid, two Nelder-Mead passes over [t_peak, ln t_half], and
// acceptance by BIC. Inputs are pre-scaled per window on the host (times in
// cadence units, flux in units of the median error).

#include <metal_stdlib>
using namespace metal;

#define FL_MAX_K 6
#define FL_MAX_NB 2
#define FL_MAX_P (FL_MAX_K + FL_MAX_NB)
#define FL_MAX_DIM (2 * FL_MAX_K)
#define FL_N_TP 11
#define FL_N_TH 20
#define FL_LANES 32

constant float FL_RISE[5] = {1.0f, 1.94053f, -0.175084f, -2.24588f, -1.12498f};
constant float FL_DAMP[2] = {0.689008f, 0.302963f};
constant float FL_DRATE[2] = {-1.60053f, -0.278318f};

// ---------------------------------------------------------------------------
// Template
// ---------------------------------------------------------------------------

inline float fl_template(float x) {
    if (x < -1.0f) return 0.0f;
    if (x <= 0.0f)
        return FL_RISE[0] + x * (FL_RISE[1] + x * (FL_RISE[2] + x * (FL_RISE[3] + x * FL_RISE[4])));
    return FL_DAMP[0] * exp(FL_DRATE[0] * x) + FL_DAMP[1] * exp(FL_DRATE[1] * x);
}

inline float fl_expm1(float x) {
    return fabs(x) < 1e-2f ? x * (1.0f + x * (0.5f + x * (1.0f / 6.0f + x * (1.0f / 24.0f))))
                           : exp(x) - 1.0f;
}

// ∫_a^b x^n dx = (b − a)·h_n(a, b)/(n + 1) with h_n = a·h_{n−1} + bⁿ, which
// avoids differencing two nearly equal antiderivatives for short exposures.
inline float fl_rise_integral(float a, float b) {
    float h = 1.0f, bn = 1.0f, s = FL_RISE[0];
    for (int n = 1; n < 5; n++) {
        bn *= b;
        h = a * h + bn;
        s += FL_RISE[n] * h / float(n + 1);
    }
    return (b - a) * s;
}

inline float fl_template_integral(float a, float b) {
    if (!(b > a)) return 0.0f;
    float total = 0.0f;
    float lo = max(a, -1.0f), hi = min(b, 0.0f);
    if (hi > lo) total += fl_rise_integral(lo, hi);
    lo = max(a, 0.0f);
    if (b > lo) {
        for (int j = 0; j < 2; j++) {
            float r = FL_DRATE[j];
            total += FL_DAMP[j] / r * exp(r * lo) * fl_expm1(r * (b - lo));
        }
    }
    return total;
}

inline float fl_template_binned(float x, float hw) {
    if (hw > 1e-7f) {
        if (x + hw <= -1.0f) return 0.0f;
        return fl_template_integral(x - hw, x + hw) / (2.0f * hw);
    }
    return fl_template(x);
}

// ---------------------------------------------------------------------------
// Per-window problem
// ---------------------------------------------------------------------------

struct FlareCtx {
    device const float* t;
    device const float* y;
    device const float* w;
    device const float* s;
    int n;
    int nb;
    float half_exp, tp_lo, tp_hi, lt_lo, lt_hi, cadence;
    float area, min_cov;
    uint lane;
};

// Cholesky solve of an m×m SPD system (row stride m) in place; false if not PD.
inline bool fl_cholesky_solve(thread float* a, thread float* b, int m) {
    for (int j = 0; j < m; j++) {
        float d = a[j * m + j];
        for (int k = 0; k < j; k++) d -= a[j * m + k] * a[j * m + k];
        if (!(d > 1e-6f * max(fabs(a[j * m + j]), 1e-30f))) return false;
        d = sqrt(d);
        a[j * m + j] = d;
        for (int i = j + 1; i < m; i++) {
            float s = a[i * m + j];
            for (int k = 0; k < j; k++) s -= a[i * m + k] * a[j * m + k];
            a[i * m + j] = s / d;
        }
    }
    for (int i = 0; i < m; i++) {
        float s = b[i];
        for (int k = 0; k < i; k++) s -= a[i * m + k] * b[k];
        b[i] = s / a[i * m + i];
    }
    for (int i = m - 1; i >= 0; i--) {
        float s = b[i];
        for (int k = i + 1; k < m; k++) s -= a[k * m + i] * b[k];
        b[i] = s / a[i * m + i];
    }
    return true;
}

// Columns at one point: k flare templates then the baseline powers of t.
inline void fl_columns(thread const FlareCtx& c, float ti, int k,
                       thread const float* tp, thread const float* inv_th, thread const float* hw,
                       thread float* col) {
    for (int j = 0; j < k; j++) col[j] = fl_template_binned((ti - tp[j]) * inv_th[j], hw[j]);
    float pw = 1.0f;
    for (int b = 0; b < c.nb; b++) { col[k + b] = pw; pw *= ti; }
}

// Profiled χ² for θ = [t_peak_1, ln t_half_1, …]: non-negative amplitudes and a
// free baseline by weighted least squares. Components with less than
// min_cov of their area on observed points are held at zero.
static float fl_profile(thread const FlareCtx& c, thread const float* theta, int k, thread float* coef) {
    const int p = k + c.nb;
    float tp[FL_MAX_K], inv_th[FL_MAX_K], hw[FL_MAX_K], th[FL_MAX_K];
    for (int j = 0; j < k; j++) {
        th[j] = exp(theta[2 * j + 1]);
        inv_th[j] = 1.0f / th[j];
        hw[j] = c.half_exp * inv_th[j];
        tp[j] = theta[2 * j];
    }

    float gram[FL_MAX_P * FL_MAX_P];
    float rhs[FL_MAX_P];
    float cov[FL_MAX_K];
    for (int a = 0; a < p * p; a++) gram[a] = 0.0f;
    for (int a = 0; a < p; a++) rhs[a] = 0.0f;
    for (int j = 0; j < k; j++) cov[j] = 0.0f;

    for (int i = int(c.lane); i < c.n; i += FL_LANES) {
        float col[FL_MAX_P];
        fl_columns(c, c.t[i], k, tp, inv_th, hw, col);
        float wi = c.w[i], yi = c.y[i];
        for (int a = 0; a < p; a++) {
            float wa = wi * col[a];
            rhs[a] += wa * yi;
            for (int b = a; b < p; b++) gram[a * p + b] += wa * col[b];
        }
        for (int j = 0; j < k; j++) cov[j] += col[j] * c.s[i];
    }
    for (int a = 0; a < p; a++) {
        rhs[a] = simd_sum(rhs[a]);
        for (int b = a; b < p; b++) {
            float v = simd_sum(gram[a * p + b]);
            gram[a * p + b] = v;
            gram[b * p + a] = v;
        }
    }
    for (int j = 0; j < k; j++) cov[j] = simd_sum(cov[j]);

    bool active[FL_MAX_P];
    for (int j = 0; j < p; j++) active[j] = (j >= k) || (cov[j] >= c.min_cov * th[j] * c.area);
    for (int j = 0; j < p; j++) coef[j] = 0.0f;

    // Active set: drop the most negative (or degenerate) flare and re-solve.
    for (int it = 0; it <= k; it++) {
        int idx[FL_MAX_P];
        int m = 0;
        for (int j = 0; j < p; j++) if (active[j]) idx[m++] = j;
        float g[FL_MAX_P * FL_MAX_P];
        float r[FL_MAX_P];
        for (int a = 0; a < m; a++) {
            r[a] = rhs[idx[a]];
            for (int b = 0; b < m; b++) g[a * m + b] = gram[idx[a] * p + idx[b]];
        }
        for (int j = 0; j < p; j++) coef[j] = 0.0f;
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
            if (active[j] && coef[j] < 0.0f && (drop < 0 || coef[j] < coef[drop])) drop = j;
        if (drop < 0) break;
        active[drop] = false;
    }

    float chi2 = 0.0f;
    for (int i = int(c.lane); i < c.n; i += FL_LANES) {
        float col[FL_MAX_P];
        fl_columns(c, c.t[i], k, tp, inv_th, hw, col);
        float m = 0.0f;
        for (int a = 0; a < p; a++) m += coef[a] * col[a];
        float r = c.y[i] - m;
        chi2 += c.w[i] * r * r;
    }
    return simd_sum(chi2);
}

// χ² at θ clamped into bounds, plus a penalty on the squared violation.
static float fl_cost(thread const FlareCtx& c, thread const float* theta, int k) {
    float clamped[FL_MAX_DIM];
    float excess = 0.0f;
    float wtp = c.tp_hi - c.tp_lo, wlt = c.lt_hi - c.lt_lo;
    for (int j = 0; j < k; j++) {
        float tp = theta[2 * j], lt = theta[2 * j + 1];
        float etp = (max(c.tp_lo - tp, 0.0f) + max(tp - c.tp_hi, 0.0f)) / wtp;
        float elt = (max(c.lt_lo - lt, 0.0f) + max(lt - c.lt_hi, 0.0f)) / wlt;
        excess += etp * etp + elt * elt;
        clamped[2 * j] = clamp(tp, c.tp_lo, c.tp_hi);
        clamped[2 * j + 1] = clamp(lt, c.lt_lo, c.lt_hi);
    }
    float coef[FL_MAX_P];
    return fl_profile(c, clamped, k, coef) + 1e6f * excess;
}

static void fl_clamp(thread const FlareCtx& c, thread float* theta, int k) {
    for (int j = 0; j < k; j++) {
        theta[2 * j] = clamp(theta[2 * j], c.tp_lo, c.tp_hi);
        theta[2 * j + 1] = clamp(theta[2 * j + 1], c.lt_lo, c.lt_hi);
    }
}

// Grid-search component k (0-based) on top of theta[0..2k), centred on the
// largest positive residual; writes theta[2k], theta[2k+1].
static void fl_seed(thread const FlareCtx& c, thread float* theta, int k) {
    float coef[FL_MAX_P];
    fl_profile(c, theta, k, coef);
    float tp[FL_MAX_K], inv_th[FL_MAX_K], hw[FL_MAX_K];
    for (int j = 0; j < k; j++) {
        inv_th[j] = exp(-theta[2 * j + 1]);
        hw[j] = c.half_exp * inv_th[j];
        tp[j] = theta[2 * j];
    }
    float best_s = -INFINITY;
    int best_i = 0x7fffffff;
    for (int i = int(c.lane); i < c.n; i += FL_LANES) {
        float col[FL_MAX_P];
        fl_columns(c, c.t[i], k, tp, inv_th, hw, col);
        float m = 0.0f;
        for (int a = 0; a < k + c.nb; a++) m += coef[a] * col[a];
        float s = (c.y[i] - m) * sqrt(c.w[i]);
        if (s > best_s) { best_s = s; best_i = i; }
    }
    float smax = simd_max(best_s);
    int i0 = simd_min(best_s == smax ? best_i : 0x7fffffff);
    if (i0 == 0x7fffffff) i0 = 0;

    float t0 = c.t[i0];
    float dt = c.cadence;
    float best = INFINITY, best_tp = t0, best_lt = c.lt_lo;
    for (int a = 0; a < FL_N_TP; a++) {
        float tpa = clamp(t0 - 1.5f * dt + 2.5f * dt * float(a) / float(FL_N_TP - 1), c.tp_lo, c.tp_hi);
        for (int b = 0; b < FL_N_TH; b++) {
            float lt = c.lt_lo + (c.lt_hi - c.lt_lo) * float(b) / float(FL_N_TH - 1);
            theta[2 * k] = tpa;
            theta[2 * k + 1] = lt;
            float chi2 = fl_profile(c, theta, k + 1, coef);
            if (chi2 < best) { best = chi2; best_tp = tpa; best_lt = lt; }
        }
    }
    theta[2 * k] = best_tp;
    theta[2 * k + 1] = best_lt;
}

// Nelder-Mead on fl_cost from x (in place); returns the cost at the clamped
// result, or leaves x unchanged if that is no better than the start.
static float fl_nelder_mead(thread const FlareCtx& c, thread float* x, int k, int max_iters,
                            float tol_abs, float tol_rel) {
    const int d = 2 * k;
    float v[(FL_MAX_DIM + 1) * FL_MAX_DIM];
    float f[FL_MAX_DIM + 1];
    float f_start = fl_cost(c, x, k);
    for (int i = 0; i <= d; i++) {
        for (int j = 0; j < d; j++) v[i * d + j] = x[j];
        if (i > 0) v[i * d + (i - 1)] += ((i - 1) % 2 == 0) ? 0.5f * c.cadence : 0.3f;
        f[i] = (i == 0) ? f_start : fl_cost(c, &v[i * d], k);
    }
    float cen[FL_MAX_DIM], xr[FL_MAX_DIM], xt[FL_MAX_DIM];
    for (int iter = 0; iter < max_iters; iter++) {
        // Order vertices by cost (insertion sort; d + 1 <= 13).
        for (int i = 1; i <= d; i++) {
            for (int j = i; j > 0 && f[j] < f[j - 1]; j--) {
                float tf = f[j]; f[j] = f[j - 1]; f[j - 1] = tf;
                for (int q = 0; q < d; q++) {
                    float tv = v[j * d + q]; v[j * d + q] = v[(j - 1) * d + q]; v[(j - 1) * d + q] = tv;
                }
            }
        }
        float mean = 0.0f;
        for (int i = 0; i <= d; i++) mean += f[i];
        mean /= float(d + 1);
        float var = 0.0f;
        for (int i = 0; i <= d; i++) var += (f[i] - mean) * (f[i] - mean);
        if (sqrt(var / float(d + 1)) < tol_abs + tol_rel * fabs(f[0])) break;

        for (int q = 0; q < d; q++) {
            float s = 0.0f;
            for (int i = 0; i < d; i++) s += v[i * d + q];
            cen[q] = s / float(d);
        }
        thread float* worst = &v[d * d];
        for (int q = 0; q < d; q++) xr[q] = 2.0f * cen[q] - worst[q];
        float fr = fl_cost(c, xr, k);
        if (fr < f[0]) {
            for (int q = 0; q < d; q++) xt[q] = cen[q] + 2.0f * (xr[q] - cen[q]);
            float fe = fl_cost(c, xt, k);
            if (fe < fr) { for (int q = 0; q < d; q++) worst[q] = xt[q]; f[d] = fe; }
            else { for (int q = 0; q < d; q++) worst[q] = xr[q]; f[d] = fr; }
        } else if (fr < f[d - 1]) {
            for (int q = 0; q < d; q++) worst[q] = xr[q];
            f[d] = fr;
        } else {
            bool outside = fr < f[d];
            for (int q = 0; q < d; q++)
                xt[q] = cen[q] + 0.5f * ((outside ? xr[q] : worst[q]) - cen[q]);
            float fc = fl_cost(c, xt, k);
            if (outside ? (fc <= fr) : (fc < f[d])) {
                for (int q = 0; q < d; q++) worst[q] = xt[q];
                f[d] = fc;
            } else {
                for (int i = 1; i <= d; i++) {
                    for (int q = 0; q < d; q++) v[i * d + q] = v[q] + 0.5f * (v[i * d + q] - v[q]);
                    f[i] = fl_cost(c, &v[i * d], k);
                }
            }
        }
    }
    int best = 0;
    for (int i = 1; i <= d; i++) if (f[i] < f[best]) best = i;
    float xb[FL_MAX_DIM];
    for (int q = 0; q < d; q++) xb[q] = v[best * d + q];
    fl_clamp(c, xb, k);
    float fb = fl_cost(c, xb, k);
    if (fb <= f_start) {
        for (int q = 0; q < d; q++) x[q] = xb[q];
        return fb;
    }
    return f_start;
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

kernel void batch_flare_fit(
    device const float* all_t        [[buffer(0)]],
    device const float* all_y        [[buffer(1)]],
    device const float* all_w        [[buffer(2)]],
    device const float* all_s        [[buffer(3)]],
    device const int*   offsets      [[buffer(4)]],
    device const float* win_params   [[buffer(5)]],   // [n_windows * 6]
    device float*       out_theta    [[buffer(6)]],   // [n_windows * FL_MAX_DIM]
    device int*         out_k        [[buffer(7)]],   // [n_windows]
    device float*       out_bic      [[buffer(8)]],   // [n_windows * (FL_MAX_K + 1)]
    device int*         out_nbic     [[buffer(9)]],   // [n_windows]
    device const int*   cfg_i        [[buffer(10)]],  // [n_windows, max_k, n_baseline, iters_per_dim]
    device const float* cfg_f        [[buffer(11)]],  // [min_delta_bic, area, min_cov, tol_abs, tol_rel]
    uint tid  [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint win = tid / FL_LANES;
    int n_windows = cfg_i[0];
    if (int(win) >= n_windows) return;
    int max_k = min(cfg_i[1], FL_MAX_K);
    int nb = cfg_i[2];
    int iters_per_dim = cfg_i[3];
    float min_dbic = cfg_f[0];

    int start = offsets[win];
    FlareCtx c;
    c.t = all_t + start;
    c.y = all_y + start;
    c.w = all_w + start;
    c.s = all_s + start;
    c.n = offsets[win + 1] - start;
    c.nb = nb;
    device const float* wp = win_params + win * 6;
    c.half_exp = wp[0];
    c.tp_lo = wp[1];
    c.tp_hi = wp[2];
    c.lt_lo = wp[3];
    c.lt_hi = wp[4];
    c.cadence = wp[5];
    c.area = cfg_f[1];
    c.min_cov = cfg_f[2];
    c.lane = lane;

    float ln_n = log(float(c.n));
    float best_theta[FL_MAX_DIM];
    float theta[FL_MAX_DIM];
    for (int q = 0; q < FL_MAX_DIM; q++) { best_theta[q] = 0.0f; theta[q] = 0.0f; }
    float coef[FL_MAX_P];
    float best_bic = fl_profile(c, best_theta, 0, coef) + float(nb) * ln_n;
    float bics[FL_MAX_K + 1];
    bics[0] = best_bic;
    int nbic = 1, best_k = 0;

    for (int k = 1; k <= max_k; k++) {
        if (c.n < 3 * k + nb + 1) break;
        for (int q = 0; q < 2 * (k - 1); q++) theta[q] = best_theta[q];
        fl_seed(c, theta, k - 1);
        int iters = iters_per_dim * 2 * k;
        fl_nelder_mead(c, theta, k, iters, cfg_f[3], cfg_f[4]);
        float chi2 = fl_nelder_mead(c, theta, k, iters, cfg_f[3], cfg_f[4]);
        float kb = chi2 + float(3 * k + nb) * ln_n;
        bics[nbic++] = kb;
        if (kb < best_bic - min_dbic) {
            best_bic = kb;
            best_k = k;
            for (int q = 0; q < 2 * k; q++) best_theta[q] = theta[q];
        } else {
            break;
        }
    }

    if (lane == 0) {
        for (int q = 0; q < FL_MAX_DIM; q++) out_theta[win * FL_MAX_DIM + q] = best_theta[q];
        out_k[win] = best_k;
        for (int q = 0; q <= FL_MAX_K; q++) out_bic[win * (FL_MAX_K + 1) + q] = q < nbic ? bics[q] : NAN;
        out_nbic[win] = nbic;
    }
}
