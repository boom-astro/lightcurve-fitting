// Metal compute shader for batch 2D Gaussian Process fitting (time x wavelength).
//
// One threadgroup per source. Threads parallelize over hyperparameter combos
// (amp x ls_time x ls_wave). Fits an anisotropic RBF kernel across all
// bands jointly, then predicts at a (time, wavelength) query grid.
//
// Ported from cuda/gp2d.cu — all doubles converted to float for Metal.

#include <metal_stdlib>
using namespace metal;

#define GP2D_MAX_M    40   // max subsampled training points per source
#define GP2D_MAX_PRED 300  // max prediction points (n_time x n_wave)
#define GP2D_MAX_HP   64   // max hyperparameter combos

// State layout per source:
//   alpha[40] + x_time[40] + x_wave[40] + amp + inv_2lst2 + inv_2lsw2 + y_mean + m
#define GP2D_STATE_SIZE 125

// =========================================================================
// Inline helpers
// =========================================================================

inline float gp2d_rbf(
    float t1, float w1, float t2, float w2,
    float amp, float inv_2lst2, float inv_2lsw2)
{
    float dt = t1 - t2;
    float dw = w1 - w2;
    return amp * exp(-dt * dt * inv_2lst2 - dw * dw * inv_2lsw2);
}

inline bool gp2d_cholesky(thread float* a, int n) {
    for (int j = 0; j < n; j++) {
        float s = a[j * n + j];
        for (int k = 0; k < j; k++)
            s -= a[j * n + k] * a[j * n + k];
        if (s <= 0.0f) return false;
        a[j * n + j] = sqrt(s);
        float ljj = a[j * n + j];
        for (int i = j + 1; i < n; i++) {
            float si = a[i * n + j];
            for (int k = 0; k < j; k++)
                si -= a[i * n + k] * a[j * n + k];
            a[i * n + j] = si / ljj;
        }
        for (int i = 0; i < j; i++)
            a[i * n + j] = 0.0f;
    }
    return true;
}

inline bool gp2d_cholesky_tg(threadgroup float* a, int n) {
    for (int j = 0; j < n; j++) {
        float s = a[j * n + j];
        for (int k = 0; k < j; k++)
            s -= a[j * n + k] * a[j * n + k];
        if (s <= 0.0f) return false;
        a[j * n + j] = sqrt(s);
        float ljj = a[j * n + j];
        for (int i = j + 1; i < n; i++) {
            float si = a[i * n + j];
            for (int k = 0; k < j; k++)
                si -= a[i * n + k] * a[j * n + k];
            a[i * n + j] = si / ljj;
        }
        for (int i = 0; i < j; i++)
            a[i * n + j] = 0.0f;
    }
    return true;
}

inline void gp2d_solve_l(const thread float* l, const thread float* b, thread float* x, int n) {
    for (int i = 0; i < n; i++) {
        float s = b[i];
        for (int j = 0; j < i; j++) s -= l[i * n + j] * x[j];
        x[i] = s / l[i * n + i];
    }
}

inline void gp2d_solve_l_tg(const threadgroup float* l, const thread float* b, thread float* x, int n) {
    for (int i = 0; i < n; i++) {
        float s = b[i];
        for (int j = 0; j < i; j++) s -= l[i * n + j] * x[j];
        x[i] = s / l[i * n + i];
    }
}

inline void gp2d_solve_lt(const thread float* l, const thread float* b, thread float* x, int n) {
    for (int i = n - 1; i >= 0; i--) {
        float s = b[i];
        for (int j = i + 1; j < n; j++) s -= l[j * n + i] * x[j];
        x[i] = s / l[i * n + i];
    }
}

inline void gp2d_solve_l_tg_to_tg(const threadgroup float* l, const threadgroup float* b, threadgroup float* x, int n) {
    for (int i = 0; i < n; i++) {
        float s = b[i];
        for (int j = 0; j < i; j++) s -= l[i * n + j] * x[j];
        x[i] = s / l[i * n + i];
    }
}

inline void gp2d_solve_lt_tg_to_tg(const threadgroup float* l, const threadgroup float* b, threadgroup float* x, int n) {
    for (int i = n - 1; i >= 0; i--) {
        float s = b[i];
        for (int j = i + 1; j < n; j++) s -= l[j * n + i] * x[j];
        x[i] = s / l[i * n + i];
    }
}

// Try one (amp, ls_time, ls_wave) combo. Returns NLML score.
inline float gp2d_try_hyperparams(
    const threadgroup float* sub_t, const threadgroup float* sub_w,
    const threadgroup float* sub_v, const threadgroup float* sub_nv,
    int m, float y_mean,
    float amp, float inv_2lst2, float inv_2lsw2,
    thread float* K, thread float* alpha, thread float* tmp)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j <= i; j++) {
            float v = gp2d_rbf(sub_t[i], sub_w[i], sub_t[j], sub_w[j],
                                amp, inv_2lst2, inv_2lsw2);
            K[i * m + j] = v;
            K[j * m + i] = v;
        }
        float nv = sub_nv[i];
        if (nv < 1e-6f) nv = 1e-6f;
        K[i * m + i] += nv;
    }

    if (!gp2d_cholesky(K, m)) return 1e30f;

    float y_c[GP2D_MAX_M];
    for (int i = 0; i < m; i++) y_c[i] = sub_v[i] - y_mean;
    gp2d_solve_l(K, y_c, tmp, m);
    gp2d_solve_lt(K, tmp, alpha, m);

    // Negative log marginal likelihood (NLML):
    // 0.5 * y' * alpha + sum(log(diag(L))) + 0.5 * m * log(2*pi)
    float data_fit = 0.0f;
    for (int i = 0; i < m; i++) data_fit += y_c[i] * alpha[i];
    float log_det = 0.0f;
    for (int i = 0; i < m; i++) log_det += log(K[i * m + i]);
    float nlml = 0.5f * data_fit + log_det + 0.5f * (float)m * log(2.0f * M_PI_F);
    return isfinite(nlml) ? nlml : 1e30f;
}

// =========================================================================
// Kernel: Fit 2D GP + predict at query grid
// =========================================================================
//
// One THREADGROUP per source. threads_per_threadgroup = n_hp_total.
// Each thread evaluates one (amp, ls_t, ls_w) combo.
// Thread 0 picks winner, refits, then all threads predict.

struct Gp2dParams {
    int n_sources;
    int n_pred;
    int n_hp_amp;
    int n_hp_lst;
    int n_hp_lsw;
    int max_subsample;
};

kernel void batch_gp2d_fit_predict(
    device const float* all_times       [[buffer(0)]],
    device const float* all_waves       [[buffer(1)]],
    device const float* all_mags        [[buffer(2)]],
    device const float* all_noise_var   [[buffer(3)]],
    device const int*   src_offsets     [[buffer(4)]],
    device const float* query_times     [[buffer(5)]],
    device const float* query_waves     [[buffer(6)]],
    device const float* hp_amps         [[buffer(7)]],
    device const float* hp_lst          [[buffer(8)]],
    device const float* hp_lsw          [[buffer(9)]],
    device float*       gp_state        [[buffer(10)]],
    device float*       pred_grid       [[buffer(11)]],
    device float*       std_grid        [[buffer(12)]],
    device const Gp2dParams& params     [[buffer(13)]],
    uint bid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]])
{
    int src = (int)bid;
    int n_hp_total = params.n_hp_amp * params.n_hp_lst * params.n_hp_lsw;

    if (src >= params.n_sources) return;

    // Threadgroup shared memory
    threadgroup float sh_sub_t[GP2D_MAX_M];
    threadgroup float sh_sub_w[GP2D_MAX_M];
    threadgroup float sh_sub_v[GP2D_MAX_M];
    threadgroup float sh_sub_nv[GP2D_MAX_M];
    threadgroup float sh_scores[GP2D_MAX_HP];
    threadgroup float sh_y_mean[1];
    threadgroup float sh_best_amp[1];
    threadgroup float sh_best_inv_t[1];
    threadgroup float sh_best_inv_w[1];
    threadgroup float sh_alpha[GP2D_MAX_M];
    threadgroup float sh_L[GP2D_MAX_M * GP2D_MAX_M];
    threadgroup int   sh_m_val[1];
    threadgroup int   sh_winner[1];

    int obs_start = src_offsets[src];
    int obs_end   = src_offsets[src + 1];
    int n_obs     = obs_end - obs_start;

    device float* out_state = gp_state + (long)src * GP2D_STATE_SIZE;
    device float* out_pred  = pred_grid + (long)src * params.n_pred;
    device float* out_std   = std_grid  + (long)src * params.n_pred;

    // Thread 0 initializes
    if (tid == 0) {
        for (int i = 0; i < GP2D_STATE_SIZE; i++) out_state[i] = 0.0f;
        for (int i = 0; i < params.n_pred; i++) {
            out_pred[i] = NAN;
            out_std[i]  = NAN;
        }

        if (n_obs >= 3) {
            int m = n_obs;
            if (m > params.max_subsample) m = params.max_subsample;
            if (m > GP2D_MAX_M) m = GP2D_MAX_M;

            if (n_obs <= m) {
                for (int i = 0; i < n_obs; i++) {
                    sh_sub_t[i]  = all_times[obs_start + i];
                    sh_sub_w[i]  = all_waves[obs_start + i];
                    sh_sub_v[i]  = all_mags[obs_start + i];
                    sh_sub_nv[i] = all_noise_var[obs_start + i];
                }
                m = n_obs;
            } else {
                // Uniform subsample
                float step = (float)(n_obs - 1) / (float)(m - 1);
                for (int i = 0; i < m; i++) {
                    int idx = (int)(i * step + 0.5f);
                    if (idx >= n_obs) idx = n_obs - 1;
                    sh_sub_t[i]  = all_times[obs_start + idx];
                    sh_sub_w[i]  = all_waves[obs_start + idx];
                    sh_sub_v[i]  = all_mags[obs_start + idx];
                    sh_sub_nv[i] = all_noise_var[obs_start + idx];
                }
            }

            float ym = 0.0f;
            for (int i = 0; i < m; i++) ym += sh_sub_v[i];
            ym /= (float)m;

            sh_y_mean[0] = ym;
            sh_m_val[0]  = m;
        } else {
            sh_m_val[0] = 0;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    int m = sh_m_val[0];
    if (m < 3) return;
    float y_mean = sh_y_mean[0];

    // ---- Each thread evaluates one (amp, ls_t, ls_w) combo ----
    float my_score = 1e30f;
    float my_amp = 0.0f;
    float my_inv_t = 0.0f;
    float my_inv_w = 0.0f;

    if ((int)tid < n_hp_total) {
        int ia  = (int)tid / (params.n_hp_lst * params.n_hp_lsw);
        int rem = (int)tid % (params.n_hp_lst * params.n_hp_lsw);
        int it  = rem / params.n_hp_lsw;
        int iw  = rem % params.n_hp_lsw;

        float amp = hp_amps[ia];
        float lst = hp_lst[it];
        float lsw = hp_lsw[iw];

        if (lst >= 0.1f && lsw >= 0.001f) {
            float inv_2lst2 = 0.5f / (lst * lst);
            float inv_2lsw2 = 0.5f / (lsw * lsw);
            my_amp   = amp;
            my_inv_t = inv_2lst2;
            my_inv_w = inv_2lsw2;

            // Local scratch
            float K[GP2D_MAX_M * GP2D_MAX_M];
            float alpha[GP2D_MAX_M];
            float tmp[GP2D_MAX_M];

            my_score = gp2d_try_hyperparams(
                sh_sub_t, sh_sub_w, sh_sub_v, sh_sub_nv, m, y_mean,
                amp, inv_2lst2, inv_2lsw2, K, alpha, tmp);
        }
    }

    if ((int)tid < GP2D_MAX_HP) {
        sh_scores[tid] = my_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Thread 0 finds winner ----
    if (tid == 0) {
        float best = 1e30f;
        int best_idx = 0;
        for (int i = 0; i < n_hp_total && i < GP2D_MAX_HP; i++) {
            if (sh_scores[i] < best) {
                best = sh_scores[i];
                best_idx = i;
            }
        }
        sh_winner[0] = best_idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if ((int)tid == sh_winner[0]) {
        sh_best_amp[0]   = my_amp;
        sh_best_inv_t[0] = my_inv_t;
        sh_best_inv_w[0] = my_inv_w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float amp       = sh_best_amp[0];
    float inv_2lst2 = sh_best_inv_t[0];
    float inv_2lsw2 = sh_best_inv_w[0];

    if (amp == 0.0f) return;  // all combos failed

    // ---- Thread 0 refits with best hyperparameters ----
    if (tid == 0) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j <= i; j++) {
                float v = gp2d_rbf(sh_sub_t[i], sh_sub_w[i],
                                    sh_sub_t[j], sh_sub_w[j],
                                    amp, inv_2lst2, inv_2lsw2);
                sh_L[i * m + j] = v;
                sh_L[j * m + i] = v;
            }
            float nv = sh_sub_nv[i];
            if (nv < 1e-6f) nv = 1e-6f;
            sh_L[i * m + i] += nv;
        }

        if (!gp2d_cholesky_tg(sh_L, m)) {
            sh_m_val[0] = 0;
        } else {
            threadgroup float y_c[GP2D_MAX_M];
            threadgroup float tmp[GP2D_MAX_M];
            for (int i = 0; i < m; i++) y_c[i] = sh_sub_v[i] - y_mean;
            gp2d_solve_l_tg_to_tg(sh_L, y_c, tmp, m);
            gp2d_solve_lt_tg_to_tg(sh_L, tmp, sh_alpha, m);

            // Store state:
            //   [0..40)  alpha
            //   [40..80) x_time
            //   [80..120) x_wave
            //   120: amp, 121: inv_2lst2, 122: inv_2lsw2, 123: y_mean, 124: m
            for (int i = 0; i < m; i++) out_state[i] = sh_alpha[i];
            for (int i = m; i < GP2D_MAX_M; i++) out_state[i] = 0.0f;
            for (int i = 0; i < m; i++) out_state[GP2D_MAX_M + i] = sh_sub_t[i];
            for (int i = m; i < GP2D_MAX_M; i++) out_state[GP2D_MAX_M + i] = 0.0f;
            for (int i = 0; i < m; i++) out_state[2 * GP2D_MAX_M + i] = sh_sub_w[i];
            for (int i = m; i < GP2D_MAX_M; i++) out_state[2 * GP2D_MAX_M + i] = 0.0f;
            out_state[120] = amp;
            out_state[121] = inv_2lst2;
            out_state[122] = inv_2lsw2;
            out_state[123] = y_mean;
            out_state[124] = (float)m;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    m = sh_m_val[0];
    if (m < 3) return;

    // ---- All threads cooperate on predictions ----
    for (int q = (int)tid; q < params.n_pred; q += (int)tpg) {
        float qt = query_times[q];
        float qw = query_waves[q];

        float k_star[GP2D_MAX_M];
        float dot = 0.0f;
        for (int i = 0; i < m; i++) {
            k_star[i] = gp2d_rbf(qt, qw, sh_sub_t[i], sh_sub_w[i],
                                  amp, inv_2lst2, inv_2lsw2);
            dot += k_star[i] * sh_alpha[i];
        }
        out_pred[q] = dot + y_mean;

        // Variance: k** - v^T v where L v = k_star
        float v[GP2D_MAX_M];
        gp2d_solve_l_tg(sh_L, k_star, v, m);
        float vtv = 0.0f;
        for (int i = 0; i < m; i++) vtv += v[i] * v[i];
        float var = amp - vtv;
        if (var < 1e-6f) var = 1e-6f;
        out_std[q] = sqrt(var);
    }
}
