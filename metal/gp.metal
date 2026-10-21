// Metal compute shaders for batch Gaussian Process fitting and prediction.
// Ported from cuda/gp.cu.
//
// Kernel 1: One threadgroup per band. Threads parallelize over
// hyperparameter combos and prediction points.
//
// Kernel 2: One thread per observation for mean prediction using fitted state.

#include <metal_stdlib>
using namespace metal;

#define GP_MAX_M   25
#define GP_N_PRED  50
// GP state layout per band: alpha[25] + x_train[25] + amp + inv_2ls2 + y_mean + m
#define GP_STATE_SIZE 54

// Max hyperparameter combos (threads per threadgroup in kernel 1)
#define GP_MAX_HP  32

// =========================================================================
// Inline helpers
// =========================================================================

inline float gp_rbf(float x1, float x2, float amp, float inv_2ls2) {
    float d = x1 - x2;
    return amp * exp(-d * d * inv_2ls2);
}

// In-place Cholesky of n x n symmetric PD matrix (row-major).
// Returns false if not PD.
inline bool gp_cholesky(thread float* a, int n) {
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

// Overload for threadgroup memory (used in refit + prediction)
inline bool gp_cholesky(threadgroup float* a, int n) {
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

inline void gp_solve_l(thread const float* l, thread const float* b, thread float* x, int n) {
    for (int i = 0; i < n; i++) {
        float s = b[i];
        for (int j = 0; j < i; j++) s -= l[i * n + j] * x[j];
        x[i] = s / l[i * n + i];
    }
}

inline void gp_solve_lt(thread const float* l, thread const float* b, thread float* x, int n) {
    for (int i = n - 1; i >= 0; i--) {
        float s = b[i];
        for (int j = i + 1; j < n; j++) s -= l[j * n + i] * x[j];
        x[i] = s / l[i * n + i];
    }
}

// Overloads for threadgroup L, threadgroup b -> threadgroup x
inline void gp_solve_l(threadgroup const float* l, threadgroup const float* b, threadgroup float* x, int n) {
    for (int i = 0; i < n; i++) {
        float s = b[i];
        for (int j = 0; j < i; j++) s -= l[i * n + j] * x[j];
        x[i] = s / l[i * n + i];
    }
}

inline void gp_solve_lt(threadgroup const float* l, threadgroup const float* b, threadgroup float* x, int n) {
    for (int i = n - 1; i >= 0; i--) {
        float s = b[i];
        for (int j = i + 1; j < n; j++) s -= l[j * n + i] * x[j];
        x[i] = s / l[i * n + i];
    }
}

// Overloads for threadgroup L, thread k_star -> thread v (used in prediction)
inline void gp_solve_l(threadgroup const float* l, thread const float* b, thread float* x, int n) {
    for (int i = 0; i < n; i++) {
        float s = b[i];
        for (int j = 0; j < i; j++) s -= l[i * n + j] * x[j];
        x[i] = s / l[i * n + i];
    }
}

// Try one (amp, lengthscale) combo: build K, Cholesky, compute alpha & NLML.
inline float gp_try_hyperparams(
    threadgroup const float* sub_t, threadgroup const float* sub_v, threadgroup const float* sub_nv,
    int m, float y_mean,
    float amp, float inv_2ls2,
    thread float* K, thread float* alpha, thread float* tmp)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j <= i; j++) {
            float v = gp_rbf(sub_t[i], sub_t[j], amp, inv_2ls2);
            K[i * m + j] = v;
            K[j * m + i] = v;
        }
        float nv = sub_nv[i];
        if (nv < 1e-6f) nv = 1e-6f;
        K[i * m + i] += nv;
    }

    if (!gp_cholesky(K, m)) return 1e30f;

    float y_c[GP_MAX_M];
    for (int i = 0; i < m; i++) y_c[i] = sub_v[i] - y_mean;
    gp_solve_l(K, y_c, tmp, m);
    gp_solve_lt(K, tmp, alpha, m);

    // Negative log marginal likelihood (NLML):
    // 0.5 * y' * alpha + sum(log(diag(L))) + 0.5 * m * log(2*pi)
    float data_fit = 0.0f;
    for (int i = 0; i < m; i++) data_fit += y_c[i] * alpha[i];
    float log_det = 0.0f;
    for (int i = 0; i < m; i++) log_det += log(K[i * m + i]);
    float nlml = 0.5f * data_fit + log_det + 0.5f * float(m) * log(2.0f * M_PI_F);
    return isfinite(nlml) ? nlml : 1e30f;
}

// =========================================================================
// Kernel 1: Fit GP + predict at grid points
// =========================================================================
//
// One THREADGROUP per band. threads_per_threadgroup = n_hp_total.
// Each thread evaluates one hyperparameter combo in parallel.
// Thread 0 picks the winner, refits, then all threads cooperate on
// the GP_N_PRED grid predictions.
//
// Threadgroup memory layout (explicit arrays):
//   float sh_sub_t[GP_MAX_M]
//   float sh_sub_v[GP_MAX_M]
//   float sh_sub_nv[GP_MAX_M]
//   float sh_scores[GP_MAX_HP]
//   float sh_y_mean[1]
//   float sh_best_amp[1]
//   float sh_best_inv[1]
//   float sh_alpha[GP_MAX_M]
//   float sh_L[GP_MAX_M * GP_MAX_M]
//   int   sh_m_val[1]
//   int   sh_winner[1]

kernel void batch_gp_fit_predict(
    device const float*  all_times      [[buffer(0)]],
    device const float*  all_mags       [[buffer(1)]],
    device const float*  all_noise_var  [[buffer(2)]],
    device const int*    band_offsets   [[buffer(3)]],
    device const float*  query_times    [[buffer(4)]],
    device const float*  hp_amps        [[buffer(5)]],
    device const float*  hp_ls          [[buffer(6)]],
    device float*        gp_state       [[buffer(7)]],
    device float*        pred_grid      [[buffer(8)]],
    device float*        std_grid       [[buffer(9)]],
    constant int&        n_bands        [[buffer(10)]],
    constant int&        n_hp_amp       [[buffer(11)]],
    constant int&        n_hp_ls        [[buffer(12)]],
    constant int&        max_subsample  [[buffer(13)]],
    uint bid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]])
{
    int band = int(bid);
    int n_hp_total = n_hp_amp * n_hp_ls;

    if (band >= n_bands) return;

    // Threadgroup memory
    threadgroup float sh_sub_t[GP_MAX_M];
    threadgroup float sh_sub_v[GP_MAX_M];
    threadgroup float sh_sub_nv[GP_MAX_M];
    threadgroup float sh_scores[GP_MAX_HP];
    threadgroup float sh_y_mean[1];
    threadgroup float sh_best_amp[1];
    threadgroup float sh_best_inv[1];
    threadgroup float sh_alpha[GP_MAX_M];
    threadgroup float sh_L[GP_MAX_M * GP_MAX_M];
    threadgroup int   sh_m_val[1];
    threadgroup int   sh_winner[1];

    int obs_start = band_offsets[band];
    int obs_end   = band_offsets[band + 1];
    int n_obs     = obs_end - obs_start;

    device float* out_state = gp_state + long(band) * GP_STATE_SIZE;
    device float* out_pred  = pred_grid + long(band) * GP_N_PRED;
    device float* out_std   = std_grid  + long(band) * GP_N_PRED;

    // Thread 0 initializes outputs and loads subsampled data
    if (tid == 0) {
        for (int i = 0; i < GP_STATE_SIZE; i++) out_state[i] = 0.0f;
        for (int i = 0; i < GP_N_PRED; i++) {
            out_pred[i] = NAN;
            out_std[i]  = NAN;
        }

        if (n_obs >= 3) {
            int m = n_obs;
            if (m > max_subsample) m = max_subsample;
            if (m > GP_MAX_M) m = GP_MAX_M;

            if (n_obs <= m) {
                for (int i = 0; i < n_obs; i++) {
                    sh_sub_t[i]  = all_times[obs_start + i];
                    sh_sub_v[i]  = all_mags[obs_start + i];
                    sh_sub_nv[i] = all_noise_var[obs_start + i];
                }
                m = n_obs;
            } else {
                float step = float(n_obs - 1) / float(m - 1);
                for (int i = 0; i < m; i++) {
                    int idx = int(float(i) * step + 0.5f);
                    if (idx >= n_obs) idx = n_obs - 1;
                    sh_sub_t[i]  = all_times[obs_start + idx];
                    sh_sub_v[i]  = all_mags[obs_start + idx];
                    sh_sub_nv[i] = all_noise_var[obs_start + idx];
                }
            }

            float ym = 0.0f;
            for (int i = 0; i < m; i++) ym += sh_sub_v[i];
            ym /= float(m);

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

    // ---- Each thread evaluates one hyperparameter combo ----
    float my_score = 1e30f;
    float my_amp = 0.0f;
    float my_inv2ls2 = 0.0f;

    if (int(tid) < n_hp_total) {
        int ia = int(tid) / n_hp_ls;
        int il = int(tid) % n_hp_ls;
        float amp = hp_amps[ia];
        float ls  = hp_ls[il];

        if (ls >= 0.1f) {
            float inv_2ls2 = 0.5f / (ls * ls);
            my_amp = amp;
            my_inv2ls2 = inv_2ls2;

            // Local scratch for this thread's Cholesky
            float K[GP_MAX_M * GP_MAX_M];
            float alpha[GP_MAX_M];
            float tmp[GP_MAX_M];

            my_score = gp_try_hyperparams(
                sh_sub_t, sh_sub_v, sh_sub_nv, m, y_mean,
                amp, inv_2ls2, K, alpha, tmp);
        }
    }

    sh_scores[tid] = my_score;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Thread 0 finds the winner ----
    if (tid == 0) {
        float best = 1e30f;
        int best_idx = 0;
        for (int i = 0; i < n_hp_total && i < GP_MAX_HP; i++) {
            if (sh_scores[i] < best) {
                best = sh_scores[i];
                best_idx = i;
            }
        }
        sh_winner[0] = best_idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Winner thread writes its hyperparameters to threadgroup memory
    if (int(tid) == sh_winner[0]) {
        sh_best_amp[0] = my_amp;
        sh_best_inv[0] = my_inv2ls2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float amp      = sh_best_amp[0];
    float inv_2ls2 = sh_best_inv[0];

    if (amp == 0.0f && inv_2ls2 == 0.0f) return;  // all combos failed

    // ---- Thread 0 refits with best hyperparameters and stores state ----
    if (tid == 0) {
        // Rebuild K + Cholesky in threadgroup L
        for (int i = 0; i < m; i++) {
            for (int j = 0; j <= i; j++) {
                float v = gp_rbf(sh_sub_t[i], sh_sub_t[j], amp, inv_2ls2);
                sh_L[i * m + j] = v;
                sh_L[j * m + i] = v;
            }
            float nv = sh_sub_nv[i];
            if (nv < 1e-6f) nv = 1e-6f;
            sh_L[i * m + i] += nv;
        }

        if (!gp_cholesky(sh_L, m)) {
            sh_m_val[0] = 0;  // signal failure
        } else {
            threadgroup float y_c[GP_MAX_M];
            threadgroup float tmp[GP_MAX_M];
            for (int i = 0; i < m; i++) y_c[i] = sh_sub_v[i] - y_mean;
            gp_solve_l(sh_L, y_c, tmp, m);
            gp_solve_lt(sh_L, tmp, sh_alpha, m);

            // Store GP state
            for (int i = 0; i < m; i++) out_state[i] = sh_alpha[i];
            for (int i = m; i < GP_MAX_M; i++) out_state[i] = 0.0f;
            for (int i = 0; i < m; i++) out_state[GP_MAX_M + i] = sh_sub_t[i];
            for (int i = m; i < GP_MAX_M; i++) out_state[GP_MAX_M + i] = 0.0f;
            out_state[50] = amp;
            out_state[51] = inv_2ls2;
            out_state[52] = y_mean;
            out_state[53] = float(m);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    m = sh_m_val[0];
    if (m < 3) return;

    // ---- All threads cooperate on GP_N_PRED grid predictions ----
    for (int q = int(tid); q < GP_N_PRED; q += int(tpg)) {
        float t = query_times[q];

        float k_star[GP_MAX_M];
        float dot = 0.0f;
        for (int i = 0; i < m; i++) {
            k_star[i] = gp_rbf(t, sh_sub_t[i], amp, inv_2ls2);
            dot += k_star[i] * sh_alpha[i];
        }
        out_pred[q] = dot + y_mean;

        // Variance: k** - v^T v where L v = k_star
        float v[GP_MAX_M];
        gp_solve_l(sh_L, k_star, v, m);
        float vtv = 0.0f;
        for (int i = 0; i < m; i++) vtv += v[i] * v[i];
        float var = amp - vtv;
        if (var < 1e-6f) var = 1e-6f;
        out_std[q] = sqrt(var);
    }
}

// =========================================================================
// Kernel 2: Predict at observation points using fitted GP state
// =========================================================================
//
// One thread per observation. Mean prediction only (no std needed for chi2).

kernel void batch_gp_predict_obs(
    device const float*  gp_state    [[buffer(0)]],
    device const float*  all_times   [[buffer(1)]],
    device const int*    obs_to_band [[buffer(2)]],
    device float*        pred_obs    [[buffer(3)]],
    constant int&        total_obs   [[buffer(4)]],
    uint idx [[thread_position_in_grid]])
{
    if (int(idx) >= total_obs) return;

    int band = obs_to_band[idx];
    device const float* state = gp_state + long(band) * GP_STATE_SIZE;

    int m = int(state[53]);
    if (m <= 0 || m > GP_MAX_M) { pred_obs[idx] = NAN; return; }

    float amp      = state[50];
    float inv_2ls2 = state[51];
    float y_mean   = state[52];

    float t   = all_times[idx];
    float dot = 0.0f;
    for (int i = 0; i < m; i++) {
        float x_i = state[GP_MAX_M + i];
        dot += gp_rbf(t, x_i, amp, inv_2ls2) * state[i];
    }
    pred_obs[idx] = dot + y_mean;
}
