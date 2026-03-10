// CUDA kernel for batch 2D Gaussian Process fitting (time × wavelength).
//
// One block per source. Threads parallelize over hyperparameter combos
// (amp × ls_time × ls_wave). Fits an anisotropic RBF kernel across all
// bands jointly, then predicts at a (time, wavelength) query grid.

#include <math.h>

#define GP2D_MAX_M    40   // max subsampled training points per source
#define GP2D_MAX_PRED 300  // max prediction points (n_time × n_wave)
#define GP2D_MAX_HP   64   // max hyperparameter combos

// State layout per source:
//   alpha[40] + x_time[40] + x_wave[40] + amp + inv_2lst2 + inv_2lsw2 + y_mean + m
#define GP2D_STATE_SIZE 125

// =========================================================================
// Device helpers
// =========================================================================

__device__ inline double gp2d_rbf(
    double t1, double w1, double t2, double w2,
    double amp, double inv_2lst2, double inv_2lsw2)
{
    double dt = t1 - t2;
    double dw = w1 - w2;
    return amp * exp(-dt * dt * inv_2lst2 - dw * dw * inv_2lsw2);
}

__device__ bool gp2d_cholesky(double* a, int n) {
    for (int j = 0; j < n; j++) {
        double s = a[j * n + j];
        for (int k = 0; k < j; k++)
            s -= a[j * n + k] * a[j * n + k];
        if (s <= 0.0) return false;
        a[j * n + j] = sqrt(s);
        double ljj = a[j * n + j];
        for (int i = j + 1; i < n; i++) {
            double si = a[i * n + j];
            for (int k = 0; k < j; k++)
                si -= a[i * n + k] * a[j * n + k];
            a[i * n + j] = si / ljj;
        }
        for (int i = 0; i < j; i++)
            a[i * n + j] = 0.0;
    }
    return true;
}

__device__ void gp2d_solve_l(const double* l, const double* b, double* x, int n) {
    for (int i = 0; i < n; i++) {
        double s = b[i];
        for (int j = 0; j < i; j++) s -= l[i * n + j] * x[j];
        x[i] = s / l[i * n + i];
    }
}

__device__ void gp2d_solve_lt(const double* l, const double* b, double* x, int n) {
    for (int i = n - 1; i >= 0; i--) {
        double s = b[i];
        for (int j = i + 1; j < n; j++) s -= l[j * n + i] * x[j];
        x[i] = s / l[i * n + i];
    }
}

// Try one (amp, ls_time, ls_wave) combo. Returns train RMS.
__device__ double gp2d_try_hyperparams(
    const double* sub_t, const double* sub_w,
    const double* sub_v, const double* sub_nv,
    int m, double y_mean,
    double amp, double inv_2lst2, double inv_2lsw2,
    double* K, double* alpha, double* tmp)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j <= i; j++) {
            double v = gp2d_rbf(sub_t[i], sub_w[i], sub_t[j], sub_w[j],
                                amp, inv_2lst2, inv_2lsw2);
            K[i * m + j] = v;
            K[j * m + i] = v;
        }
        double nv = sub_nv[i];
        if (nv < 1e-10) nv = 1e-10;
        K[i * m + i] += nv;
    }

    if (!gp2d_cholesky(K, m)) return 1e99;

    double y_c[GP2D_MAX_M];
    for (int i = 0; i < m; i++) y_c[i] = sub_v[i] - y_mean;
    gp2d_solve_l(K, y_c, tmp, m);
    gp2d_solve_lt(K, tmp, alpha, m);

    // Negative log marginal likelihood (NLML):
    // 0.5 * y' * alpha + sum(log(diag(L))) + 0.5 * m * log(2*pi)
    double data_fit = 0.0;
    for (int i = 0; i < m; i++) data_fit += y_c[i] * alpha[i];
    double log_det = 0.0;
    for (int i = 0; i < m; i++) log_det += log(K[i * m + i]);
    double nlml = 0.5 * data_fit + log_det + 0.5 * (double)m * log(2.0 * M_PI);
    return isfinite(nlml) ? nlml : 1e99;
}

// =========================================================================
// Kernel: Fit 2D GP + predict at query grid
// =========================================================================
//
// One BLOCK per source. blockDim.x = n_hp_total.
// Each thread evaluates one (amp, ls_t, ls_w) combo.
// Thread 0 picks winner, refits, then all threads predict.
//
// Shared memory layout:
//   double sub_t[GP2D_MAX_M]
//   double sub_w[GP2D_MAX_M]
//   double sub_v[GP2D_MAX_M]
//   double sub_nv[GP2D_MAX_M]
//   double scores[GP2D_MAX_HP]
//   double y_mean[1]
//   double best_amp[1], best_inv2lst2[1], best_inv2lsw2[1]
//   double alpha[GP2D_MAX_M]
//   double L[GP2D_MAX_M * GP2D_MAX_M]
//   int    m_val[1], winner[1]

extern "C" __global__ void batch_gp2d_fit_predict(
    const double* __restrict__ all_times,
    const double* __restrict__ all_waves,
    const double* __restrict__ all_mags,
    const double* __restrict__ all_noise_var,
    const int*    __restrict__ src_offsets,    // [n_sources+1]
    const double* __restrict__ query_times,    // [n_pred] (time for each pred point)
    const double* __restrict__ query_waves,    // [n_pred] (wave for each pred point)
    const double* __restrict__ hp_amps,        // [n_hp_amp]
    const double* __restrict__ hp_lst,         // [n_hp_lst]
    const double* __restrict__ hp_lsw,         // [n_hp_lsw]
    double* __restrict__ gp_state,             // [n_sources * GP2D_STATE_SIZE]
    double* __restrict__ pred_grid,            // [n_sources * n_pred]
    double* __restrict__ std_grid,             // [n_sources * n_pred]
    int n_sources,
    int n_pred,
    int n_hp_amp,
    int n_hp_lst,
    int n_hp_lsw,
    int max_subsample)
{
    int src = blockIdx.x;
    int tid = threadIdx.x;
    int n_hp_total = n_hp_amp * n_hp_lst * n_hp_lsw;

    if (src >= n_sources) return;

    // Shared memory
    extern __shared__ char smem[];
    double* sh_sub_t      = (double*)smem;
    double* sh_sub_w      = sh_sub_t + GP2D_MAX_M;
    double* sh_sub_v      = sh_sub_w + GP2D_MAX_M;
    double* sh_sub_nv     = sh_sub_v + GP2D_MAX_M;
    double* sh_scores     = sh_sub_nv + GP2D_MAX_M;
    double* sh_y_mean     = sh_scores + GP2D_MAX_HP;
    double* sh_best_amp   = sh_y_mean + 1;
    double* sh_best_inv_t = sh_best_amp + 1;
    double* sh_best_inv_w = sh_best_inv_t + 1;
    double* sh_alpha      = sh_best_inv_w + 1;
    double* sh_L          = sh_alpha + GP2D_MAX_M;
    int*    sh_m_val      = (int*)(sh_L + GP2D_MAX_M * GP2D_MAX_M);
    int*    sh_winner     = sh_m_val + 1;

    int obs_start = src_offsets[src];
    int obs_end   = src_offsets[src + 1];
    int n_obs     = obs_end - obs_start;

    double* out_state = gp_state + (long long)src * GP2D_STATE_SIZE;
    double* out_pred  = pred_grid + (long long)src * n_pred;
    double* out_std   = std_grid  + (long long)src * n_pred;

    // Thread 0 initializes
    if (tid == 0) {
        for (int i = 0; i < GP2D_STATE_SIZE; i++) out_state[i] = 0.0;
        for (int i = 0; i < n_pred; i++) {
            out_pred[i] = nan("");
            out_std[i]  = nan("");
        }

        if (n_obs >= 3) {
            int m = n_obs;
            if (m > max_subsample) m = max_subsample;
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
                double step = (double)(n_obs - 1) / (double)(m - 1);
                for (int i = 0; i < m; i++) {
                    int idx = (int)(i * step + 0.5);
                    if (idx >= n_obs) idx = n_obs - 1;
                    sh_sub_t[i]  = all_times[obs_start + idx];
                    sh_sub_w[i]  = all_waves[obs_start + idx];
                    sh_sub_v[i]  = all_mags[obs_start + idx];
                    sh_sub_nv[i] = all_noise_var[obs_start + idx];
                }
            }

            double ym = 0.0;
            for (int i = 0; i < m; i++) ym += sh_sub_v[i];
            ym /= (double)m;

            sh_y_mean[0] = ym;
            sh_m_val[0]  = m;
        } else {
            sh_m_val[0] = 0;
        }
    }

    __syncthreads();

    int m = sh_m_val[0];
    if (m < 3) return;
    double y_mean = sh_y_mean[0];

    // ---- Each thread evaluates one (amp, ls_t, ls_w) combo ----
    double my_score = 1e99;
    double my_amp = 0.0;
    double my_inv_t = 0.0;
    double my_inv_w = 0.0;

    if (tid < n_hp_total) {
        int ia  = tid / (n_hp_lst * n_hp_lsw);
        int rem = tid % (n_hp_lst * n_hp_lsw);
        int it  = rem / n_hp_lsw;
        int iw  = rem % n_hp_lsw;

        double amp  = hp_amps[ia];
        double lst  = hp_lst[it];
        double lsw  = hp_lsw[iw];

        if (lst >= 0.1 && lsw >= 0.001) {
            double inv_2lst2 = 0.5 / (lst * lst);
            double inv_2lsw2 = 0.5 / (lsw * lsw);
            my_amp   = amp;
            my_inv_t = inv_2lst2;
            my_inv_w = inv_2lsw2;

            // Local scratch
            double K[GP2D_MAX_M * GP2D_MAX_M];
            double alpha[GP2D_MAX_M];
            double tmp[GP2D_MAX_M];

            my_score = gp2d_try_hyperparams(
                sh_sub_t, sh_sub_w, sh_sub_v, sh_sub_nv, m, y_mean,
                amp, inv_2lst2, inv_2lsw2, K, alpha, tmp);
        }
    }

    sh_scores[tid] = my_score;
    __syncthreads();

    // ---- Thread 0 finds winner ----
    if (tid == 0) {
        double best = 1e99;
        int best_idx = 0;
        for (int i = 0; i < n_hp_total && i < GP2D_MAX_HP; i++) {
            if (sh_scores[i] < best) {
                best = sh_scores[i];
                best_idx = i;
            }
        }
        sh_winner[0] = best_idx;
    }
    __syncthreads();

    if (tid == sh_winner[0]) {
        sh_best_amp[0]   = my_amp;
        sh_best_inv_t[0] = my_inv_t;
        sh_best_inv_w[0] = my_inv_w;
    }
    __syncthreads();

    double amp      = sh_best_amp[0];
    double inv_2lst2 = sh_best_inv_t[0];
    double inv_2lsw2 = sh_best_inv_w[0];

    if (amp == 0.0) return;  // all combos failed

    // ---- Thread 0 refits with best hyperparameters ----
    if (tid == 0) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j <= i; j++) {
                double v = gp2d_rbf(sh_sub_t[i], sh_sub_w[i],
                                    sh_sub_t[j], sh_sub_w[j],
                                    amp, inv_2lst2, inv_2lsw2);
                sh_L[i * m + j] = v;
                sh_L[j * m + i] = v;
            }
            double nv = sh_sub_nv[i];
            if (nv < 1e-10) nv = 1e-10;
            sh_L[i * m + i] += nv;
        }

        if (!gp2d_cholesky(sh_L, m)) {
            sh_m_val[0] = 0;
        } else {
            double y_c[GP2D_MAX_M];
            double tmp[GP2D_MAX_M];
            for (int i = 0; i < m; i++) y_c[i] = sh_sub_v[i] - y_mean;
            gp2d_solve_l(sh_L, y_c, tmp, m);
            gp2d_solve_lt(sh_L, tmp, sh_alpha, m);

            // Store state:
            //   [0..40)  alpha
            //   [40..80) x_time
            //   [80..120) x_wave
            //   120: amp, 121: inv_2lst2, 122: inv_2lsw2, 123: y_mean, 124: m
            for (int i = 0; i < m; i++) out_state[i] = sh_alpha[i];
            for (int i = m; i < GP2D_MAX_M; i++) out_state[i] = 0.0;
            for (int i = 0; i < m; i++) out_state[GP2D_MAX_M + i] = sh_sub_t[i];
            for (int i = m; i < GP2D_MAX_M; i++) out_state[GP2D_MAX_M + i] = 0.0;
            for (int i = 0; i < m; i++) out_state[2 * GP2D_MAX_M + i] = sh_sub_w[i];
            for (int i = m; i < GP2D_MAX_M; i++) out_state[2 * GP2D_MAX_M + i] = 0.0;
            out_state[120] = amp;
            out_state[121] = inv_2lst2;
            out_state[122] = inv_2lsw2;
            out_state[123] = y_mean;
            out_state[124] = (double)m;
        }
    }
    __syncthreads();

    m = sh_m_val[0];
    if (m < 3) return;

    // ---- All threads cooperate on predictions ----
    for (int q = tid; q < n_pred; q += blockDim.x) {
        double qt = query_times[q];
        double qw = query_waves[q];

        double k_star[GP2D_MAX_M];
        double dot = 0.0;
        for (int i = 0; i < m; i++) {
            k_star[i] = gp2d_rbf(qt, qw, sh_sub_t[i], sh_sub_w[i],
                                  amp, inv_2lst2, inv_2lsw2);
            dot += k_star[i] * sh_alpha[i];
        }
        out_pred[q] = dot + y_mean;

        // Variance: k** - v^T v where L v = k_star
        double v[GP2D_MAX_M];
        gp2d_solve_l(sh_L, k_star, v, m);
        double vtv = 0.0;
        for (int i = 0; i < m; i++) vtv += v[i] * v[i];
        double var = amp - vtv;
        if (var < 1e-10) var = 1e-10;
        out_std[q] = sqrt(var);
    }
}

// =========================================================================
// Host-side launch wrapper
// =========================================================================

extern "C" void launch_batch_gp2d_fit_predict(
    const double* all_times,
    const double* all_waves,
    const double* all_mags,
    const double* all_noise_var,
    const int*    src_offsets,
    const double* query_times,
    const double* query_waves,
    const double* hp_amps,
    const double* hp_lst,
    const double* hp_lsw,
    double* gp_state,
    double* pred_grid,
    double* std_grid,
    int n_sources,
    int n_pred,
    int n_hp_amp,
    int n_hp_lst,
    int n_hp_lsw,
    int max_subsample,
    int /*grid*/, int block)
{
    // Shared memory: 4*GP2D_MAX_M + GP2D_MAX_HP + 4 + GP2D_MAX_M + GP2D_MAX_M*GP2D_MAX_M doubles + 2 ints
    size_t smem_bytes = (4 * GP2D_MAX_M + GP2D_MAX_HP + 4 + GP2D_MAX_M + GP2D_MAX_M * GP2D_MAX_M) * sizeof(double)
                        + 2 * sizeof(int);
    batch_gp2d_fit_predict<<<n_sources, block, smem_bytes>>>(
        all_times, all_waves, all_mags, all_noise_var, src_offsets,
        query_times, query_waves,
        hp_amps, hp_lst, hp_lsw,
        gp_state, pred_grid, std_grid,
        n_sources, n_pred, n_hp_amp, n_hp_lst, n_hp_lsw, max_subsample);
}
