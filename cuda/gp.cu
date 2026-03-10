// CUDA kernels for batch Gaussian Process fitting and prediction.
//
// Kernel 1: One block per band. Within each block, threads parallelize over
// hyperparameter combos and prediction points. This gives n_bands blocks
// with n_hp threads each, much better GPU utilization than 1-thread-per-band.
//
// Kernel 2: One thread per observation for mean prediction using fitted state.

#include <math.h>

#define GP_MAX_M   25
#define GP_N_PRED  50
// GP state layout per band: alpha[25] + x_train[25] + amp + inv_2ls2 + y_mean + m
#define GP_STATE_SIZE 54

// Max hyperparameter combos (threads per block in kernel 1)
#define GP_MAX_HP  32

// =========================================================================
// Device helpers
// =========================================================================

__device__ inline double gp_rbf(double x1, double x2, double amp, double inv_2ls2) {
    double d = x1 - x2;
    return amp * exp(-d * d * inv_2ls2);
}

// In-place Cholesky of n x n symmetric PD matrix (row-major).
// Returns false if not PD.
__device__ bool gp_cholesky(double* a, int n) {
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

__device__ void gp_solve_l(const double* l, const double* b, double* x, int n) {
    for (int i = 0; i < n; i++) {
        double s = b[i];
        for (int j = 0; j < i; j++) s -= l[i * n + j] * x[j];
        x[i] = s / l[i * n + i];
    }
}

__device__ void gp_solve_lt(const double* l, const double* b, double* x, int n) {
    for (int i = n - 1; i >= 0; i--) {
        double s = b[i];
        for (int j = i + 1; j < n; j++) s -= l[j * n + i] * x[j];
        x[i] = s / l[i * n + i];
    }
}

// Try one (amp, lengthscale) combo: build K, Cholesky, compute alpha & train RMS.
__device__ double gp_try_hyperparams(
    const double* sub_t, const double* sub_v, const double* sub_nv,
    int m, double y_mean,
    double amp, double inv_2ls2,
    double* K, double* alpha, double* tmp)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j <= i; j++) {
            double v = gp_rbf(sub_t[i], sub_t[j], amp, inv_2ls2);
            K[i * m + j] = v;
            K[j * m + i] = v;
        }
        double nv = sub_nv[i];
        if (nv < 1e-10) nv = 1e-10;
        K[i * m + i] += nv;
    }

    if (!gp_cholesky(K, m)) return 1e99;

    double y_c[GP_MAX_M];
    for (int i = 0; i < m; i++) y_c[i] = sub_v[i] - y_mean;
    gp_solve_l(K, y_c, tmp, m);
    gp_solve_lt(K, tmp, alpha, m);

    // Negative log marginal likelihood (NLML):
    // 0.5 * y' * alpha + sum(log(diag(L))) + 0.5 * m * log(2*pi)
    // K now holds L from Cholesky; alpha = K^{-1} * y_c
    double data_fit = 0.0;
    for (int i = 0; i < m; i++) data_fit += y_c[i] * alpha[i];
    double log_det = 0.0;
    for (int i = 0; i < m; i++) log_det += log(K[i * m + i]);
    double nlml = 0.5 * data_fit + log_det + 0.5 * (double)m * log(2.0 * M_PI);
    return isfinite(nlml) ? nlml : 1e99;
}

// =========================================================================
// Kernel 1: Fit GP + predict at grid points
// =========================================================================
//
// One BLOCK per band. blockDim.x = n_hp_total (n_hp_amp * n_hp_ls).
// Each thread evaluates one hyperparameter combo in parallel.
// Thread 0 picks the winner, refits, then all threads cooperate on
// the GP_N_PRED grid predictions.
//
// Shared memory layout:
//   double sub_t[GP_MAX_M]       subsampled times
//   double sub_v[GP_MAX_M]       subsampled values
//   double sub_nv[GP_MAX_M]      subsampled noise variance
//   double scores[GP_MAX_HP]     per-thread RMS scores
//   int    winner_idx[1]         index of best thread
//   double y_mean[1]             shared y_mean
//   int    m_val[1]              shared m
//   double best_amp[1]           winning amplitude
//   double best_inv2ls2[1]       winning inv_2ls2
//   double alpha[GP_MAX_M]       final alpha (from refit)
//   double L[GP_MAX_M*GP_MAX_M]  final Cholesky factor (from refit)

extern "C" __global__ void batch_gp_fit_predict(
    const double* __restrict__ all_times,
    const double* __restrict__ all_mags,
    const double* __restrict__ all_noise_var,
    const int*    __restrict__ band_offsets,
    const double* __restrict__ query_times,   // [GP_N_PRED]
    const double* __restrict__ hp_amps,       // [n_hp_amp]
    const double* __restrict__ hp_ls,         // [n_hp_ls]
    double* __restrict__ gp_state,            // [n_bands * GP_STATE_SIZE]
    double* __restrict__ pred_grid,           // [n_bands * GP_N_PRED]
    double* __restrict__ std_grid,            // [n_bands * GP_N_PRED]
    int n_bands,
    int n_hp_amp,
    int n_hp_ls,
    int max_subsample)
{
    int band = blockIdx.x;
    int tid  = threadIdx.x;
    int n_hp_total = n_hp_amp * n_hp_ls;

    if (band >= n_bands) return;

    // Shared memory
    extern __shared__ char smem[];
    double* sh_sub_t    = (double*)smem;
    double* sh_sub_v    = sh_sub_t  + GP_MAX_M;
    double* sh_sub_nv   = sh_sub_v  + GP_MAX_M;
    double* sh_scores   = sh_sub_nv + GP_MAX_M;
    double* sh_y_mean   = sh_scores + GP_MAX_HP;
    double* sh_best_amp = sh_y_mean + 1;
    double* sh_best_inv = sh_best_amp + 1;
    double* sh_alpha    = sh_best_inv + 1;
    double* sh_L        = sh_alpha + GP_MAX_M;
    int*    sh_m_val    = (int*)(sh_L + GP_MAX_M * GP_MAX_M);
    int*    sh_winner   = sh_m_val + 1;

    int obs_start = band_offsets[band];
    int obs_end   = band_offsets[band + 1];
    int n_obs     = obs_end - obs_start;

    double* out_state = gp_state + (long long)band * GP_STATE_SIZE;
    double* out_pred  = pred_grid + (long long)band * GP_N_PRED;
    double* out_std   = std_grid  + (long long)band * GP_N_PRED;

    // Thread 0 initializes outputs and loads subsampled data
    if (tid == 0) {
        for (int i = 0; i < GP_STATE_SIZE; i++) out_state[i] = 0.0;
        for (int i = 0; i < GP_N_PRED; i++) {
            out_pred[i] = nan("");
            out_std[i]  = nan("");
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
                double step = (double)(n_obs - 1) / (double)(m - 1);
                for (int i = 0; i < m; i++) {
                    int idx = (int)(i * step + 0.5);
                    if (idx >= n_obs) idx = n_obs - 1;
                    sh_sub_t[i]  = all_times[obs_start + idx];
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

    // ---- Each thread evaluates one hyperparameter combo ----
    double my_score = 1e99;
    double my_amp = 0.0;
    double my_inv2ls2 = 0.0;

    if (tid < n_hp_total) {
        int ia = tid / n_hp_ls;
        int il = tid % n_hp_ls;
        double amp = hp_amps[ia];
        double ls  = hp_ls[il];

        if (ls >= 0.1) {
            double inv_2ls2 = 0.5 / (ls * ls);
            my_amp = amp;
            my_inv2ls2 = inv_2ls2;

            // Local scratch for this thread's Cholesky
            double K[GP_MAX_M * GP_MAX_M];
            double alpha[GP_MAX_M];
            double tmp[GP_MAX_M];

            my_score = gp_try_hyperparams(
                sh_sub_t, sh_sub_v, sh_sub_nv, m, y_mean,
                amp, inv_2ls2, K, alpha, tmp);
        }
    }

    sh_scores[tid] = my_score;
    __syncthreads();

    // ---- Thread 0 finds the winner ----
    if (tid == 0) {
        double best = 1e99;
        int best_idx = 0;
        for (int i = 0; i < n_hp_total && i < GP_MAX_HP; i++) {
            if (sh_scores[i] < best) {
                best = sh_scores[i];
                best_idx = i;
            }
        }
        sh_winner[0] = best_idx;
    }
    __syncthreads();

    // Winner thread writes its hyperparameters to shared memory
    if (tid == sh_winner[0]) {
        sh_best_amp[0] = my_amp;
        sh_best_inv[0] = my_inv2ls2;
    }
    __syncthreads();

    double amp      = sh_best_amp[0];
    double inv_2ls2 = sh_best_inv[0];

    if (amp == 0.0 && inv_2ls2 == 0.0) return;  // all combos failed

    // ---- Thread 0 refits with best hyperparameters and stores state ----
    if (tid == 0) {
        // Rebuild K + Cholesky in shared L
        for (int i = 0; i < m; i++) {
            for (int j = 0; j <= i; j++) {
                double v = gp_rbf(sh_sub_t[i], sh_sub_t[j], amp, inv_2ls2);
                sh_L[i * m + j] = v;
                sh_L[j * m + i] = v;
            }
            double nv = sh_sub_nv[i];
            if (nv < 1e-10) nv = 1e-10;
            sh_L[i * m + i] += nv;
        }

        if (!gp_cholesky(sh_L, m)) {
            sh_m_val[0] = 0;  // signal failure
        } else {
            double y_c[GP_MAX_M];
            double tmp[GP_MAX_M];
            for (int i = 0; i < m; i++) y_c[i] = sh_sub_v[i] - y_mean;
            gp_solve_l(sh_L, y_c, tmp, m);
            gp_solve_lt(sh_L, tmp, sh_alpha, m);

            // Store GP state
            for (int i = 0; i < m; i++) out_state[i] = sh_alpha[i];
            for (int i = m; i < GP_MAX_M; i++) out_state[i] = 0.0;
            for (int i = 0; i < m; i++) out_state[GP_MAX_M + i] = sh_sub_t[i];
            for (int i = m; i < GP_MAX_M; i++) out_state[GP_MAX_M + i] = 0.0;
            out_state[50] = amp;
            out_state[51] = inv_2ls2;
            out_state[52] = y_mean;
            out_state[53] = (double)m;
        }
    }
    __syncthreads();

    m = sh_m_val[0];
    if (m < 3) return;

    // ---- All threads cooperate on GP_N_PRED grid predictions ----
    for (int q = tid; q < GP_N_PRED; q += blockDim.x) {
        double t = query_times[q];

        double k_star[GP_MAX_M];
        double dot = 0.0;
        for (int i = 0; i < m; i++) {
            k_star[i] = gp_rbf(t, sh_sub_t[i], amp, inv_2ls2);
            dot += k_star[i] * sh_alpha[i];
        }
        out_pred[q] = dot + y_mean;

        // Variance: k** - v^T v where L v = k_star
        double v[GP_MAX_M];
        gp_solve_l(sh_L, k_star, v, m);
        double vtv = 0.0;
        for (int i = 0; i < m; i++) vtv += v[i] * v[i];
        double var = amp - vtv;
        if (var < 1e-10) var = 1e-10;
        out_std[q] = sqrt(var);
    }
}

// =========================================================================
// Kernel 2: Predict at observation points using fitted GP state
// =========================================================================
//
// One thread per observation. Mean prediction only (no std needed for chi2).

extern "C" __global__ void batch_gp_predict_obs(
    const double* __restrict__ gp_state,
    const double* __restrict__ all_times,
    const int*    __restrict__ obs_to_band,
    double* __restrict__ pred_obs,
    int total_obs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_obs) return;

    int band = obs_to_band[idx];
    const double* state = gp_state + (long long)band * GP_STATE_SIZE;

    int m = (int)state[53];
    if (m <= 0 || m > GP_MAX_M) { pred_obs[idx] = nan(""); return; }

    double amp      = state[50];
    double inv_2ls2 = state[51];
    double y_mean   = state[52];

    double t   = all_times[idx];
    double dot = 0.0;
    for (int i = 0; i < m; i++) {
        double x_i = state[GP_MAX_M + i];
        dot += gp_rbf(t, x_i, amp, inv_2ls2) * state[i];
    }
    pred_obs[idx] = dot + y_mean;
}

// =========================================================================
// Host-side launch wrappers
// =========================================================================

extern "C" void launch_batch_gp_fit_predict(
    const double* all_times,
    const double* all_mags,
    const double* all_noise_var,
    const int*    band_offsets,
    const double* query_times,
    const double* hp_amps,
    const double* hp_ls,
    double* gp_state,
    double* pred_grid,
    double* std_grid,
    int n_bands,
    int n_hp_amp,
    int n_hp_ls,
    int max_subsample,
    int /*grid*/, int block)
{
    // block = n_hp_total (n_hp_amp * n_hp_ls), grid = n_bands
    // Shared memory: 3*GP_MAX_M + GP_MAX_HP + 3 + GP_MAX_M + GP_MAX_M*GP_MAX_M doubles + 2 ints
    size_t smem_bytes = (3 * GP_MAX_M + GP_MAX_HP + 3 + GP_MAX_M + GP_MAX_M * GP_MAX_M) * sizeof(double) + 2 * sizeof(int);
    batch_gp_fit_predict<<<n_bands, block, smem_bytes>>>(
        all_times, all_mags, all_noise_var, band_offsets,
        query_times, hp_amps, hp_ls,
        gp_state, pred_grid, std_grid,
        n_bands, n_hp_amp, n_hp_ls, max_subsample);
}

extern "C" void launch_batch_gp_predict_obs(
    const double* gp_state,
    const double* all_times,
    const int*    obs_to_band,
    double* pred_obs,
    int total_obs,
    int grid, int block)
{
    batch_gp_predict_obs<<<grid, block>>>(
        gp_state, all_times, obs_to_band, pred_obs, total_obs);
}
