//! Lightweight Gaussian Process for 1D time series.
//!
//! Two modes:
//! - **Dense**: O(n³) Cholesky on all data. Best for n ≤ ~200.
//! - **Sparse (FITC)**: O(n·m²) with m inducing points. Best for n > 200.
//!
//! Optimized for lightcurve fitting:
//! - 1D input (time), RBF kernel hardcoded
//! - Compact row-major storage, no heap allocation during predict
//! - Avoids ndarray/BLAS overhead for small matrices

// ---------------------------------------------------------------------------
// RBF kernel (1D, inlined)
// ---------------------------------------------------------------------------

#[inline(always)]
fn rbf(x1: f64, x2: f64, amp: f64, inv_2ls2: f64) -> f64 {
    let d = x1 - x2;
    amp * (-d * d * inv_2ls2).exp()
}

// ---------------------------------------------------------------------------
// Row-major Cholesky & triangular solves
// ---------------------------------------------------------------------------

/// In-place Cholesky of symmetric positive-definite n×n matrix (row-major).
fn cholesky(a: &mut [f64], n: usize) -> bool {
    for j in 0..n {
        let mut s = a[j * n + j];
        for k in 0..j {
            s -= a[j * n + k] * a[j * n + k];
        }
        if s <= 0.0 { return false; }
        a[j * n + j] = s.sqrt();
        let ljj = a[j * n + j];
        for i in (j + 1)..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = s / ljj;
        }
        // zero upper triangle
        for i in 0..j { a[i * n + j] = 0.0; }
    }
    true
}

/// Solve L x = b (lower triangular, row-major n×n).
fn solve_l(l: &[f64], b: &[f64], x: &mut [f64], n: usize) {
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i { s -= l[i * n + j] * x[j]; }
        x[i] = s / l[i * n + i];
    }
}

/// Solve L^T x = b (L is lower triangular, row-major n×n).
fn solve_lt(l: &[f64], b: &[f64], x: &mut [f64], n: usize) {
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n { s -= l[j * n + i] * x[j]; }
        x[i] = s / l[i * n + i];
    }
}

// =========================================================================
// DenseGP — full Cholesky on training data
// =========================================================================

/// Dense GP for small datasets (n ≤ ~200).
#[derive(Clone)]
pub struct DenseGP {
    x: Vec<f64>,
    alpha: Vec<f64>,  // (K + σ²I)^{-1} (y - y_mean)
    l: Vec<f64>,      // Cholesky factor (n×n, row-major)
    n: usize,
    amp: f64,
    inv_2ls2: f64,
    y_mean: f64,
}

impl DenseGP {
    pub fn fit(
        times: &[f64], values: &[f64], noise_var: &[f64],
        amp: f64, lengthscale: f64,
    ) -> Option<Self> {
        let n = times.len();
        if n == 0 { return None; }
        let inv_2ls2 = 0.5 / (lengthscale * lengthscale);
        let y_mean = values.iter().sum::<f64>() / n as f64;

        // Build K + noise*I
        let mut k = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..=i {
                let v = rbf(times[i], times[j], amp, inv_2ls2);
                k[i * n + j] = v;
                k[j * n + i] = v;
            }
            let nv = if noise_var.len() == 1 { noise_var[0] } else { noise_var[i] };
            k[i * n + i] += nv.max(1e-10);
        }

        // Cholesky
        if !cholesky(&mut k, n) { return None; }
        let l = k;

        // alpha = L^{-T} L^{-1} (y - y_mean)
        let y_centered: Vec<f64> = values.iter().map(|v| v - y_mean).collect();
        let mut tmp = vec![0.0; n];
        let mut alpha = vec![0.0; n];
        solve_l(&l, &y_centered, &mut tmp, n);
        solve_lt(&l, &tmp, &mut alpha, n);

        Some(DenseGP { x: times.to_vec(), alpha, l, n, amp, inv_2ls2, y_mean })
    }

    /// Return the kernel amplitude (σ²).
    #[inline]
    pub fn kernel_amp(&self) -> f64 { self.amp }

    /// Return the kernel lengthscale (ρ), in the same units as training times.
    #[inline]
    pub fn kernel_lengthscale(&self) -> f64 { (0.5 / self.inv_2ls2).sqrt() }

    #[inline]
    pub fn predict(&self, query: &[f64]) -> Vec<f64> {
        let n = self.n;
        let mut out = Vec::with_capacity(query.len());
        for &t in query {
            let mut dot = 0.0;
            for i in 0..n {
                dot += rbf(t, self.x[i], self.amp, self.inv_2ls2) * self.alpha[i];
            }
            out.push(dot + self.y_mean);
        }
        out
    }

    pub fn predict_with_std(&self, query: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = self.n;
        let nq = query.len();
        let mut means = Vec::with_capacity(nq);
        let mut stds = Vec::with_capacity(nq);
        let mut k_star = vec![0.0; n];
        let mut v = vec![0.0; n];

        for &t in query {
            // Build k_star
            let mut dot = 0.0;
            for i in 0..n {
                k_star[i] = rbf(t, self.x[i], self.amp, self.inv_2ls2);
                dot += k_star[i] * self.alpha[i];
            }
            means.push(dot + self.y_mean);

            // Variance: k** - v^T v  where L v = k_star
            solve_l(&self.l, &k_star, &mut v, n);
            let vtv: f64 = v[..n].iter().map(|x| x * x).sum();
            let var = (self.amp - vtv).max(1e-10);
            stds.push(var.sqrt());
        }

        (means, stds)
    }

    /// RMS of predictions vs targets at training points (leave-one-out proxy).
    pub fn train_rms(&self, values: &[f64]) -> f64 {
        let pred = self.predict(&self.x);
        let n = pred.len().max(1) as f64;
        let rss: f64 = pred.iter().zip(values.iter())
            .map(|(p, v)| (p - v) * (p - v)).sum();
        (rss / n).sqrt()
    }
}

// =========================================================================
// SparseGP — FITC approximation with m inducing points
// =========================================================================

/// Sparse GP (FITC) for large datasets (n > ~200).
pub struct SparseGP {
    z: Vec<f64>,       // inducing points (m,)
    alpha: Vec<f64>,   // precomputed weights (m,)
    l_mm: Vec<f64>,    // Cholesky of K_mm (m×m)
    l_b: Vec<f64>,     // Cholesky of B (m×m)
    m: usize,
    amp: f64,
    inv_2ls2: f64,
    y_mean: f64,
}

impl SparseGP {
    pub fn fit(
        times: &[f64], values: &[f64], noise_var: &[f64],
        amp: f64, lengthscale: f64, m: usize,
    ) -> Option<Self> {
        let n = times.len();
        if n == 0 || m == 0 { return None; }
        let m = m.min(n);
        let inv_2ls2 = 0.5 / (lengthscale * lengthscale);
        let y_mean = values.iter().sum::<f64>() / n as f64;

        // Inducing points: uniform spacing
        let t_min = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let t_max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let z: Vec<f64> = (0..m)
            .map(|i| t_min + (i as f64) * (t_max - t_min) / (m - 1).max(1) as f64)
            .collect();

        // K_mm
        let mut k_mm = vec![0.0; m * m];
        for i in 0..m {
            for j in 0..=i {
                let v = rbf(z[i], z[j], amp, inv_2ls2);
                k_mm[i * m + j] = v;
                k_mm[j * m + i] = v;
            }
            k_mm[i * m + i] += 1e-6;
        }

        let mut l_mm = k_mm.clone();
        if !cholesky(&mut l_mm, m) { return None; }

        // K_mn columns + V = L_mm^{-1} K_mn + lambda_inv
        // Process one observation at a time to avoid O(n*m) allocation
        let mut b = k_mm; // B starts as K_mm
        let mut w = vec![0.0; m]; // K_mn Λ^{-1} y
        let mut k_col = vec![0.0; m];
        let mut v_col = vec![0.0; m];

        for j in 0..n {
            // k_col = K(:, x_j)
            for i in 0..m { k_col[i] = rbf(z[i], times[j], amp, inv_2ls2); }

            // v_col = L_mm^{-1} k_col
            solve_l(&l_mm, &k_col, &mut v_col, m);

            // Q_jj = ||v_col||^2
            let q_jj: f64 = v_col[..m].iter().map(|x| x * x).sum();
            let nv = if noise_var.len() == 1 { noise_var[0] } else { noise_var[j] };
            let lambda_j = (amp - q_jj).max(0.0) + nv;
            let li = 1.0 / lambda_j.max(1e-12);

            // B += li * k_col * k_col^T (symmetric rank-1 update)
            for r in 0..m {
                let scaled = li * k_col[r];
                for c in r..m {
                    b[r * m + c] += scaled * k_col[c];
                }
            }

            // w += li * y_centered_j * k_col
            let val = li * (values[j] - y_mean);
            for i in 0..m { w[i] += k_col[i] * val; }
        }

        // Mirror upper triangle to lower
        for r in 0..m { for c in 0..r { b[r * m + c] = b[c * m + r]; } }

        let mut l_b = b;
        if !cholesky(&mut l_b, m) { return None; }

        // alpha = L_B^{-T} L_B^{-1} w
        let mut tmp = vec![0.0; m];
        let mut alpha = vec![0.0; m];
        solve_l(&l_b, &w, &mut tmp, m);
        solve_lt(&l_b, &tmp, &mut alpha, m);

        Some(SparseGP { z, alpha, l_mm, l_b, m, amp, inv_2ls2, y_mean })
    }

    pub fn predict(&self, query: &[f64]) -> Vec<f64> {
        let m = self.m;
        let mut out = Vec::with_capacity(query.len());
        for &t in query {
            let mut dot = 0.0;
            for i in 0..m {
                dot += rbf(t, self.z[i], self.amp, self.inv_2ls2) * self.alpha[i];
            }
            out.push(dot + self.y_mean);
        }
        out
    }

    pub fn predict_with_std(&self, query: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let m = self.m;
        let nq = query.len();
        let mut means = Vec::with_capacity(nq);
        let mut stds = Vec::with_capacity(nq);
        let mut k_star = vec![0.0; m];
        let mut v_mm = vec![0.0; m];
        let mut v_b = vec![0.0; m];

        for &t in query {
            let mut dot = 0.0;
            for i in 0..m {
                k_star[i] = rbf(t, self.z[i], self.amp, self.inv_2ls2);
                dot += k_star[i] * self.alpha[i];
            }
            means.push(dot + self.y_mean);

            // var = k** - ||L_mm^{-1} k*||^2 + ||L_B^{-1} k*||^2
            solve_l(&self.l_mm, &k_star, &mut v_mm, m);
            solve_l(&self.l_b, &k_star, &mut v_b, m);
            let qf_mm: f64 = v_mm[..m].iter().map(|x| x * x).sum();
            let qf_b: f64 = v_b[..m].iter().map(|x| x * x).sum();
            let var = (self.amp - qf_mm + qf_b).max(1e-10);
            stds.push(var.sqrt());
        }

        (means, stds)
    }

    /// Approximate negative log marginal likelihood for model selection.
    pub fn approx_nlml(&self, times: &[f64], values: &[f64], noise_var: &[f64]) -> f64 {
        let pred = self.predict(times);
        let n = times.len() as f64;
        let mut nlml = 0.0;
        for i in 0..times.len() {
            let nv = if noise_var.len() == 1 { noise_var[0] } else { noise_var[i] };
            let r = values[i] - pred[i];
            nlml += r * r / nv + nv.ln();
        }
        nlml / n
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_data(n: usize) -> (Vec<f64>, Vec<f64>) {
        let times: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
        let values: Vec<f64> = times.iter().map(|t| (t * 0.5).sin()).collect();
        (times, values)
    }

    #[test]
    fn dense_gp_recovers_signal() {
        let (times, values) = sine_data(50);
        let gp = DenseGP::fit(&times, &values, &[0.01], 1.0, 3.0)
            .expect("fit should succeed");
        let query = vec![0.5, 2.5, 4.0];
        let (pred, std) = gp.predict_with_std(&query);
        for (i, &t) in query.iter().enumerate() {
            let expected = (t * 0.5).sin();
            assert!((pred[i] - expected).abs() < 0.15,
                "t={t}: pred={:.3}, expected={:.3}", pred[i], expected);
            assert!(std[i] > 0.0);
        }
    }

    #[test]
    fn sparse_gp_recovers_signal() {
        let (times, values) = sine_data(200);
        let gp = SparseGP::fit(&times, &values, &[0.01], 1.0, 3.0, 30)
            .expect("fit should succeed");
        let query = vec![0.5, 5.0, 10.0];
        let (pred, std) = gp.predict_with_std(&query);
        for (i, &t) in query.iter().enumerate() {
            let expected = (t * 0.5).sin();
            assert!((pred[i] - expected).abs() < 0.2,
                "t={t}: pred={:.3}, expected={:.3}", pred[i], expected);
            assert!(std[i] > 0.0);
        }
    }

    #[test]
    fn dense_matches_sparse_on_small_data() {
        let (times, values) = sine_data(30);
        let dense = DenseGP::fit(&times, &values, &[0.01], 1.0, 3.0).unwrap();
        let sparse = SparseGP::fit(&times, &values, &[0.01], 1.0, 3.0, 30).unwrap();

        let query = vec![0.5, 1.5, 2.5];
        let d_pred = dense.predict(&query);
        let s_pred = sparse.predict(&query);
        for i in 0..query.len() {
            assert!((d_pred[i] - s_pred[i]).abs() < 0.1,
                "q={}: dense={:.4}, sparse={:.4}", query[i], d_pred[i], s_pred[i]);
        }
    }

    #[test]
    fn dense_handles_3_points() {
        let gp = DenseGP::fit(&[1.0, 2.0, 3.0], &[1.0, 2.0, 1.5], &[0.1], 1.0, 1.0)
            .expect("small n should work");
        let pred = gp.predict(&[2.0]);
        assert!((pred[0] - 2.0).abs() < 0.5);
    }
}
