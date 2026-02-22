use scirs2_core::ndarray::{Array1, Axis};
use sklears_core::traits::Fit;
use sklears_gaussian_process::{
    kernels::{ConstantKernel, ProductKernel, SumKernel, WhiteKernel, RBF},
    GaussianProcessRegressor, GprTrained, Kernel,
};

/// Fit a GP with the given amplitude, lengthscale, and alpha parameters.
pub fn fit_sklears_gp(
    times: &Array1<f64>,
    values: &Array1<f64>,
    amp: f64,
    lengthscale: f64,
    alpha: f64,
) -> Option<GaussianProcessRegressor<GprTrained>> {
    let cst: Box<dyn Kernel> = Box::new(ConstantKernel::new(amp));
    let rbf: Box<dyn Kernel> = Box::new(RBF::new(lengthscale));
    let prod = Box::new(ProductKernel::new(vec![cst, rbf]));
    let white = Box::new(WhiteKernel::new(1e-10));
    let kernel = SumKernel::new(vec![prod, white]);

    let gp = GaussianProcessRegressor::new()
        .kernel(Box::new(kernel))
        .alpha(alpha)
        .normalize_y(true);

    let xt = times.view().insert_axis(Axis(1)).to_owned();
    gp.fit(&xt, values).ok()
}

/// Subsample data to at most `max_points` by uniform striding.
pub fn subsample_data(
    times: &[f64],
    mags: &[f64],
    errors: &[f64],
    max_points: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    if times.len() <= max_points {
        return (times.to_vec(), mags.to_vec(), errors.to_vec());
    }

    let step = times.len() as f64 / max_points as f64;
    let mut indices = Vec::with_capacity(max_points);
    for i in 0..max_points {
        let idx = ((i as f64 + 0.5) * step).floor() as usize;
        indices.push(idx.min(times.len() - 1));
    }

    let times_sub: Vec<f64> = indices.iter().map(|&i| times[i]).collect();
    let mags_sub: Vec<f64> = indices.iter().map(|&i| mags[i]).collect();
    let errors_sub: Vec<f64> = indices.iter().map(|&i| errors[i]).collect();

    (times_sub, mags_sub, errors_sub)
}
