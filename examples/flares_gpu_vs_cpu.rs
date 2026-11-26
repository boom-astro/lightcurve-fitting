//! Compare the GPU flare search with the CPU one on the Davenport et al. (2014)
//! GJ 1243 Kepler catalogue (data from github.com/jradavenport/GJ1243-Flares).
//!
//!   cargo run --release --features metal --example flares_gpu_vs_cpu -- DATA_DIR
//!
//! DATA_DIR holds gj1243_master_slc.dat and gj1243_master_flares.tbl, e.g. as
//! downloaded by scripts/eval_flares_kepler.py.

#[cfg(not(any(feature = "cuda", feature = "metal")))]
fn main() {
    eprintln!("build with --features metal or --features cuda");
}

#[cfg(any(feature = "cuda", feature = "metal"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;

    use lightcurve_fitting::gpu::GpuContext;
    use lightcurve_fitting::{fit_flares_batch, fit_flares_batch_gpu, BandData, FlareFitOptions};

    const MIN: f64 = 1.0 / 1440.0;
    const CADENCE: f64 = 58.8487 / 86400.0;

    let dir = std::path::PathBuf::from(std::env::args().nth(1).ok_or("usage: flares_gpu_vs_cpu DATA_DIR")?);
    let rows = |name: &str| -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
        let text = std::fs::read_to_string(dir.join(name))?;
        Ok(text
            .lines()
            .filter(|l| !l.trim_start().starts_with('#') && !l.trim().is_empty())
            .map(|l| l.split_whitespace().map(|v| v.parse().unwrap_or(f64::NAN)).collect())
            .collect())
    };
    let lc = rows("gj1243_master_slc.dat")?;
    let cat = rows("gj1243_master_flares.tbl")?;
    let t: Vec<f64> = lc.iter().map(|r| r[0]).collect();

    // Same windows as scripts/eval_flares_kepler.py: padded catalogue start/stop,
    // detrended flux normalised to its median in the window.
    let windows: Vec<BandData> = cat
        .iter()
        .map(|r| {
            let (ts, te) = (r[2], r[3]);
            let d = te - ts;
            let (lo, hi) = (ts - (0.5 * d).max(10.0 * MIN), te + d.max(15.0 * MIN));
            let a = t.partition_point(|&x| x < lo);
            let b = t.partition_point(|&x| x <= hi);
            let mut f: Vec<f64> = lc[a..b].iter().map(|r| r[3]).collect();
            let mut sorted = f.clone();
            sorted.sort_by(|x, y| x.total_cmp(y));
            let med = sorted[sorted.len() / 2];
            let errors = lc[a..b].iter().map(|r| r[2] / med).collect();
            f.iter_mut().for_each(|v| *v = *v / med - 1.0);
            BandData { times: t[a..b].to_vec(), values: f, errors }
        })
        .collect();
    let opts = FlareFitOptions { exposure_time: Some(CADENCE), max_flares: 5, ..Default::default() };
    println!("{} windows", windows.len());

    let t0 = Instant::now();
    let cpu = fit_flares_batch(&windows, &opts);
    let t_cpu = t0.elapsed().as_secs_f64();
    let t0 = Instant::now();
    let gpu_ctx = GpuContext::new(0)?;
    let t_ctx = t0.elapsed().as_secs_f64();
    let t0 = Instant::now();
    let gpu = fit_flares_batch_gpu(&gpu_ctx, &windows, &opts)?;
    let t_gpu = t0.elapsed().as_secs_f64();
    println!("CPU (rayon, all cores): {t_cpu:.2} s   GPU: {t_gpu:.2} s (+{t_ctx:.2} s context setup)");

    let pct = |v: &mut Vec<f64>, q: f64| -> f64 {
        if v.is_empty() {
            return f64::NAN;
        }
        v.sort_by(|a, b| a.total_cmp(b));
        v[((v.len() - 1) as f64 * q).round() as usize]
    };

    let mut same_k = 0;
    let mut more = 0;
    let mut fewer = 0;
    let mut dchi2 = Vec::new();
    let mut d_tp = Vec::new();
    let mut d_th = Vec::new();
    let mut d_ed = Vec::new();
    let mut ed_ratio = Vec::new();
    let mut mismatch_dbic = Vec::new();
    for (c, g) in cpu.iter().zip(&gpu) {
        let (Some(c), Some(g)) = (c, g) else { continue };
        let (kc, kg) = (c.flares.len(), g.flares.len());
        dchi2.push(g.chi2 - c.chi2);
        let ed = |r: &lightcurve_fitting::FlareFitResult| r.flares.iter().map(|f| f.equivalent_duration).sum::<f64>();
        if ed(c) > 0.0 && ed(g) > 0.0 {
            ed_ratio.push(ed(g) / ed(c));
        }
        if kc != kg {
            if kg > kc { more += 1 } else { fewer += 1 }
            mismatch_dbic.push(g.bic - c.bic);
            continue;
        }
        same_k += 1;
        for (cf, gf) in c.flares.iter().zip(&g.flares) {
            if let (Some(s1), Some(s2), Some(s3)) = (cf.t_peak_err, cf.t_half_err, cf.equivalent_duration_err) {
                d_tp.push((gf.t_peak - cf.t_peak).abs() / s1);
                d_th.push((gf.t_half - cf.t_half).abs() / s2);
                d_ed.push((gf.equivalent_duration - cf.equivalent_duration).abs() / s3);
            }
        }
    }
    let n = dchi2.len();
    println!("\nwindows fitted by both: {n}");
    println!(
        "same number of components: {:.4} (GPU more: {more}, GPU fewer: {fewer})",
        same_k as f64 / n as f64
    );
    println!(
        "  in mismatches, BIC(GPU) - BIC(CPU): median {:.2}, 5-95%: {:.2} .. {:.2}",
        pct(&mut mismatch_dbic, 0.5),
        pct(&mut mismatch_dbic, 0.05),
        pct(&mut mismatch_dbic, 0.95)
    );
    let frac = |v: &[f64], f: &dyn Fn(f64) -> bool| v.iter().filter(|&&x| f(x)).count() as f64 / v.len() as f64;
    println!(
        "chi2(GPU) - chi2(CPU): median {:.4}; GPU worse by >0.5: {:.4}, >2: {:.4}; GPU better by >0.5: {:.4}",
        pct(&mut dchi2.clone(), 0.5),
        frac(&dchi2, &|x| x > 0.5),
        frac(&dchi2, &|x| x > 2.0),
        frac(&dchi2, &|x| x < -0.5)
    );
    println!("matched components: {}", d_tp.len());
    for (name, v) in [("t_peak", &mut d_tp), ("t_half", &mut d_th), ("ED", &mut d_ed)] {
        let below = frac(v, &|x| x < 0.25);
        println!(
            "  |{name} GPU - CPU| / sigma: median {:.4}, 95% {:.3}, 99% {:.3}; < 0.25 sigma: {:.4}",
            pct(v, 0.5),
            pct(v, 0.95),
            pct(v, 0.99),
            below
        );
    }
    println!(
        "total ED per window, GPU / CPU: median {:.4}, 5-95%: {:.4} .. {:.4}",
        pct(&mut ed_ratio, 0.5),
        pct(&mut ed_ratio, 0.05),
        pct(&mut ed_ratio, 0.95)
    );
    Ok(())
}
