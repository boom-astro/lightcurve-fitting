# Benchmarks

## Methodology

All benchmarks use synthetic Bazin-model light curves with realistic noise
in three bands (g, r, i), matching the typical ZTF photometry format.
Each run fits **100 light curves** simultaneously.

- **CPU**: Rust sequential processing (single-threaded)
- **GPU**: CUDA batch processing across all sources simultaneously
- **Hardware**: NVIDIA Tesla P100 (12 GB), Skylake Xeon
- **Timing**: best of 3 runs after 1 warmup iteration
- **Metric**: throughput in total observations processed per second
  (`n_sources * n_points_per_band * 3_bands / wall_sec`)

### Fitters benchmarked

| Fitter | Description |
|--------|-------------|
| **parametric** | PSO model selection (8 models, 30 particles × 60 iters × 2-3 adaptive restarts) + Laplace uncertainty |
| **parametric + MultiBazin** | As above, plus greedy MultiBazin fit (K=1..4, 3 restarts each) on the first band |
| **nonparametric** | Gaussian-process interpolation + feature extraction |

### GP strategy

The nonparametric fitter uses a hybrid GP approach for scalability:

- **n <= 100 observations**: Custom DenseGP on all data. O(n^3).
- **n > 100 observations**: Sparse FITC approximation with 30 inducing points. O(n*m^2) — linear in observation count.

Both implementations use hand-rolled inline RBF kernels and row-major Cholesky.

## Point-Count Scaling (100 sources)

![Point-count scaling](throughput_points.png)

### Throughput table (observations/sec)

| pts/band | NP CPU | NP GPU | Param CPU | Param CPU-par (32t) | Param GPU | MultiBazin GPU |
|---------:|-------:|-------:|----------:|--------------------:|----------:|---------------:|
| 64 | 55K | 2.0M | 14K | 347K | 5.4M | TBD |
| 128 | 75K | 3.3M | 14K | 396K | 8.0M | TBD |
| 256 | 88K | 4.8M | 15K | 400K | 10.2M | TBD |
| 512 | 106K | 6.7M | 14K | 387K | 12.0M | TBD |
| 1,024 | 122K | 7.2M | 16K | 422K | 12.2M | TBD |
| 2,048 | 141K | 8.1M | 17K | 427K | 12.6M | TBD |
| 4,096 | 155K | 7.9M | 17K | 428K | 12.9M | TBD |
| 8,192 | 279K | 7.8M | 17K | 431K | 13.1M | TBD |

### Source-count scaling

Source-count scaling is approximately flat for both fitters: parametric CPU
runs at ~11K obs/sec and nonparametric CPU at ~50-69K obs/sec regardless
of source count (10-1000 sources at 30 pts/band).

### Discussion

**GPU acceleration**: Both fitters achieve 8-13M obs/sec on GPU at high
point counts — up to **750x faster than parametric CPU** and **28x faster
than nonparametric CPU**. The GPU implementations batch all sources into a
single kernel launch, amortizing launch overhead.

**Parametric GPU vs nonparametric GPU**: At low point counts (64 pts/band),
parametric GPU is ~2.7x faster than nonparametric GPU because the PSO
particle evaluations (30 particles x 50 iterations x 8 models = 12,000
work items per source) provide massive thread-level parallelism.
The nonparametric fit kernel launches one block per band (300 blocks for
100 sources x 3 bands), with threads parallelizing across hyperparameter
combos. At high point counts (4096+), parametric GPU maintains its lead
as both fitters plateau near GPU memory bandwidth limits.

**CPU comparison**: Nonparametric CPU is 4-16x faster than parametric CPU.
Nonparametric throughput scales well with point count (sparse GP is linear
in n), while parametric CPU plateaus at ~17K obs/sec as PSO iteration
cost dominates.

**CPU-par (rayon)**: The parametric CPU-par backend uses Rayon to
parallelize source-level fitting across 32 threads. It achieves 25-31x
speedup over single-threaded parametric CPU, peaking at ~430K obs/sec at
high point counts. Scaling is near-linear with core count, making it a
practical choice when GPU hardware is unavailable.

**MultiBazin GPU**: The MultiBazin greedy fitter (K=1..4 Bazin components,
3 PSO restarts per K) runs as a standalone GPU batch kernel. It iterates
K sequentially (each K runs all sources in parallel), with CPU-side
residual-based seeding between K steps. Expected throughput is lower than
the 8-model parametric GPU due to the sequential K loop (4 rounds of PSO
vs 8 independent models), but benefits from warm-start seeding and
per-source early stopping.

**LSST DDF implications**: At 8,192 pts/band (representative of a
well-sampled DDF source), GPU processing reaches 13M obs/sec — fitting
100 sources with ~2.5M total observations in under 0.2 seconds.

## Reproducing

```bash
# Run all benchmarks (requires CUDA for GPU columns)
cargo test --release --features cuda --test bench_throughput -- --ignored --nocapture

# Individual suites
cargo test --release --features cuda --test bench_throughput nonparametric_point_scaling -- --ignored --nocapture
cargo test --release --features cuda --test bench_throughput parametric_point_scaling -- --ignored --nocapture

# CPU-only
cargo test --release --test bench_throughput -- --ignored --nocapture

# Generate plot from CSVs
python3 benchmarks/plot_throughput.py
```

Results are written to `benchmarks/nonparametric_points.csv` and
`benchmarks/parametric_points.csv`.
