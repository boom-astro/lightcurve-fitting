#!/bin/bash
#SBATCH --partition=milan-gpu
#SBATCH --job-name=ztf-extract
#SBATCH --account=oz480
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/fred/oz480/mcoughli/data_ztf/extract_%j.log

module load cuda/12.8.0

cd /home/mcoughli/lightcurve-fitting

# Build with CUDA support
cargo clean -p lightcurve-fitting --release
CARGO_TARGET_DIR=/fred/oz480/mcoughli/envs/cargo-target \
    cargo build --release --features "cuda cli"

INPUT_DIR=/fred/oz480/mcoughli/data_ztf
OUTPUT_DIR=/fred/oz480/mcoughli/data_ztf/features
BINARY=/fred/oz480/mcoughli/envs/cargo-target/release/extract-features

mkdir -p "$OUTPUT_DIR"

"$BINARY" \
    --input-dir "$INPUT_DIR" \
    --format ztf \
    --output-dir "$OUTPUT_DIR" \
    --batch-size 512 \
    --gpu-device 0 \
    --skip-existing
