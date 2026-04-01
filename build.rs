#[cfg(feature = "cuda")]
fn build_cuda() {
    // Compile CUDA kernels via cc crate with CUDA support
    cc::Build::new()
        .cuda(true)
        .flag("-gencode=arch=compute_60,code=sm_60")  // Pascal (P100)
        .flag("-gencode=arch=compute_70,code=sm_70")  // Volta (V100)
        .flag("-gencode=arch=compute_80,code=sm_80")  // Ampere (A100)
        .flag("-gencode=arch=compute_89,code=sm_89")  // Ada (L40, RTX 4090)
        .flag("-gencode=arch=compute_90,code=sm_90")  // Hopper (H100)
        .flag("-Wno-deprecated-gpu-targets")
        .file("cuda/models.cu")
        .file("cuda/gp.cu")
        .file("cuda/gp2d.cu")
        .file("cuda/svi.cu")
        .compile("lightcurve_cuda");

    // Link against the CUDA runtime
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rerun-if-changed=cuda/models_device.h");
    println!("cargo:rerun-if-changed=cuda/models.cu");
    println!("cargo:rerun-if-changed=cuda/gp.cu");
    println!("cargo:rerun-if-changed=cuda/gp2d.cu");
    println!("cargo:rerun-if-changed=cuda/svi.cu");
}

#[cfg(feature = "metal")]
fn build_metal() {
    use std::process::Command;

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let metal_files = [
        "metal/models.metal",
        "metal/gp.metal",
        "metal/gp2d.metal",
        "metal/svi.metal",
    ];

    let mut air_paths = Vec::new();

    for src in &metal_files {
        let stem = std::path::Path::new(src)
            .file_stem()
            .unwrap()
            .to_str()
            .unwrap();
        let air_path = format!("{}/{}.air", out_dir, stem);

        let status = Command::new("xcrun")
            .args([
                "-sdk", "macosx", "metal",
                "-c", src,
                "-o", &air_path,
                "-std=metal3.0",
                "-I", "metal/", // for #include "models_device.h"
            ])
            .status()
            .unwrap_or_else(|e| panic!("Failed to run xcrun metal for {}: {}", src, e));

        assert!(
            status.success(),
            "Metal shader compilation failed for {}",
            src
        );
        air_paths.push(air_path);
        println!("cargo:rerun-if-changed={}", src);
    }

    // Link .air files into a single .metallib
    let metallib_path = format!("{}/lightcurve.metallib", out_dir);
    let mut cmd = Command::new("xcrun");
    cmd.args(["-sdk", "macosx", "metallib"]);
    for air in &air_paths {
        cmd.arg(air);
    }
    cmd.args(["-o", &metallib_path]);

    let status = cmd
        .status()
        .expect("Failed to run xcrun metallib");

    assert!(status.success(), "metallib linking failed");

    println!("cargo:rerun-if-changed=metal/models_device.h");
}

fn main() {
    #[cfg(feature = "cuda")]
    build_cuda();

    #[cfg(feature = "metal")]
    build_metal();
}
