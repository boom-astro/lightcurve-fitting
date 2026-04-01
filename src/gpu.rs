//! GPU-accelerated batch model evaluation and fitting.
//!
//! This module provides a unified interface for GPU backends (CUDA, Metal).
//! Shared types are defined in `gpu_types` and re-exported here.
//! The active backend is selected at compile time via feature flags.

// Note: gpu_types, gpu_cuda, and gpu_metal are declared as sibling modules
// in lib.rs and re-exported through this module for backwards compatibility.

// Re-export everything from sibling modules through here so callers
// can continue to use `gpu::GpuModelName`, `gpu::GpuContext`, etc.
pub use super::gpu_types::*;

#[cfg(feature = "cuda")]
pub use super::gpu_cuda::*;

#[cfg(feature = "metal")]
pub use super::gpu_metal::*;
