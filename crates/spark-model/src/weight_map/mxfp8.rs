// SPDX-License-Identifier: AGPL-3.0-only

//! MXFP8_E4M3 microscaling dequantization for PrismaQuant / compressed-tensors.
//!
//! MXFP8 stores FP8 E4M3 weights with a shared uint8 E8M0 exponent per block of
//! 32 elements (group_size=32). The dequant is:
//!
//!   bf16[i,j] = fp8_e4m3_to_f32(weight[i,j]) * 2^(weight_scale[i, j/32] - 127)
//!
//! This module provides CPU-side dequant at load time. The result is a BF16
//! DenseWeight on GPU, which the caller may keep or further quantize to NVFP4.

#![allow(unused_imports)]

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::*;

/// FP8 E4M3 lookup table: 256 entries mapping byte value → f32.
///
/// Reproduces the IEEE-754-like FP8 E4M3 format:
///   - 1 sign bit, 4 exponent bits, 3 mantissa bits
///   - exponent bias = 7
///   - no infinite / NaN (all exponent=1111 values are NaN)
fn fp8_e4m3_lut() -> &'static [f32; 256] {
    use std::sync::LazyLock;
    static LUT: LazyLock<[f32; 256]> = LazyLock::new(|| {
        let mut lut = [0.0f32; 256];
        let mut i = 0u16;
        while i < 256 {
            let sign = ((i >> 7) & 1) as f32;
            let exp = ((i >> 3) & 0xF) as i32;
            let mant = (i & 0x7) as f32;
            let val = if exp == 0 {
                let s: f32 = if sign == 0.0 { 1.0 } else { -1.0 };
                s * mant / 8.0 * 2.0f32.powi(-6)
            } else if exp == 15 {
                f32::NAN
            } else {
                let s: f32 = if sign == 0.0 { 1.0 } else { -1.0 };
                s * (1.0 + mant / 8.0) * 2.0f32.powi(exp - 7)
            };
            lut[i as usize] = val;
            i += 1;
        }
        lut
    });
    &LUT
}

/// Dequantize a single MXFP8 element: FP8 E4M3 value × E8M0 block scale.
fn mxfp8_dequant_element(fp8_byte: u8, e8m0_byte: u8) -> f32 {
    let fp8_val = fp8_e4m3_lut()[fp8_byte as usize];
    if fp8_val.is_nan() || fp8_val == 0.0 {
        return 0.0;
    }
    // E8M0: unsigned 8-bit exponent. Scale = 2^(e8m0 - 127)
    let scale = 2.0f32.powi(e8m0_byte as i32 - 127);
    fp8_val * scale
}

/// Dequantize MXFP8_E4M3 block-scaled weight → BF16 on GPU.
///
/// Reads:
///   - `{prefix}.weight`: FP8E4M3 tensor of shape `[N, K]`
///   - `{prefix}.weight_scale`: uint8 E8M0 tensor of shape `[N, K/32]`
///
/// Returns a BF16 DenseWeight on GPU.
pub(crate) fn dequant_mxfp8_to_bf16(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let w = store.get(&format!("{prefix}.weight"))?;
    ensure!(
        w.dtype == WeightDtype::FP8E4M3,
        "Expected FP8E4M3 for {prefix}.weight, got {:?}",
        w.dtype,
    );
    ensure!(
        w.shape.len() == 2,
        "Expected 2D weight for {prefix}, got {:?}",
        w.shape
    );
    let n = w.shape[0];
    let k = w.shape[1];
    let group_size = 32usize;
    ensure!(
        k % group_size == 0,
        "MXFP8 requires K ({k}) divisible by group_size ({group_size})"
    );
    let num_groups = k / group_size;

    let fp8_size = n * k;
    tracing::debug!("MXFP8 dequant: {prefix} shape=[{n},{k}] groups={num_groups}");

    // Download FP8 weight bytes
    let mut fp8_buf = vec![0u8; fp8_size];
    gpu.copy_d2h(w.ptr, &mut fp8_buf).with_context(|| {
        let free = gpu.free_memory().unwrap_or(0);
        format!(
            "MXFP8 D2H failed for {prefix}.weight: ptr={}, size={fp8_size}, free={:.1} GB",
            w.ptr.0,
            free as f64 / (1024.0 * 1024.0 * 1024.0),
        )
    })?;

    // Download E8M0 scale bytes (one per group)
    let s = store.get(&format!("{prefix}.weight_scale"))?;
    // Per-channel FP32 scale ([N, 1] or 1D [N]) — PrismaQuant float-quantized.
    // MXFP8 expects uint8 E8M0 block scales of shape [N, K/32].
    // If the scale doesn't match MXFP8 shape, redirect to per-channel dequant.
    if s.shape.len() == 1 || (s.shape.len() == 2 && s.shape[1] <= 1 && s.shape[0] == n) {
        tracing::debug!(
            "{prefix}: scale shape {:?} → redirecting to per-channel FP8 dequant",
            s.shape
        );
        return dequant_fp8_per_channel_to_bf16(store, prefix, gpu);
    }
    let scale_shape_n = s.shape[0];
    let scale_shape_k = s.shape[1];
    ensure!(
        scale_shape_n == n && scale_shape_k == num_groups,
        "MXFP8 scale shape [{scale_shape_n}, {scale_shape_k}] != expected [{n}, {num_groups}]"
    );
    let scale_size = n * num_groups;
    let mut scale_buf = vec![0u8; scale_size];
    gpu.copy_d2h(s.ptr, &mut scale_buf)?;

    // Dequant on CPU: bf16[i, j] = fp8[i, j] * 2^(e8m0[i, j/32] - 127)
    let bf16_size = n * k;
    let mut bf16_buf: Vec<u8> = Vec::with_capacity(bf16_size * 2);
    for i in 0..n {
        let row_offset = i * k;
        let scale_row_offset = i * num_groups;
        for g in 0..num_groups {
            let e8m0 = scale_buf[scale_row_offset + g];
            for jj in 0..group_size {
                let fp8_byte = fp8_buf[row_offset + g * group_size + jj];
                let f32_val = mxfp8_dequant_element(fp8_byte, e8m0);
                let bf16_val = f32_to_bf16(f32_val);
                bf16_buf.extend_from_slice(&bf16_val.to_le_bytes());
            }
        }
    }

    // Upload BF16 to GPU
    let ptr = gpu.alloc(bf16_buf.len())?;
    gpu.copy_h2d(&bf16_buf, ptr)?;
    tracing::debug!(
        "MXFP8 dequant complete: {prefix} → BF16 [{n},{k}] uploaded to ptr={}",
        ptr.0,
    );

    Ok(DenseWeight { weight: ptr })
}

/// Variant of `dequant_mxfp8_to_bf16` that reads MXFP8 data from
/// `weight_packed` instead of `weight`. PrismaQuant checkpoints may
/// store MXFP8 FP8E4M3 bytes under the `weight_packed` tensor name
/// (the standard NVFP4 convention), even though the data is not
/// NVFP4 E2M1. The scale tensor (`weight_scale`, uint8 E8M0)
/// distinguishes MXFP8 from NVFP4 at the detection layer.
pub(crate) fn dequant_mxfp8_packed_to_bf16(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let w = store.get(&format!("{prefix}.weight_packed"))?;
    ensure!(
        w.shape.len() == 2,
        "Expected 2D weight_packed for {prefix}, got {:?}",
        w.shape
    );
    let n = w.shape[0];
    let k = w.shape[1];
    let group_size = 32usize;
    ensure!(
        k % group_size == 0,
        "MXFP8 requires K ({k}) divisible by group_size ({group_size})"
    );
    let num_groups = k / group_size;

    let fp8_size = n * k;
    tracing::debug!("MXFP8 packed dequant: {prefix} shape=[{n},{k}] groups={num_groups}");

    let mut fp8_buf = vec![0u8; fp8_size];
    gpu.copy_d2h(w.ptr, &mut fp8_buf).with_context(|| {
        let free = gpu.free_memory().unwrap_or(0);
        format!(
            "MXFP8 packed D2H failed for {prefix}.weight_packed: ptr={}, size={fp8_size}, free={:.1} GB",
            w.ptr.0,
            free as f64 / (1024.0 * 1024.0 * 1024.0),
        )
    })?;

    let s = store.get(&format!("{prefix}.weight_scale"))?;
    if s.shape.len() == 1 || (s.shape.len() == 2 && s.shape[1] <= 1 && s.shape[0] == n) {
        tracing::debug!(
            "{prefix}: scale shape {:?} → redirecting to per-channel FP8 dequant",
            s.shape
        );
        return dequant_fp8_per_channel_to_bf16(store, prefix, gpu);
    }
    let scale_shape_n = s.shape[0];
    let scale_shape_k = s.shape[1];
    ensure!(
        scale_shape_n == n && scale_shape_k == num_groups,
        "MXFP8 packed scale shape [{scale_shape_n}, {scale_shape_k}] != expected [{n}, {num_groups}]"
    );
    let scale_size = n * num_groups;
    let mut scale_buf = vec![0u8; scale_size];
    gpu.copy_d2h(s.ptr, &mut scale_buf)?;

    let bf16_size = n * k;
    let mut bf16_buf: Vec<u8> = Vec::with_capacity(bf16_size * 2);
    for i in 0..n {
        let row_offset = i * k;
        let scale_row_offset = i * num_groups;
        for g in 0..num_groups {
            let e8m0 = scale_buf[scale_row_offset + g];
            for jj in 0..group_size {
                let fp8_byte = fp8_buf[row_offset + g * group_size + jj];
                let f32_val = mxfp8_dequant_element(fp8_byte, e8m0);
                let bf16_val = f32_to_bf16(f32_val);
                bf16_buf.extend_from_slice(&bf16_val.to_le_bytes());
            }
        }
    }

    let ptr = gpu.alloc(bf16_buf.len())?;
    gpu.copy_h2d(&bf16_buf, ptr)?;
    tracing::debug!(
        "MXFP8 packed dequant complete: {prefix} → BF16 [{n},{k}] uploaded to ptr={}",
        ptr.0,
    );

    Ok(DenseWeight { weight: ptr })
}
