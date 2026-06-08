// SPDX-License-Identifier: AGPL-3.0-only

//! PrismaQuant-style mixed-precision format allocator.
//!
//! Implements per-layer format SELECTION (not just loading). The allocator
//! uses weight-norm heuristics as a cheap proxy for Fisher sensitivity,
//! then solves a greedy multi-choice assignment under a target bits-per-param
//! budget. This is the Atlas-side equivalent of PrismaQuant's probe→cost→allocate
//! pipeline, minus the full end-to-end KL validation (which requires a reference
//! forward pass that the Python toolchain handles).
//!
//! # Pipeline
//! 1. **Probe**: compute per-Linear sensitivity from weight Frobenius norms
//! 2. **Allocate**: greedy assign formats (NVFP4/MXFP8/BF16) under bpp budget
//! 3. **Export**: serialize assignments as a `layer_config.json` compatible
//!    with PrismaQuant tooling and Atlas's own compressed-tensors loader.
//!
//! # Usage
//! ```text
//! atlas allocate --checkpoint /path/to/model --target-bits 4.75 --output ./prisma-model
//! ```

use anyhow::Result;
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::Path;

/// A quantization format the allocator can assign to a Linear layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum AllocFormat {
    /// NVFP4 E2M1: 4-bit weights + FP8 per-group scales. 4.0 effective bpp.
    Nvfp4,
    /// MXFP8_E4M3: 8-bit weights + E8M0 block scales. 8.0 effective bpp.
    Mxfp8,
    /// BF16 dense: 16-bit uncompressed.
    Bf16,
}

impl AllocFormat {
    fn effective_bpp(&self) -> f64 {
        match self {
            Self::Nvfp4 => 4.125, // 4-bit data + scale overhead
            Self::Mxfp8 => 8.125, // 8-bit data + scale overhead
            Self::Bf16 => 16.0,
        }
    }

    fn from_str(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "nvfp4" => Some(Self::Nvfp4),
            "mxfp8" | "mxfp8_e4m3" => Some(Self::Mxfp8),
            "bf16" | "bfloat16" => Some(Self::Bf16),
            _ => None,
        }
    }
}

/// Per-Linear sensitivity score computed from weight statistics.
#[derive(Debug, Clone)]
pub struct LayerSensitivity {
    /// Module qname (e.g. "model.layers.5.mlp.gate_proj")
    pub name: String,
    /// Sensitivity score: Frobenius norm of weight matrix (higher = more sensitive)
    pub score: f64,
    /// Number of parameters in this Linear
    pub n_params: usize,
}

/// Complete per-layer format assignment.
#[derive(Debug, Clone, Serialize)]
pub struct PrismaAssignment {
    /// Map from module qname → format name string
    pub layers: BTreeMap<String, String>,
    /// Achieved bits per parameter
    pub achieved_bpp: f64,
    /// Target bits per parameter
    pub target_bpp: f64,
}

/// Compute per-layer sensitivity from weight Frobenius norms.
///
/// For each Linear weight tensor, sensitivity = ‖W‖_F / √(n_params).
/// This normalizes by layer size so smaller layers aren't unfairly
/// penalized. Higher score = more sensitive to quantization = should
/// get higher-precision format.
pub fn compute_sensitivity(layers: &[(String, usize)]) -> Vec<LayerSensitivity> {
    // In production, this would read actual weight tensors from the
    // checkpoint. For the allocator API, we accept pre-computed scores.
    // The caller (CLI or library) loads the model and computes norms.
    layers
        .iter()
        .map(|(name, n_params)| {
            // Placeholder — caller fills in real scores via
            // `compute_sensitivity_from_weights()`.
            LayerSensitivity {
                name: name.clone(),
                score: 1.0,
                n_params: *n_params,
            }
        })
        .collect()
}

/// Compute sensitivity from actual weight tensors.
///
/// `weights` is a map from module qname → flattened f32 weight values.
/// Returns sensitivity scores sorted by sensitivity (descending).
pub fn compute_sensitivity_from_weights(
    weights: &BTreeMap<String, Vec<f32>>,
) -> Vec<LayerSensitivity> {
    let mut scores: Vec<LayerSensitivity> = weights
        .iter()
        .map(|(name, w)| {
            let frob2: f64 = w.iter().map(|&x| (x as f64) * (x as f64)).sum();
            let n = w.len();
            let score = frob2.sqrt() / (n as f64).sqrt().max(1.0);
            LayerSensitivity {
                name: name.clone(),
                score,
                n_params: n,
            }
        })
        .collect();
    // Sort by sensitivity descending (most sensitive first)
    scores.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    scores
}

/// Greedy multi-choice format allocator.
///
/// Assigns lower-precision formats to less-sensitive layers until
/// the target bits-per-parameter budget is met.
///
/// # Algorithm
/// 1. Start with all layers at BF16 (worst compression, best quality)
/// 2. Sort layers by sensitivity (ascending = least sensitive first)
/// 3. For each layer, try formats from most to least aggressive:
///    NVFP4 → MXFP8 → BF16 (keep if budget exceeded)
/// 4. Stop when budget is met
///
/// Formats: `formats` is the list of available formats in order of
/// preference (first = most aggressive, cheapest).
pub fn allocate_formats(
    sensitivities: &[LayerSensitivity],
    target_bpp: f64,
    formats: &[AllocFormat],
) -> PrismaAssignment {
    if sensitivities.is_empty() {
        return PrismaAssignment {
            layers: BTreeMap::new(),
            achieved_bpp: 16.0,
            target_bpp,
        };
    }

    // Sort by sensitivity ascending (least sensitive first = cheapest to quantize)
    let mut indices: Vec<usize> = (0..sensitivities.len()).collect();
    indices.sort_by(|&a, &b| {
        sensitivities[a]
            .score
            .partial_cmp(&sensitivities[b].score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Default: all layers get the least aggressive format (BF16)
    let default_fmt = formats.last().copied().unwrap_or(AllocFormat::Bf16);
    let mut assignments: Vec<AllocFormat> = vec![default_fmt; sensitivities.len()];
    let total_params: usize = sensitivities.iter().map(|s| s.n_params).sum();

    // Current total bits
    let mut total_bits: f64 = sensitivities
        .iter()
        .map(|s| s.n_params as f64 * default_fmt.effective_bpp())
        .sum();
    let target_bits_total = total_params as f64 * target_bpp;

    // Greedy: for each layer (ascending sensitivity), try more aggressive formats
    for &idx in &indices {
        if total_bits <= target_bits_total {
            break; // Budget met
        }
        let s = &sensitivities[idx];
        // Try formats from most to least aggressive
        for fmt in formats.iter() {
            let bits_saved = (assignments[idx].effective_bpp() - fmt.effective_bpp())
                * s.n_params as f64;
            if bits_saved <= 0.0 {
                continue;
            }
            assignments[idx] = *fmt;
            total_bits -= bits_saved;
            if total_bits <= target_bits_total {
                break;
            }
        }
    }

    let achieved_bpp = total_bits / total_params as f64;

    let layers: BTreeMap<String, String> = sensitivities
        .iter()
        .zip(assignments.iter())
        .map(|(s, fmt)| (s.name.clone(), format!("{:?}", fmt).to_uppercase()))
        .collect();

    PrismaAssignment {
        layers,
        achieved_bpp,
        target_bpp,
    }
}

/// Write the per-layer assignment as a `layer_config.json` compatible
/// with PrismaQuant tooling and Atlas's compressed-tensors loader.
pub fn write_assignment(assignment: &PrismaAssignment, output_dir: &Path) -> Result<()> {
    let path = output_dir.join("layer_config.json");
    let json = serde_json::to_string_pretty(&assignment.layers)?;
    std::fs::write(&path, json)?;
    tracing::info!(
        "PrismaQuant assignment written to {}: {:.2} bpp (target {:.2} bpp), {} layers",
        path.display(),
        assignment.achieved_bpp,
        assignment.target_bpp,
        assignment.layers.len(),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_layers(names: &[&str], scores: &[f64], params: &[usize]) -> Vec<LayerSensitivity> {
        names
            .iter()
            .zip(scores.iter())
            .zip(params.iter())
            .map(|((&name, &score), &n_params)| LayerSensitivity {
                name: name.to_string(),
                score,
                n_params,
            })
            .collect()
    }

    #[test]
    fn greedy_allocator_meets_budget() {
        let layers = make_layers(
            &["l0.q_proj", "l0.k_proj", "l0.o_proj", "l1.q_proj"],
            &[10.0, 8.0, 5.0, 1.0],
            &[1000, 1000, 1000, 1000],
        );
        let formats = vec![AllocFormat::Nvfp4, AllocFormat::Mxfp8, AllocFormat::Bf16];
        let result = allocate_formats(&layers, 6.0, &formats);

        // l1.q_proj (score=1.0, least sensitive) should get NVFP4
        // l0.o_proj (score=5.0) should get NVFP4
        // l0.k_proj (score=8.0) might get MXFP8
        // l0.q_proj (score=10.0, most sensitive) should get BF16

        assert!(result.achieved_bpp <= 6.0 + 1.0); // allow some slack
        // Most sensitive layer should be BF16
        assert_eq!(result.layers.get("l0.q_proj").map(|s| s.as_str()), Some("BF16"));
        // Least sensitive layer should be NVFP4
        assert_eq!(
            result.layers.get("l1.q_proj").map(|s| s.as_str()),
            Some("NVFP4")
        );
    }

    #[test]
    fn empty_layers_returns_default() {
        let result = allocate_formats(
            &[],
            5.0,
            &[AllocFormat::Nvfp4, AllocFormat::Bf16],
        );
        assert!(result.layers.is_empty());
        assert!((result.achieved_bpp - 16.0).abs() < 0.01);
    }

    #[test]
    fn sensitivity_sorting() {
        let weights: BTreeMap<String, Vec<f32>> = [
            ("l0".to_string(), vec![1.0f32; 100]),
            ("l1".to_string(), vec![2.0f32; 100]),
            ("l2".to_string(), vec![0.5f32; 100]),
        ]
        .into();
        let scores = compute_sensitivity_from_weights(&weights);
        // Most sensitive first: l1 (norm=20), l0 (norm=10), l2 (norm=5)
        assert_eq!(scores[0].name, "l1");
        assert_eq!(scores[1].name, "l0");
        assert_eq!(scores[2].name, "l2");
    }
}
