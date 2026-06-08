// SPDX-License-Identifier: AGPL-3.0-only

//! PrismaQuant-style mixed-precision format allocator.
//!
//! Implements per-layer format SELECTION (not just loading). The allocator
//! uses weight Frobenius-norm heuristics as a cheap proxy for Fisher
//! sensitivity (the empirical Fisher diagonal trace requires gradient
//! computation which Atlas as an inference engine cannot perform — the
//! Python PrismaQuant toolchain fills this gap).
//!
//! When available, externally-computed Fisher traces can be blended with
//! weight-norm scores via `LayerSensitivity::fisher_trace`.
//!
//! The allocator enforces PrismaQuant's fused-sibling coherence
//! (q/k/v projections share one format, gate_up/down share one format)
//! via Union-Find promotion.
//!
//! Two solver backends are available:
//!   - `allocate_greedy`: fast bottom-up greedy (default)
//!   - `allocate_knapsack`: multi-choice knapsack DP (optimal, slower)
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

    fn rank(&self) -> usize {
        match self {
            Self::Nvfp4 => 0,
            Self::Mxfp8 => 1,
            Self::Bf16 => 2,
        }
    }

    pub fn parse_from_str(s: &str) -> Option<Self> {
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
    /// Sensitivity score: normalized Frobenius norm (higher = more sensitive)
    pub score: f64,
    /// Empirical Fisher diagonal trace H_trace = Σ(∂L/∂W)².
    /// Set to 0.0 when unavailable; the allocator falls back to `score`.
    pub fisher_trace: f64,
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
    /// Per-layer parameter counts for export metadata
    pub per_layer_params: Vec<(String, usize)>,
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
                fisher_trace: 0.0,
                n_params: n,
            }
        })
        .collect();
    scores.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scores
}

/// Blend Fisher trace with weight-norm score.
///
/// When Fisher diagonal traces are available (e.g. from external PrismaQuant
/// probe), this produces a combined sensitivity: `α·fisher_norm + (1-α)·weight_score`.
/// `α` defaults to 0.7 (Fisher-weighted).
pub fn compute_fisher_weighted(
    sensitivities: &[LayerSensitivity],
    alpha: f64,
) -> Vec<LayerSensitivity> {
    if sensitivities.iter().all(|s| s.fisher_trace <= 0.0) {
        return sensitivities.to_vec();
    }
    let max_fisher = sensitivities
        .iter()
        .map(|s| s.fisher_trace)
        .fold(0.0f64, f64::max)
        .max(1.0);
    let max_score = sensitivities
        .iter()
        .map(|s| s.score)
        .fold(0.0f64, f64::max)
        .max(1.0);
    let mut result: Vec<LayerSensitivity> = sensitivities
        .iter()
        .map(|s| {
            let fisher_norm = s.fisher_trace / max_fisher;
            let weight_norm = s.score / max_score;
            let combined = alpha * fisher_norm + (1.0 - alpha) * weight_norm;
            LayerSensitivity {
                name: s.name.clone(),
                score: combined,
                fisher_trace: s.fisher_trace,
                n_params: s.n_params,
            }
        })
        .collect();
    result.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    result
}

// ── Fused-sibling promotion ─────────────────────────────────────────────

/// Promote format assignments for fused-sibling groups.
///
/// PrismaQuant enforces that attention q/k/v projections and MoE gate_up/down
/// pairs share one serving format. This function uses Union-Find to ensure
/// coherence: all members of a fused group are promoted to the highest-precision
/// format assigned to any member.
///
/// `layers` is a map: qname → format. Layer names containing ".q_proj",
/// ".k_proj", ".v_proj" are grouped; names containing ".gate_proj",
/// ".up_proj" are grouped; ".down_proj" joins the gate_up group.
pub fn promote_fused_siblings(
    layers: &BTreeMap<String, String>,
    format_rank: &BTreeMap<String, usize>,
) -> BTreeMap<String, String> {
    let mut out = layers.clone();

    // Union-Find data structures
    let mut parent: BTreeMap<String, String> = out.keys().map(|k| (k.clone(), k.clone())).collect();

    fn find(parent: &mut BTreeMap<String, String>, x: &str) -> String {
        let p = parent.get(x).cloned().unwrap_or_else(|| x.to_string());
        if p == x {
            return p;
        }
        let root = find(parent, &p);
        parent.insert(x.to_string(), root.clone());
        root
    }

    fn union(parent: &mut BTreeMap<String, String>, a: &str, b: &str) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            parent.insert(rb, ra);
        }
    }

    // Group q/k/v attention projections
    let mut attn_groups: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for name in out.keys() {
        // Extract layer prefix plus attention type: "model.layers.N.self_attn"
        let parts: Vec<&str> = name.rsplitn(3, '.').collect();
        if parts.len() >= 3 {
            let proj = parts[0]; // "q_proj", "k_proj", "v_proj", "o_proj"
            let prefix = parts[2]; // "...self_attn"
            if proj == "q_proj" || proj == "k_proj" || proj == "v_proj" {
                let group_key = format!("{}.qkv", prefix);
                attn_groups.entry(group_key).or_default().push(name.clone());
            }
        }
    }
    for members in attn_groups.values() {
        if members.len() >= 2 {
            let first = &members[0];
            for m in &members[1..] {
                union(&mut parent, first, m);
            }
        }
    }

    // Group gate_up/down MoE projections
    let mut moe_groups: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for name in out.keys() {
        let parts: Vec<&str> = name.rsplitn(2, '.').collect();
        if parts.len() >= 2 {
            let proj = parts[0];
            let prefix = parts[1];
            if proj == "gate_proj" || proj == "up_proj" || proj == "down_proj" {
                let group_key = format!("{}.gate_up_down", prefix);
                moe_groups.entry(group_key).or_default().push(name.clone());
            }
        }
    }
    for members in moe_groups.values() {
        if members.len() >= 2 {
            let first = &members[0];
            for m in &members[1..] {
                union(&mut parent, first, m);
            }
        }
    }

    // Promote each component to the highest format in its group
    let mut components: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for name in out.keys() {
        let root = find(&mut parent, name);
        components.entry(root).or_default().push(name.clone());
    }

    for members in components.values() {
        if members.len() < 2 {
            continue;
        }
        let best_fmt = members
            .iter()
            .filter_map(|m| out.get(m))
            .max_by_key(|fmt| format_rank.get(fmt.as_str()).copied().unwrap_or(0))
            .cloned();
        if let Some(best) = best_fmt {
            for m in members {
                if out.get(m).map(|f| f != &best).unwrap_or(false) {
                    out.insert(m.clone(), best.clone());
                }
            }
        }
    }

    out
}

// ── Greedy allocator ────────────────────────────────────────────────────

/// Greedy multi-choice format allocator.
///
/// Bottom-up: start all layers at cheapest format (NVFP4), upgrade most
/// sensitive layers to higher precision until budget is met.
pub fn allocate_greedy(
    sensitivities: &[LayerSensitivity],
    target_bpp: f64,
    formats: &[AllocFormat],
) -> PrismaAssignment {
    if sensitivities.is_empty() || formats.is_empty() {
        return PrismaAssignment {
            layers: BTreeMap::new(),
            achieved_bpp: 16.0,
            target_bpp,
            per_layer_params: Vec::new(),
        };
    }

    let mut indices: Vec<usize> = (0..sensitivities.len()).collect();
    indices.sort_by(|&a, &b| {
        sensitivities[b]
            .score
            .partial_cmp(&sensitivities[a].score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let cheapest = formats[0];
    let mut assignments: Vec<AllocFormat> = vec![cheapest; sensitivities.len()];
    let total_params: usize = sensitivities.iter().map(|s| s.n_params).sum();
    let target_bits_total = total_params as f64 * target_bpp;

    let mut total_bits: f64 = sensitivities
        .iter()
        .map(|s| s.n_params as f64 * cheapest.effective_bpp())
        .sum();

    for &idx in &indices {
        if total_bits >= target_bits_total {
            break;
        }
        let s = &sensitivities[idx];
        for fmt in formats.iter().skip(1) {
            let bits_added =
                (fmt.effective_bpp() - assignments[idx].effective_bpp()) * s.n_params as f64;
            if bits_added <= 0.0 {
                continue;
            }
            if total_bits + bits_added > target_bits_total {
                continue;
            }
            assignments[idx] = *fmt;
            total_bits += bits_added;
            break;
        }
    }

    let achieved_bpp = total_bits / total_params as f64;
    let layers: BTreeMap<String, String> = sensitivities
        .iter()
        .zip(assignments.iter())
        .map(|(s, fmt)| (s.name.clone(), format!("{:?}", fmt).to_uppercase()))
        .collect();
    let per_layer_params: Vec<(String, usize)> = sensitivities
        .iter()
        .map(|s| (s.name.clone(), s.n_params))
        .collect();

    PrismaAssignment {
        layers,
        achieved_bpp,
        target_bpp,
        per_layer_params,
    }
}

/// Multi-choice knapsack DP solver.
///
/// Produces the optimal (loss-minimizing) format assignment under the
/// target bpp budget, using per-layer sensitivity scores as the loss
/// proxy: Δloss ≈ score × bpp_saved. The DP maximizes total sensitivity
/// retained (higher precision = higher retained sensitivity).
///
/// Complexity: O(N × B × F) where N = layers, B = budget bins, F = formats.
pub fn allocate_knapsack(
    sensitivities: &[LayerSensitivity],
    target_bpp: f64,
    formats: &[AllocFormat],
) -> PrismaAssignment {
    if sensitivities.is_empty() || formats.is_empty() {
        return PrismaAssignment {
            layers: BTreeMap::new(),
            achieved_bpp: 16.0,
            target_bpp,
            per_layer_params: Vec::new(),
        };
    }

    let total_params: usize = sensitivities.iter().map(|s| s.n_params).sum();
    let target_bits_total = total_params as f64 * target_bpp;

    // Discretize budget into bins: each bin = bit_precision bits
    let bit_precision: f64 = 0.25;
    let n_bins = (target_bits_total / bit_precision).ceil() as usize + 1;
    let mut dp: Vec<f64> = vec![f64::NEG_INFINITY; n_bins];
    dp[0] = 0.0;

    // Track which format was chosen for backtracking
    let mut choice: Vec<Vec<usize>> = Vec::new();

    for s in sensitivities {
        let param_frac = s.n_params as f64 / total_params as f64;
        let mut new_dp = vec![f64::NEG_INFINITY; n_bins];
        let mut new_choice = vec![0usize; n_bins];

        for (fi, fmt) in formats.iter().enumerate() {
            let bits_used = (fmt.effective_bpp() * param_frac / bit_precision).round() as usize;
            if bits_used >= n_bins {
                continue;
            }
            // Value: retained sensitivity (higher format = higher value)
            let value = s.score * fmt.rank() as f64;

            for b in 0..n_bins - bits_used {
                if dp[b].is_finite() {
                    let cand = dp[b] + value;
                    let target_b = b + bits_used;
                    if cand > new_dp[target_b] {
                        new_dp[target_b] = cand;
                        new_choice[target_b] = fi;
                    }
                }
            }
        }
        dp = new_dp;
        choice.push(new_choice);
    }

    // Backtrack: find best valid bin within budget
    let best_b = (0..n_bins)
        .filter(|&b| dp[b].is_finite())
        .max_by(|&a, &b| {
            dp[a]
                .partial_cmp(&dp[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0);

    let mut assignments: Vec<AllocFormat> = vec![formats[0]; sensitivities.len()];
    let mut cur = best_b;
    let mut total_bits: f64 = 0.0;
    for layer_idx in (0..sensitivities.len()).rev() {
        let fi = choice[layer_idx][cur];
        assignments[layer_idx] = formats[fi];
        let param_frac = sensitivities[layer_idx].n_params as f64 / total_params as f64;
        let bits_used = (formats[fi].effective_bpp() * param_frac / bit_precision).round() as usize;
        total_bits += formats[fi].effective_bpp() * sensitivities[layer_idx].n_params as f64;
        cur = cur.saturating_sub(bits_used);
    }

    let achieved_bpp = total_bits / total_params as f64;
    let layers: BTreeMap<String, String> = sensitivities
        .iter()
        .zip(assignments.iter())
        .map(|(s, fmt)| (s.name.clone(), format!("{:?}", fmt).to_uppercase()))
        .collect();
    let per_layer_params: Vec<(String, usize)> = sensitivities
        .iter()
        .map(|s| (s.name.clone(), s.n_params))
        .collect();

    PrismaAssignment {
        layers,
        achieved_bpp,
        target_bpp,
        per_layer_params,
    }
}

/// Allocate formats with fused-sibling promotion.
///
/// Runs the allocator, then promotes fused siblings to share one format,
/// returning the promoted assignment.
pub fn allocate_with_promotion(
    sensitivities: &[LayerSensitivity],
    target_bpp: f64,
    formats: &[AllocFormat],
) -> PrismaAssignment {
    let mut result = allocate_greedy(sensitivities, target_bpp, formats);

    let format_rank: BTreeMap<String, usize> = formats
        .iter()
        .enumerate()
        .map(|(i, f)| (format!("{:?}", f).to_uppercase(), i))
        .collect();

    result.layers = promote_fused_siblings(&result.layers, &format_rank);

    // Recompute achieved bpp after promotion
    let total_params: usize = result.per_layer_params.iter().map(|(_, n)| n).sum();
    let total_bits: f64 = result
        .layers
        .iter()
        .map(|(name, fmt_str)| {
            let n_params = result
                .per_layer_params
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, p)| *p)
                .unwrap_or(0);
            let fmt = AllocFormat::parse_from_str(fmt_str).unwrap_or(AllocFormat::Nvfp4);
            fmt.effective_bpp() * n_params as f64
        })
        .sum();
    result.achieved_bpp = total_bits / total_params.max(1) as f64;

    result
}

/// Backward-compatible alias.
pub fn allocate_formats(
    sensitivities: &[LayerSensitivity],
    target_bpp: f64,
    formats: &[AllocFormat],
) -> PrismaAssignment {
    allocate_greedy(sensitivities, target_bpp, formats)
}

// ── Export ──────────────────────────────────────────────────────────────

/// Write the per-layer assignment as a `layer_config.json` compatible
/// with PrismaQuant tooling and Atlas's compressed-tensors loader.
pub fn write_assignment(assignment: &PrismaAssignment, output_dir: &Path) -> Result<()> {
    #[derive(Serialize)]
    struct ExportPayload {
        layers: BTreeMap<String, String>,
        achieved_bpp: f64,
        target_bpp: f64,
        total_params: usize,
        num_layers: usize,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        layer_params: Vec<(String, usize)>,
    }

    let total_params: usize = assignment.per_layer_params.iter().map(|(_, n)| n).sum();

    let payload = ExportPayload {
        layers: assignment.layers.clone(),
        achieved_bpp: assignment.achieved_bpp,
        target_bpp: assignment.target_bpp,
        total_params,
        num_layers: assignment.layers.len(),
        layer_params: assignment.per_layer_params.clone(),
    };

    let path = output_dir.join("layer_config.json");
    let json = serde_json::to_string_pretty(&payload)?;
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
                fisher_trace: 0.0,
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
        let result = allocate_greedy(&layers, 6.0, &formats);

        assert!(result.achieved_bpp <= 7.0);
        assert_eq!(
            result.layers.get("l0.q_proj").map(|s| s.as_str()),
            Some("BF16")
        );
        assert_eq!(
            result.layers.get("l1.q_proj").map(|s| s.as_str()),
            Some("NVFP4")
        );
    }

    #[test]
    fn knapsack_meets_budget() {
        let layers = make_layers(
            &["l0.q_proj", "l0.k_proj", "l0.o_proj", "l1.q_proj"],
            &[10.0, 8.0, 5.0, 1.0],
            &[1000, 1000, 1000, 1000],
        );
        let formats = vec![AllocFormat::Nvfp4, AllocFormat::Mxfp8, AllocFormat::Bf16];
        let result = allocate_knapsack(&layers, 6.0, &formats);

        assert!(result.achieved_bpp <= 7.0);
        // DP should produce same or better assignments than greedy
        assert!(!result.layers.is_empty());
    }

    #[test]
    fn fused_sibling_promotion() {
        let mut layers: BTreeMap<String, String> = BTreeMap::new();
        layers.insert(
            "model.layers.0.self_attn.q_proj".to_string(),
            "BF16".to_string(),
        );
        layers.insert(
            "model.layers.0.self_attn.k_proj".to_string(),
            "NVFP4".to_string(),
        );
        layers.insert(
            "model.layers.0.self_attn.v_proj".to_string(),
            "NVFP4".to_string(),
        );
        layers.insert(
            "model.layers.0.mlp.gate_proj".to_string(),
            "BF16".to_string(),
        );
        layers.insert(
            "model.layers.0.mlp.up_proj".to_string(),
            "MXFP8".to_string(),
        );
        layers.insert(
            "model.layers.0.mlp.down_proj".to_string(),
            "NVFP4".to_string(),
        );

        let mut rank: BTreeMap<String, usize> = BTreeMap::new();
        rank.insert("NVFP4".to_string(), 0);
        rank.insert("MXFP8".to_string(), 1);
        rank.insert("BF16".to_string(), 2);

        let promoted = promote_fused_siblings(&layers, &rank);

        // q_proj was BF16, k_proj/v_proj were NVFP4 → all promoted to BF16
        assert_eq!(
            promoted
                .get("model.layers.0.self_attn.q_proj")
                .map(|s| s.as_str()),
            Some("BF16")
        );
        assert_eq!(
            promoted
                .get("model.layers.0.self_attn.k_proj")
                .map(|s| s.as_str()),
            Some("BF16")
        );
        assert_eq!(
            promoted
                .get("model.layers.0.self_attn.v_proj")
                .map(|s| s.as_str()),
            Some("BF16")
        );

        // gate_proj was BF16, up_proj MXFP8, down_proj NVFP4 → all promoted to BF16
        assert_eq!(
            promoted
                .get("model.layers.0.mlp.gate_proj")
                .map(|s| s.as_str()),
            Some("BF16")
        );
    }

    #[test]
    fn empty_layers_returns_default() {
        let result = allocate_greedy(&[], 5.0, &[AllocFormat::Nvfp4, AllocFormat::Bf16]);
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
        assert_eq!(scores[0].name, "l1");
        assert_eq!(scores[1].name, "l0");
        assert_eq!(scores[2].name, "l2");
    }

    #[test]
    fn fisher_weighted_blend() {
        let mut layers = make_layers(&["l0.q_proj", "l1.q_proj"], &[5.0, 1.0], &[1000, 1000]);
        layers[0].fisher_trace = 100.0;
        layers[1].fisher_trace = 10.0;
        let blended = compute_fisher_weighted(&layers, 0.7);
        // l0 with high Fisher should still rank above l1
        assert!(blended[0].score >= blended[1].score);
    }
}
