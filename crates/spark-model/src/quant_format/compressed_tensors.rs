// SPDX-License-Identifier: AGPL-3.0-only

//! Neural Magic `llm-compressor` / compressed-tensors NVFP4 serialization.
//!
//! The dominant community NVFP4 format, shipped by `Sehyo/*`, `RedHatAI/*`,
//! `nm-testing/*`, and most third-party re-quants. Tensor-name convention:
//!
//! | field                | tensor name            | dtype         |
//! | -------------------- | ---------------------- | ------------- |
//! | packed FP4 payload   | `.weight_packed`       | uint8 packed  |
//! | per-group FP8 scales | `.weight_scale`        | float8_e4m3   |
//! | per-tensor scalar    | `.weight_global_scale` | f32 scalar    |
//! | activation scale     | `.input_global_scale`  | f32 scalar    |
//!
//! Scale convention: `weight_global_scale` is the RECIPROCAL of ModelOpt's
//! `weight_scale_2` (verified empirically — see `quantized_v2` comment in
//! `weight_map.rs`).
//!
//! Unquantized modules are declared either in the top-level `ignore`
//! array or in `config_groups.group_N.targets` / `exclude_modules`
//! (vLLM style). Both are folded into a single list during config parse.
//!
//! For PrismaQuant mixed-precision checkpoints (`format = "mixed-precision"`),
//! `config_groups` map module-path patterns to format names. The `variant_for`
//! method checks these groups to dispatch per-layer variants (NVFP4, MXFP8, BF16).

use crate::quant_format::{QuantFormat, module_matches_pattern};
use crate::weight_map::Nvfp4Variant;

/// compressed-tensors NVFP4 (or mixed-precision) checkpoint.
#[derive(Debug)]
pub struct CompressedTensorsFormat {
    /// `format` string from config (e.g. `"nvfp4-pack-quantized"`, `"mixed-precision"`).
    pub format: String,
    /// Module-path globs that stay BF16 rather than NVFP4.
    pub ignore_modules: Vec<String>,
    /// Per-format group targets from PrismaQuant `config_groups`.
    /// Each entry maps `(format_name, Vec<target_pattern>)`.
    /// Used in `variant_for()` for per-layer format dispatch.
    pub config_groups: Vec<(String, Vec<String>)>,
}

impl CompressedTensorsFormat {
    pub fn new(format: String, ignore_modules: Vec<String>) -> Self {
        Self {
            format,
            ignore_modules,
            config_groups: Vec::new(),
        }
    }

    pub fn with_groups(
        format: String,
        ignore_modules: Vec<String>,
        config_groups: Vec<(String, Vec<String>)>,
    ) -> Self {
        Self {
            format,
            ignore_modules,
            config_groups,
        }
    }
}

/// Map a PrismaQuant format string to the corresponding `Nvfp4Variant`.
fn format_to_variant(fmt: &str) -> Option<Nvfp4Variant> {
    let lower = fmt.to_ascii_lowercase();
    if lower.contains("nvfp4") || lower.contains("fp4") {
        Some(Nvfp4Variant::CompressedTensors)
    } else if lower.contains("mxfp8") || (lower.contains("fp8") && lower.contains("mx")) {
        Some(Nvfp4Variant::MxFp8)
    } else if lower.contains("bf16")
        || lower.contains("bfloat16")
        || lower.contains("float-quantized")
    {
        Some(Nvfp4Variant::Bf16Raw)
    } else if lower.contains("fp8") || lower.contains("float8") {
        Some(Nvfp4Variant::Fp8Dequanted)
    } else {
        None
    }
}

impl QuantFormat for CompressedTensorsFormat {
    fn name(&self) -> &'static str {
        "compressed-tensors"
    }

    fn base_variant(&self) -> Nvfp4Variant {
        Nvfp4Variant::CompressedTensors
    }

    fn is_ignored(&self, module_path: &str) -> bool {
        self.ignore_modules
            .iter()
            .any(|pat| module_matches_pattern(module_path, pat))
    }

    fn variant_for(&self, module_path: &str) -> Nvfp4Variant {
        // 1. Explicit ignore takes priority (user-excluded modules stay BF16)
        if self.is_ignored(module_path) {
            return Nvfp4Variant::Bf16Raw;
        }

        // 2. Check config_groups for per-layer format assignment (PrismaQuant)
        for (format_name, targets) in &self.config_groups {
            for pattern in targets {
                if module_matches_pattern(module_path, pattern) {
                    if let Some(variant) = format_to_variant(format_name) {
                        tracing::debug!(
                            "Mixed-precision: {module_path} → {format_name} ({variant:?})"
                        );
                        return variant;
                    }
                    // Unknown format — fall through to base variant.
                    // This is a configuration error: the quantization_config declares
                    // a format name that format_to_variant() doesn't recognize.
                    // Common causes: typo in format name, unsupported format string,
                    // or a PrismaQuant format that needs a new arm in format_to_variant().
                    tracing::warn!(
                        "Mixed-precision: {module_path} matched UNKNOWN format '{format_name}' \
                         — using base variant CompressedTensors. This may cause incorrect \
                         weight loading if '{format_name}' should map to MxFp8 or Bf16Raw. \
                         Known formats: nvfp4, mxfp8, bf16, float-quantized, fp8."
                    );
                    break;
                }
            }
        }

        // 3. Fall back to base variant (uniform compressed-tensors)
        self.base_variant()
    }
}
