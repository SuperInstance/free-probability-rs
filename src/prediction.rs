//! Gradient analysis and eigenvalue-distribution prediction for deep networks.
//!
//! Uses free probability (R-transform, S-transform, Marchenko–Pastur law) to:
//!
//! - Predict the combined eigenvalue spectrum when stacking multiple layers
//! - Suggest weight initialisation scales from Marchenko–Pastur theory
//! - Recommend regularisation strength from eigenvalue tail behaviour
//!
//! ## Why this matters
//!
//! Xavier init, He init, and Kaiming init all correspond to setting
//! the weight variance so that the Marchenko–Pastur spectral bulk stays
//! within a bounded range.  Free probability is the unifying language.

use crate::moments;

/// Parameters describing a transformer-like architecture.
#[derive(Debug, Clone)]
pub struct TransformerConfig {
    pub n_layers: usize,
    pub hidden_dim: f64,
    pub n_samples: f64,
    pub weight_std: f64,
    pub learning_rate: f64,
}

/// Initialisation suggestions derived from Marchenko–Pastur theory.
#[derive(Debug, Clone, PartialEq)]
pub struct InitSuggestion {
    pub suggested_std: f64,
    pub suggested_lr_scale: f64,
    pub condition_number: f64,
    pub tail_mass: f64,
}

/// Regularisation suggestions from eigenvalue tail analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct RegularizerSuggestion {
    pub lambda_reg: f64,
    pub spectral_radius: f64,
    pub outlier_fraction: f64,
    pub stability_score: f64,
}

/// Predict the combined eigenvalue distribution when stacking `n_layers`
/// freely independent layers, each with the same eigenvalue cumulants.
///
/// Because `R_{combined}(z) = Σ R_i(z)`, if all layers share cumulants:
///
/// ```text
/// κ_combined[k] = n_layers · κ_single[k]
/// ```
///
/// This lets us predict the spectrum of a deep network's covariance from
/// a single-layer measurement without ever forming the full matrix.
pub fn predict_combined_distribution(
    layer_cumulants: &[f64],
    n_layers: usize,
    combined_cumulants: &mut [f64],
) {
    let n = layer_cumulants.len().min(combined_cumulants.len());
    for k in 0..n {
        combined_cumulants[k] = n_layers as f64 * layer_cumulants[k];
    }
}

/// Suggest a regularisation strength from eigenvalue moment data.
///
/// Uses the following heuristic based on free probability:
///
/// - Compute free cumulants from moments
/// - Estimate the spectral radius as `3σ` (3-sigma rule for eigenvalues)
/// - Compare with the Marchenko–Pastur upper edge to detect outliers
/// - Adjust λ_reg proportionally to kurtosis excess and outlier fraction
/// - Scale by `√(n_layers)` (deeper → more cautious)
pub fn suggest_regularizer(
    moments: &[f64],
    config: &TransformerConfig,
) -> RegularizerSuggestion {
    if moments.is_empty() {
        return RegularizerSuggestion {
            lambda_reg: 1e-4,
            spectral_radius: 1.0,
            outlier_fraction: 0.0,
            stability_score: 1.0,
        };
    }

    let n = moments.len();
    let mut cumulants = vec![0.0_f64; n];
    moments::moment_to_cumulant(moments, &mut cumulants);

    let variance = if n >= 2 { moments[1] } else { 1.0 };
    let mean = if n >= 1 { moments[0] } else { 0.0 };
    let centered_var = (variance - mean * mean).max(0.0);

    let spectral_radius = centered_var.sqrt() * 3.0;

    // Compare with Marchenko–Pastur upper edge
    let lambda = (config.hidden_dim / config.n_samples.max(1.0))
        .max(0.01)
        .min(4.0);

    let mp_b = (1.0 + lambda.sqrt()) * (1.0 + lambda.sqrt());
    let mp_scale = mp_b.sqrt().max(1.0);
    let ratio = spectral_radius / mp_scale;

    let outlier_fraction = if ratio > 1.0 {
        ((ratio - 1.0) / ratio).min(1.0)
    } else {
        0.0
    };

    // Kurtosis excess = m₄/m₂² - 3 (for centred distribution)
    let kurtosis_excess = if n >= 4 && centered_var > 1e-12 {
        let m4 = moments[3];
        let m2 = centered_var;
        (m4 / (m2 * m2) - 3.0).max(0.0)
    } else {
        0.0
    };

    let lambda_reg = 1e-4
        * (1.0 + kurtosis_excess * 0.5)
        * (1.0 + outlier_fraction)
        * (config.n_layers as f64).sqrt();

    let stability_score = (1.0 / (1.0 + 0.1 * kurtosis_excess + outlier_fraction))
        .clamp(0.0, 1.0);

    RegularizerSuggestion {
        lambda_reg,
        spectral_radius,
        outlier_fraction,
        stability_score,
    }
}

/// Suggest a weight initialisation standard deviation from Marchenko–Pastur theory.
///
/// For a random weight matrix of size `d × d` (hidden_dim × hidden_dim),
/// the MP law with ratio `λ = d / n_samples` constrains the spectral bulk.
///
/// Xavier/He-like scaling: `σ = 1/√(hidden_dim)` gives eigenvalues in `[0, 4]`.
/// Depth scaling: divide by `√(n_layers)` for deeper nets.
pub fn suggest_initialization(config: &TransformerConfig) -> InitSuggestion {
    if config.hidden_dim <= 0.0 || config.n_samples <= 0.0 || config.n_layers == 0 {
        return InitSuggestion {
            suggested_std: 0.02,
            suggested_lr_scale: 1.0,
            condition_number: 1.0,
            tail_mass: 0.0,
        };
    }

    let d = config.hidden_dim;
    let n = config.n_samples.max(d);
    let lambda = d / n;

    // Xavier/He init: base_std = √(2/d) ≈ O(1/√d).
    // MP theory says: for W_{ij} ~ N(0, σ²/d), eigenvalues of WWᵀ/n
    // have support [σ²(1-√λ)², σ²(1+√λ)²].  Setting σ=1 gives support [0,4]
    // at λ=1 for square matrices.
    let base_std = 1.0 / d.sqrt();
    let suggested_std = (base_std / (config.n_layers as f64).sqrt())
        .clamp(1e-6, 1.0);

    // MP upper / lower edges with the suggested init
    // For W entries with std σ/√d, λ = d/n:
    //   support = [ σ²(1-√λ)² , σ²(1+√λ)² ]
    // where the eigenvalue scaling is independent of d.
    let mp_upper = (1.0 + lambda.sqrt()) * (1.0 + lambda.sqrt());
    let mp_lower = (1.0 - lambda.sqrt()).max(1e-12) * (1.0 - lambda.sqrt()).max(1e-12);

    let condition_number = mp_upper / mp_lower.max(1e-12);

    // Learning rate inversely proportional to spectral radius
    let suggested_lr_scale =
        (1.0 / (1.0 + 0.1 * mp_upper.sqrt())).clamp(0.01, 1.0);

    InitSuggestion {
        suggested_std,
        suggested_lr_scale,
        condition_number,
        tail_mass: 0.05, // heuristic default
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_combined_distribution() {
        let layer_cum = vec![1.0, 0.5, 0.1, 0.01];
        let mut combined = vec![0.0_f64; 4];

        predict_combined_distribution(&layer_cum, 4, &mut combined);

        assert!((combined[0] - 4.0).abs() < 1e-12);
        assert!((combined[1] - 2.0).abs() < 1e-12);
        assert!((combined[2] - 0.4).abs() < 1e-12);
        assert!((combined[3] - 0.04).abs() < 1e-12);

        // Verify R-transform additivity
        let z = 0.5;
        let r_single = crate::r_transform::from_cumulants(&layer_cum, z);
        let r_combined = crate::r_transform::from_cumulants(&combined, z);
        assert!((r_combined - 4.0 * r_single).abs() < 1e-12);
    }

    #[test]
    fn test_suggest_initialization() {
        let config = TransformerConfig {
            n_layers: 12,
            hidden_dim: 768.0,
            n_samples: 10000.0,
            weight_std: 0.02,
            learning_rate: 0.001,
        };

        let sug = suggest_initialization(&config);

        assert!(sug.suggested_std > 0.0, "std = {}", sug.suggested_std);
        assert!(sug.suggested_std.is_finite(), "std is NaN/Inf");
        assert!(sug.suggested_std < 1.0, "std = {} (too large)", sug.suggested_std);
        assert!(sug.suggested_std > 1e-10, "std = {} (too small)", sug.suggested_std);
        assert!(sug.suggested_lr_scale > 0.0, "lr_scale = {}", sug.suggested_lr_scale);
        assert!(sug.suggested_lr_scale.is_finite(), "lr_scale is NaN/Inf");
    }

    #[test]
    fn test_suggest_initialization_null_config_defaults() {
        let config = TransformerConfig {
            n_layers: 0,
            hidden_dim: 0.0,
            n_samples: 0.0,
            weight_std: 0.02,
            learning_rate: 0.001,
        };

        let sug = suggest_initialization(&config);

        assert!((sug.suggested_std - 0.02).abs() < 1e-12);
        assert!((sug.suggested_lr_scale - 1.0).abs() < 1e-12);
        assert!((sug.condition_number - 1.0).abs() < 1e-12);
        assert!((sug.tail_mass - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_suggest_regularizer() {
        let moments = vec![0.0, 1.0, 0.0, 3.0, 0.0, 15.0];

        let config = TransformerConfig {
            n_layers: 6,
            hidden_dim: 512.0,
            n_samples: 5000.0,
            weight_std: 0.02,
            learning_rate: 0.001,
        };

        let sug = suggest_regularizer(&moments, &config);

        assert!(sug.lambda_reg > 0.0, "lambda_reg = {}", sug.lambda_reg);
        assert!(sug.lambda_reg.is_finite(), "lambda_reg is NaN/Inf");
        assert!(
            (0.0..=1.0).contains(&sug.stability_score),
            "stability_score out of [0, 1]: {}",
            sug.stability_score
        );
        assert!(
            (0.0..=1.0).contains(&sug.outlier_fraction),
            "outlier_fraction out of [0, 1]: {}",
            sug.outlier_fraction
        );
    }

    #[test]
    fn test_suggest_regularizer_non_gaussian() {
        // Heavy-tailed distribution
        let moments = vec![0.0, 1.0, 0.0, 10.0, 0.0, 100.0];

        let config = TransformerConfig {
            n_layers: 6,
            hidden_dim: 512.0,
            n_samples: 5000.0,
            weight_std: 0.02,
            learning_rate: 0.001,
        };

        let sug = suggest_regularizer(&moments, &config);

        assert!(sug.lambda_reg > 0.0);
        assert!(sug.stability_score > 0.0 && sug.stability_score <= 1.0);
    }

    #[test]
    fn test_suggest_regularizer_empty_moments() {
        let moments = vec![];

        let config = TransformerConfig {
            n_layers: 6,
            hidden_dim: 512.0,
            n_samples: 5000.0,
            weight_std: 0.02,
            learning_rate: 0.001,
        };

        let sug = suggest_regularizer(&moments, &config);

        assert!((sug.lambda_reg - 1e-4).abs() < 1e-12);
        assert!((sug.stability_score - 1.0).abs() < 1e-12);
    }
}
