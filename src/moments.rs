//! Moment computation and free cumulant ↔ moment transforms.
//!
//! The fundamental identity linking moments `m_n` and free cumulants `κ_n`:
//!
//! ```text
//! m_n = Σ_{π ∈ NC(n)} Π_{B ∈ π} κ_{|B|}
//! ```
//!
//! where `NC(n)` is the set of non-crossing partitions of `{1, …, n}`.
//! This module implements the recursive inversion/transform.

use crate::FP_MAX_ORDER;

/// Compute the k-th raw moment E[X^k] of an empirical distribution.
///
/// # Arguments
///
/// * `dist` — Empirical distribution with sample points
/// * `k` — Moment order (1 = mean, 2 = variance + mean², etc.)
pub fn compute_moment(dist: &crate::EmpiricalDist, k: usize) -> f64 {
    if dist.points.is_empty() {
        return 0.0;
    }
    let n = dist.points.len() as f64;
    let sum: f64 = dist.points.iter().map(|&x| x.powi(k as i32)).sum();
    sum / n
}

/// Compute moments up to order `max_k`.
///
/// `output[0]` receives m₁, `output[1]` receives m₂, …, `output[max_k-1]` receives m₍ₘₐₓ_ₖ₎.
pub fn compute_moments(dist: &crate::EmpiricalDist, max_k: usize, output: &mut [f64]) {
    for k in 1..=max_k {
        if k > output.len() {
            break;
        }
        output[k - 1] = compute_moment(dist, k);
    }
}

/// Convert a moment sequence to free cumulants (moment → cumulant).
///
/// `moments[i]` = m_{i+1}, `cumulants[i]` = κ_{i+1}.
///
/// Implements the recursive formula:
/// ```text
/// κ_n = m_n - Σ_{s=1}^{n-1} κ_s · [z^{n-s}] M̃(z)^s
/// ```
/// where `M̃(z) = 1 + m₁ z + m₂ z² + …`.
pub fn moment_to_cumulant(moments: &[f64], cumulants: &mut [f64]) {
    let n = moments.len().min(cumulants.len());
    if n == 0 {
        return;
    }

    // Extended moment array: ext[0] = m₀ = 1, ext[i] = m_i for i ≥ 1
    let mut ext = vec![0.0_f64; n + 1];
    ext[0] = 1.0;
    for i in 0..n {
        ext[i + 1] = moments[i];
    }

    let mut series = vec![0.0_f64; n + 1];    // current M̃(z)^s
    let mut new_series = vec![0.0_f64; n + 1];

    cumulants[0] = moments[0]; // κ₁ = m₁

    for nn in 2..=n {
        // series = M̃(z)^1
        for j in 0..nn {
            series[j] = ext[j];
        }

        let mut val = moments[nn - 1]; // m_nn

        for s in 1..=nn - 1 {
            let t_val = if nn >= s { series[nn - s] } else { 0.0 };
            val -= cumulants[s - 1] * t_val;

            // Update series to M̃(z)^{s+1}
            if s < nn - 1 {
                new_series.fill(0.0);
                for j in 0..nn {
                    for i in 0..=j {
                        new_series[j] += series[i] * ext[j - i];
                    }
                }
                series.copy_from_slice(&new_series);
            }
        }

        cumulants[nn - 1] = val;
    }
}

/// Convert free cumulants to moments (inverse transform).
///
/// `cumulants[i]` = κ_{i+1}, `moments[i]` = m_{i+1}.
pub fn cumulant_to_moment(cumulants: &[f64], moments: &mut [f64]) {
    let n = cumulants.len().min(moments.len());
    if n == 0 {
        return;
    }

    // Extended moment array
    let mut ext = vec![0.0_f64; n + 1];
    ext[0] = 1.0;

    let mut series = vec![0.0_f64; n + 1];
    let mut new_series = vec![0.0_f64; n + 1];

    for nn in 1..=n {
        if nn == 1 {
            moments[0] = cumulants[0];
            ext[1] = moments[0];
            continue;
        }

        // series = M̃(z)^1
        for j in 0..nn {
            series[j] = ext[j];
        }

        let mut val = cumulants[nn - 1]; // κ_nn

        for s in 1..=nn - 1 {
            let t_val = if nn >= s { series[nn - s] } else { 0.0 };
            val += cumulants[s - 1] * t_val;

            if s < nn - 1 {
                new_series.fill(0.0);
                for j in 0..nn {
                    for i in 0..=j {
                        new_series[j] += series[i] * ext[j - i];
                    }
                }
                series.copy_from_slice(&new_series);
            }
        }

        moments[nn - 1] = val;
        ext[nn] = val;
    }
}

/// Validate that a moment sequence is positive semi-definite (Hankel test).
///
/// Returns `true` if all leading principal minors of the Hankel moment matrix
/// are non-negative (allowing a tolerance of 1e-10).
pub fn validate_moments(moments: &[f64]) -> bool {
    let n = moments.len();
    if n == 0 {
        return true;
    }

    // Extended moment array
    let mut ext = vec![0.0_f64; n + 1];
    ext[0] = 1.0;
    for i in 0..n {
        ext[i + 1] = moments[i];
    }

    let ext_len = n + 1;
    let max_k = (ext_len / 2 + 1).min(FP_MAX_ORDER / 2);

    for k in 1..=max_k {
        if 2 * k - 2 >= ext_len {
            break;
        }

        // Build k×k Hankel matrix
        let size = k.min(FP_MAX_ORDER / 2);
        if size == 0 {
            continue;
        }

        let mut det = 1.0_f64;
        let mut mat = vec![vec![0.0_f64; size]; size];

        for i in 0..size {
            for j in 0..size {
                mat[i][j] = if i + j < ext_len { ext[i + j] } else { 0.0 };
            }
        }

        // Gaussian elimination (in-place, determinant tracking)
        for i in 0..size {
            let mut pivot = mat[i][i];
            if pivot.abs() < 1e-12 {
                let mut found = false;
                for j in (i + 1)..size {
                    if mat[j][i].abs() > 1e-12 {
                        mat.swap(i, j);
                        pivot = mat[i][i];
                        found = true;
                        break;
                    }
                }
                if !found {
                    det = 0.0;
                    break;
                }
            }
            det *= pivot;
            for j in (i + 1)..size {
                let factor = mat[j][i] / pivot;
                for l in i..size {
                    mat[j][l] -= factor * mat[i][l];
                }
            }
        }

        if det < -1e-10 {
            return false;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EmpiricalDist;

    #[test]
    fn test_uniform_moments() {
        let n = 10000_usize;
        let points: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5) / n as f64).collect();
        let dist = EmpiricalDist {
            support_min: 0.0,
            support_max: 1.0,
            points: &points,
        };

        let mut m = vec![0.0_f64; 6];
        compute_moments(&dist, 6, &mut m);

        assert!((m[0] - 0.5).abs() < 0.01, "m1 = {}", m[0]);
        assert!((m[1] - 1.0 / 3.0).abs() < 0.01, "m2 = {}", m[1]);
        assert!((m[2] - 0.25).abs() < 0.01, "m3 = {}", m[2]);
        assert!((m[3] - 0.2).abs() < 0.01, "m4 = {}", m[3]);
        assert!((m[4] - 1.0 / 6.0).abs() < 0.01, "m5 = {}", m[4]);
    }

    #[test]
    fn test_moment_cumulant_roundtrip() {
        let moments_in = vec![2.0, 7.0, 28.0, 127.0, 626.0];
        let n = moments_in.len();
        let mut cumulants = vec![0.0_f64; n];
        let mut moments_out = vec![0.0_f64; n];

        moment_to_cumulant(&moments_in, &mut cumulants);
        cumulant_to_moment(&cumulants, &mut moments_out);

        for i in 0..n {
            assert!(
                (moments_out[i] - moments_in[i]).abs() < 1e-8,
                "i={}: expected {}, got {}",
                i,
                moments_in[i],
                moments_out[i]
            );
        }
    }

    #[test]
    fn test_gaussian_moment_cumulant_roundtrip() {
        // Standard normal: m1=0, m2=1, m3=0, m4=3, m5=0, m6=15
        let moments = vec![0.0, 1.0, 0.0, 3.0, 0.0, 15.0];
        let n = moments.len();
        let mut cumulants = vec![0.0_f64; n];
        let mut moments_rt = vec![0.0_f64; n];

        moment_to_cumulant(&moments, &mut cumulants);
        cumulant_to_moment(&cumulants, &mut moments_rt);

        for i in 0..n {
            assert!(
                (moments_rt[i] - moments[i]).abs() < 1e-8,
                "i={}: expected {}, got {}",
                i,
                moments[i],
                moments_rt[i]
            );
        }
    }

    #[test]
    fn test_degenerate_point_mass() {
        // Point mass at c=3: m_k = 3^k
        let c = 3.0;
        let moments = vec![c, c * c, c * c * c, c * c * c * c];
        let n = moments.len();
        let mut cumulants = vec![0.0_f64; n];

        moment_to_cumulant(&moments, &mut cumulants);

        // Free cumulants of point mass: κ₁ = c, κ_n = 0 for n ≥ 2
        assert!((cumulants[0] - c).abs() < 1e-10, "κ1 = {}", cumulants[0]);
        assert!(
            cumulants[1].abs() < 1e-8,
            "κ2 = {} (should be 0)",
            cumulants[1]
        );
        assert!(
            cumulants[2].abs() < 1e-8,
            "κ3 = {} (should be 0)",
            cumulants[2]
        );
        assert!(
            cumulants[3].abs() < 1e-8,
            "κ4 = {} (should be 0)",
            cumulants[3]
        );
    }

    #[test]
    fn test_single_point_distribution() {
        let points = vec![5.0];
        let dist = EmpiricalDist {
            support_min: 5.0,
            support_max: 5.0,
            points: &points,
        };

        let mut m = vec![0.0_f64; 3];
        compute_moments(&dist, 3, &mut m);

        assert!((m[0] - 5.0).abs() < 1e-12);
        assert!((m[1] - 25.0).abs() < 1e-12);
        assert!((m[2] - 125.0).abs() < 1e-12);
    }

    #[test]
    fn test_two_point_distribution() {
        let points = vec![0.0, 2.0];
        let dist = EmpiricalDist {
            support_min: 0.0,
            support_max: 2.0,
            points: &points,
        };

        let mut m = vec![0.0_f64; 4];
        compute_moments(&dist, 4, &mut m);

        assert!((m[0] - 1.0).abs() < 1e-12);
        assert!((m[1] - 2.0).abs() < 1e-12);
        assert!((m[2] - 4.0).abs() < 1e-12);
        assert!((m[3] - 8.0).abs() < 1e-12);

        // Roundtrip
        let mut cumulants = vec![0.0_f64; 4];
        let mut m_rt = vec![0.0_f64; 4];
        moment_to_cumulant(&m, &mut cumulants);
        cumulant_to_moment(&cumulants, &mut m_rt);

        for i in 0..4 {
            assert!((m_rt[i] - m[i]).abs() < 1e-8, "i={}: {} vs {}", i, m_rt[i], m[i]);
        }
    }

    #[test]
    fn test_validate_valid_moments() {
        // Gaussian(0,1): 0, 1, 0, 3
        let moments = vec![0.0, 1.0, 0.0, 3.0];
        assert!(validate_moments(&moments), "valid Gaussian rejected");
    }

    #[test]
    fn test_validate_invalid_moments() {
        // Impossible: m₂ < m₁² → variance < 0
        let moments = vec![5.0, 1.0];
        assert!(!validate_moments(&moments), "invalid moments accepted");
    }

    #[test]
    fn test_validate_uniform_moments() {
        // Uniform[0,1]
        let moments = vec![0.5, 1.0 / 3.0, 0.25, 0.2];
        assert!(validate_moments(&moments), "valid uniform rejected");
    }

    #[test]
    fn test_large_order_roundtrip() {
        // Test with higher-order moments (order 10)
        let mut moments_in = Vec::new();
        for k in 1..=10 {
            moments_in.push(2.0_f64.powi(k as i32));
        }
        let n = moments_in.len();
        let mut cumulants = vec![0.0_f64; n];
        let mut moments_out = vec![0.0_f64; n];

        moment_to_cumulant(&moments_in, &mut cumulants);
        cumulant_to_moment(&cumulants, &mut moments_out);

        for i in 0..n {
            assert!(
                (moments_out[i] - moments_in[i]).abs() < 1e-6,
                "i={}: expected {}, got {}",
                i,
                moments_in[i],
                moments_out[i]
            );
        }
    }

    #[test]
    fn test_bernoulli_moments() {
        // Bernoulli(0.5): X ∈ {0, 1}, E[X] = 0.5, E[X²] = 0.5, ..., E[X^k] = 0.5
        let points: Vec<f64> = vec![0.0, 1.0];
        let dist = EmpiricalDist {
            support_min: 0.0,
            support_max: 1.0,
            points: &points,
        };

        let mut m = vec![0.0_f64; 5];
        compute_moments(&dist, 5, &mut m);

        for k in 0..5 {
            assert!((m[k] - 0.5).abs() < 1e-12, "m{} = {}", k + 1, m[k]);
        }
    }
}
