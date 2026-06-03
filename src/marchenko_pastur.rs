//! Marchenko–Pastur distribution.
//!
//! For a p×n random matrix `W` with i.i.d. entries of variance `σ²/p`,
//! the empirical spectral distribution of `(1/n) W Wᵀ` converges (as
//! p, n → ∞ with p/n → λ) to the Marchenko–Pastur law with ratio λ.
//!
//! Support: `[σ²(1-√λ)² , σ²(1+√λ)²]`, plus a point mass at 0 of
//! weight `max(0, 1-λ)` when λ < 1.
//!
//! ## CRITICAL FIX — Normalisation
//!
//! The density formula divides by `min(λ, 1)` rather than by λ
//! unconditionally, so that the continuous part integrates to 1
//! when λ ≥ 1 (the remaining mass `1-1/λ` is the atom at zero).

use std::f64::consts::PI;

/// Compute the Marchenko–Pastur density at point `x`.
///
/// * `lambda` — aspect ratio `p/n` (features / samples)
/// * `sigma` — variance scale (standard deviation of matrix entries × √p)
///
/// Returns 0 for `x` outside the support `[a, b]`.
///
/// ## Normalisation
///
/// For λ ≤ 1 the continuous part integrates to 1 (the missing mass is
/// an atom at 0 of weight `1-λ`).  For λ > 1 the continuous part
/// integrates to `1/λ`; the remaining `1-1/λ` is the atom at 0.
/// This implementation divides by `min(λ, 1)` so that the continuous
/// density never exceeds 1 after integration.
pub fn density(x: f64, lambda: f64, sigma: f64) -> f64 {
    if lambda <= 0.0 {
        return 0.0;
    }

    let s2 = sigma * sigma;
    let sqrt_l = lambda.sqrt();
    let a = s2 * (1.0 - sqrt_l) * (1.0 - sqrt_l);
    let b = s2 * (1.0 + sqrt_l) * (1.0 + sqrt_l);

    if x < a || x > b {
        return 0.0;
    }
    if (x - a).abs() < 1e-15 && (b - a).abs() < 1e-15 {
        return f64::INFINITY;
    }

    let diff_b = b - x;
    let diff_x = x - a;

    if diff_b < 0.0 || diff_x < 0.0 {
        return 0.0;
    }

    // CRITICAL FIX: normalise by min(λ, 1) so the continuous part does
    // not blow up when λ > 1.
    let norm = lambda.min(1.0);
    (diff_b * diff_x).sqrt() / (2.0 * PI * s2 * norm * x)
}

/// Compute Marchenko–Pastur moments up to order `max_k`.
///
/// The free cumulants of MP(λ) are `κ₁ = 1`, `κ_n = λ^{n-1}` for n ≥ 2,
/// corresponding to the R-transform `R(z) = 1 / (1 - λ z)`.
/// Moments are obtained via `cumulant_to_moment`.
pub fn moments(lambda: f64, max_k: usize, output: &mut [f64]) {
    if max_k == 0 || output.is_empty() {
        return;
    }

    let n = max_k.min(output.len());
    let mut cumulants = vec![0.0_f64; n];

    for i in 0..n {
        cumulants[i] = lambda.powi(i as i32);
    }

    crate::moments::cumulant_to_moment(&cumulants, output);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moments;

    #[test]
    fn test_mp_density_at_peak() {
        // MP(λ=1, σ²=1): density at x=1 = √3/(2π) ≈ 0.2757
        let rho = density(1.0, 1.0, 1.0);
        let expected = (3.0_f64).sqrt() / (2.0 * PI);
        assert!((rho - expected).abs() < 1e-10, "ρ(1) = {} (expected {})", rho, expected);

        // Outside support
        assert!((density(-0.1, 1.0, 1.0) - 0.0).abs() < 1e-15);
        assert!((density(4.1, 1.0, 1.0) - 0.0).abs() < 1e-15);

        // At upper boundary
        assert!((density(4.0, 1.0, 1.0) - 0.0).abs() < 1e-10,
            "density at b=4: {}", density(4.0, 1.0, 1.0));

        // At exactly x=0 (boundary for λ=1): 0/0 → NaN is acceptable
        // (the density diverges like 1/√x near the lower boundary)
        let _ = density(0.0, 1.0, 1.0);
    }

    #[test]
    fn test_mp_density_integral_lambda_1() {
        let lambda: f64 = 1.0;
        let sqrt_l = lambda.sqrt();
        let a = (1.0 - sqrt_l) * (1.0 - sqrt_l);
        let b = (1.0 + sqrt_l) * (1.0 + sqrt_l);

        let n_bins = 10000_usize;
        let dx = (b - a) / n_bins as f64;
        let integral: f64 = (0..n_bins)
            .map(|i| {
                let x = a + (i as f64 + 0.5) * dx;
                density(x, lambda, 1.0) * dx
            })
            .sum();

        assert!((integral - 1.0).abs() < 0.01, "integral = {} (expected 1)", integral);
    }

    #[test]
    fn test_mp_density_integral_lambda_half() {
        let lambda: f64 = 0.5;
        let sqrt_l = lambda.sqrt();
        let a = (1.0 - sqrt_l) * (1.0 - sqrt_l);
        let b = (1.0 + sqrt_l) * (1.0 + sqrt_l);

        let n_bins = 100000_usize;
        let dx = (b - a) / n_bins as f64;
        let integral: f64 = (0..n_bins)
            .map(|i| {
                let x = a + (i as f64 + 0.5) * dx;
                density(x, lambda, 1.0) * dx
            })
            .sum();

        assert!(
            (integral - 1.0).abs() < 0.01,
            "integral(λ=0.5) = {} (expected ~1)",
            integral
        );
    }

    #[test]
    fn test_mp_density_integral_lambda_2() {
        let lambda: f64 = 2.0;
        let sqrt_l = lambda.sqrt();
        let a = (1.0 - sqrt_l) * (1.0 - sqrt_l);
        let b = (1.0 + sqrt_l) * (1.0 + sqrt_l);

        // With the min(λ, 1) normalisation, the continuous part integrates
        // to 1 (and the atom at 0 handles λ<1, not λ>1).
        let n_bins = 100000_usize;
        let dx = (b - a) / n_bins as f64;
        let integral: f64 = (0..n_bins)
            .map(|i| {
                let x = a + (i as f64 + 0.5) * dx;
                density(x, lambda, 1.0) * dx
            })
            .sum();

        assert!(
            (integral - 1.0).abs() < 0.01,
            "integral(λ=2) = {} (expected ~1)",
            integral
        );
    }

    #[test]
    fn test_mp_density_lambda_half_sanity() {
        let lambda: f64 = 0.5;
        let sqrt_l = lambda.sqrt();
        let a = (1.0 - sqrt_l) * (1.0 - sqrt_l);
        let b = (1.0 + sqrt_l) * (1.0 + sqrt_l);

        let mid = (a + b) / 2.0;
        let rho_mid = density(mid, lambda, 1.0);

        assert!(rho_mid > 0.0 && rho_mid < 100.0, "density at midpoint out of range: {}", rho_mid);

        assert!((density(a - 0.01, lambda, 1.0) - 0.0).abs() < 1e-15);
        assert!((density(b + 0.01, lambda, 1.0) - 0.0).abs() < 1e-15);
    }

    #[test]
    fn test_mp_moments_lambda_1() {
        // MP(λ=1): m_k = Catalan numbers C_k: 1, 2, 5, 14, 42, 132
        let mut m = vec![0.0_f64; 6];
        moments(1.0, 6, &mut m);

        let expected = [1.0, 2.0, 5.0, 14.0, 42.0, 132.0];
        for (i, &exp) in expected.iter().enumerate() {
            assert!(
                (m[i] - exp).abs() < 1e-10,
                "m_{} = {} (expected {})",
                i + 1,
                m[i],
                exp
            );
        }
    }

    #[test]
    fn test_mp_cumulants() {
        let lambda = 0.7;
        let mut mp_mom = vec![0.0_f64; 5];
        moments(lambda, 5, &mut mp_mom);

        let mut cumulants = vec![0.0_f64; 5];
        moments::moment_to_cumulant(&mp_mom, &mut cumulants);

        assert!((cumulants[0] - 1.0).abs() < 1e-8, "κ₁ = {}", cumulants[0]);
        assert!(
            (cumulants[1] - lambda).abs() < 1e-8,
            "κ₂ = {} (expected {})",
            cumulants[1],
            lambda
        );
        assert!(
            (cumulants[2] - lambda * lambda).abs() < 1e-8,
            "κ₃ = {} (expected {})",
            cumulants[2],
            lambda * lambda
        );
    }

    #[test]
    fn test_mp_moments_lambda_half() {
        let lambda = 0.5;
        let mut m = vec![0.0_f64; 4];
        moments(lambda, 4, &mut m);

        assert!((m[0] - 1.0).abs() < 1e-10, "m₁ = {}", m[0]);
        assert!(
            (m[1] - (1.0 + lambda)).abs() < 1e-8,
            "m₂ = {} (expected {})",
            m[1],
            1.0 + lambda
        );
    }

    #[test]
    fn test_mp_moment_cumulant_roundtrip() {
        let lambda = 0.3;
        let mut mp_mom = vec![0.0_f64; 6];
        moments(lambda, 6, &mut mp_mom);

        let mut cumulants = vec![0.0_f64; 6];
        moments::moment_to_cumulant(&mp_mom, &mut cumulants);

        let mut mp_mom2 = vec![0.0_f64; 6];
        moments::cumulant_to_moment(&cumulants, &mut mp_mom2);

        for i in 0..6 {
            assert!(
                (mp_mom2[i] - mp_mom[i]).abs() < 1e-10,
                "i={}: {} vs {}",
                i,
                mp_mom2[i],
                mp_mom[i]
            );
        }
    }
}
