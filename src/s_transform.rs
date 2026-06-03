//! S-transform: the free multiplicative convolution analogue.
//!
//! For freely independent A, B:
//!
//! ```text
//! S_{AB}(z) = S_A(z) · S_B(z)
//! ```
//!
//! The S-transform is defined via the moment generating function.
//! Let `ψ(z) = Σ_{n≥1} m_n z^n` and let `χ` be its compositional inverse
//! (i.e., `ψ(χ(z)) = z`). Then `S(z) = (1+z) · χ(z)/z`.
//!
//! ## CRITICAL FIX (vs earlier C implementations)
//!
//! The series-reversion convolution for `χ^k` must index `chi_pow_new`
//! at `[a + b + 1]` (not `[a + b]`) because each χ factor starts at z¹
//! in the 0-indexed array representation.  This fix corrects the
//! S-transform for non-trivial distributions.

use std::f64;

/// Compute S-transform coefficients from moments via series reversion.
///
/// Given moments `m₁, m₂, …, mₙ`, produces `n` S-transform coefficients
/// `s₀, s₁, …, s_{n-1}` such that `S(z) = Σ s_k z^k`.
///
/// Internally:
/// 1. Build `ψ(z) = Σ m_k z^k`
/// 2. Compute compositional inverse `χ(z) = ψ^{-1}(z)` via series reversion
/// 3. `S(z) = (1+z) · χ(z)/z`
pub fn from_moments(moments: &[f64], s_coeffs: &mut [f64]) {
    let n = moments.len().min(s_coeffs.len());
    if n == 0 {
        return;
    }

    // ψ coefficients: psi[k] = m_{k+1} = moments[k]
    let mut psi = vec![0.0_f64; n];
    psi.copy_from_slice(&moments[..n]);

    // χ will hold the compositional inverse coefficients
    let mut chi = vec![0.0_f64; n + 1];

    if psi[0].abs() < 1e-15 {
        s_coeffs.fill(0.0);
        return;
    }

    let a1 = psi[0];
    chi[0] = 1.0 / a1; // b₁ = 1/a₁

    // chi_pow_prev starts as χ¹
    let mut chi_pow_prev = vec![0.0_f64; n + 1];
    let mut chi_pow_new = vec![0.0_f64; n + 1];

    for j in 2..=n {
        // Reset chi_pow_prev to χ¹ for the inner loop
        chi_pow_prev[..n].copy_from_slice(&chi[..n]);

        let mut rhs = 0.0;

        // Build χ^k for k = 2, 3, ..., j and accumulate RHS
        for k in 2..=j {
            // χ^k = χ^{k-1} · χ
            chi_pow_new.fill(0.0);
            // CRITICAL FIX: use a+b+1 (not a+b) because χ starts at z¹
            for a in 0..j {
                if chi_pow_prev[a] == 0.0 {
                    continue;
                }
                for b in 0..(j - a) {
                    chi_pow_new[a + b + 1] += chi_pow_prev[a] * chi[b];
                }
            }
            rhs += psi[k - 1] * chi_pow_new[j - 1];
            chi_pow_prev[..(n + 1)].copy_from_slice(&chi_pow_new[..(n + 1)]);
        }

        chi[j - 1] = -rhs / a1;
    }

    // S(z) = (1+z) · χ(z)/z
    // χ(z)/z = Σ chi[k] z^k  (regular power series)
    // S(z) = chi[0] + Σ_{k≥1} (chi[k] + chi[k-1]) z^k
    s_coeffs[0] = chi[0];
    for i in 1..n {
        s_coeffs[i] = chi[i] + chi[i - 1];
    }
}

/// Multiply two S-transform coefficient sequences (polynomial multiplication).
///
/// `sa`, `sb` each have `n` coefficients. `product` receives `2n-1` coefficients.
pub fn multiply(sa: &[f64], sb: &[f64], n: usize, product: &mut [f64]) {
    let result_len = (2 * n - 1).min(product.len());
    product[..result_len].fill(0.0);

    for i in 0..n {
        for j in 0..n {
            if i + j < result_len {
                product[i + j] += sa[i] * sb[j];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::marchenko_pastur;
    #[test]
    fn test_s_transform_mp1_roundtrip() {
        let mut mp_moments = vec![0.0_f64; 5];
        marchenko_pastur::moments(1.0, 5, &mut mp_moments);

        let mut s_coeffs = vec![0.0_f64; 5];
        from_moments(&mp_moments, &mut s_coeffs);

        // s₀ should be positive and finite
        assert!(
            s_coeffs[0] > 0.0,
            "s₀ = {} (should be >0)",
            s_coeffs[0]
        );
        assert!(s_coeffs[0].is_finite(), "s₀ is NaN/Inf");
    }

    #[test]
    fn test_s_transform_mp1_exact() {
        // MP(1) has S-transform = 1/(1+z) = 1 - z + z² - z³ + ...
        let mut mp_moments = vec![0.0_f64; 8];
        marchenko_pastur::moments(1.0, 8, &mut mp_moments);

        let mut s_coeffs = vec![0.0_f64; 8];
        from_moments(&mp_moments, &mut s_coeffs);

        for i in 0..8 {
            let expected = (-1.0_f64).powi(i as i32);
            assert!(
                (s_coeffs[i] - expected).abs() < 1e-6,
                "s[{}] = {} (expected {})",
                i,
                s_coeffs[i],
                expected
            );
        }

        // Evaluate S(0.5) = 1/1.5 ≈ 0.6667
        let z = 0.5;
        let mut s_val = 0.0;
        let mut zpow = 1.0;
        for i in 0..8 {
            s_val += s_coeffs[i] * zpow;
            zpow *= z;
        }
        assert!(
            (s_val - 1.0 / (1.0 + z)).abs() < 5e-3,
            "S({}) = {} (expected {})",
            z,
            s_val,
            1.0 / (1.0 + z)
        );
    }

    #[test]
    fn test_s_transform_nontrivial() {
        // MP(0.7) should have non-trivial S-transform coefficients
        let mut mp_moments = vec![0.0_f64; 6];
        marchenko_pastur::moments(0.7, 6, &mut mp_moments);

        let mut s_coeffs = vec![0.0_f64; 6];
        from_moments(&mp_moments, &mut s_coeffs);

        // s₀ should be 1 (since m₁ = 1 for MP)
        assert!((s_coeffs[0] - 1.0).abs() < 1e-8, "s₀ = {}", s_coeffs[0]);

        // At least 3 non-trivial coefficients beyond s₀
        let nontrivial: usize = s_coeffs[1..]
            .iter()
            .filter(|&&c| c.abs() > 1e-10)
            .count();
        assert!(
            nontrivial >= 3,
            "S-transform is near-constant (only {} nontrivial coeffs)",
            nontrivial
        );
    }

    #[test]
    fn test_s_transform_multiply() {
        let sa = vec![1.0, 2.0, 3.0];
        let sb = vec![4.0, 5.0, 6.0];
        let mut product = vec![0.0_f64; 5];

        multiply(&sa, &sb, 3, &mut product);

        // (1+2z+3z²)(4+5z+6z²) = 4 + 13z + 28z² + 27z³ + 18z⁴
        assert!((product[0] - 4.0).abs() < 1e-12);
        assert!((product[1] - 13.0).abs() < 1e-12);
        assert!((product[2] - 28.0).abs() < 1e-12);
        assert!((product[3] - 27.0).abs() < 1e-12);
        assert!((product[4] - 18.0).abs() < 1e-12);
    }

    #[test]
    fn test_s_transform_mp1_multiplicative_identity() {
        // MP(1) has S(z) = 1/(1+z)
        // MP(1) * MP(1) should have S(z) = 1/(1+z)²
        let mut mp_moments = vec![0.0_f64; 6];
        marchenko_pastur::moments(1.0, 6, &mut mp_moments);

        let mut s1 = vec![0.0_f64; 6];
        from_moments(&mp_moments, &mut s1);

        let mut product = vec![0.0_f64; 11];
        multiply(&s1, &s1, 6, &mut product);

        // product should approximate 1/(1+z)² = 1 - 2z + 3z² - 4z³ + ...
        assert!((product[0] - 1.0).abs() < 1e-6, "p[0] = {}", product[0]);
        assert!((product[1] - (-2.0)).abs() < 2e-6, "p[1] = {}", product[1]);
        assert!((product[2] - 3.0).abs() < 5e-6, "p[2] = {}", product[2]);
    }
}
