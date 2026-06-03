//! R-transform: the free additive convolution analogue.
//!
//! For freely independent random variables A, B:
//!
//! ```text
//! R_{A+B}(z) = R_A(z) + R_B(z)
//! ```
//!
//! where `R(z) = Σ_{n=1}^{N} κ_n · z^{n-1}` is defined from free cumulants.
//!
//! The Stieltjes transform `S(z)` and R-transform are related via
//! the fixed-point equation `S(z) = 1 / (z - R(S(z)))`, allowing
//! recovery of the spectral density from cumulants.

/// Evaluate the R-transform `R(z) = Σ κ_n · z^{n-1}` at a point `z`.
///
/// `cumulants[i]` = κ_{i+1}.  The series is truncated to the length of the slice.
pub fn from_cumulants(cumulants: &[f64], z: f64) -> f64 {
    let mut result = 0.0;
    let mut z_power = 1.0; // z^0
    for &k in cumulants {
        result += k * z_power;
        z_power *= z;
    }
    result
}

/// Add two cumulant sequences: `κ_sum[i] = κ_a[i] + κ_b[i]`.
///
/// All three slices must have at least `n` elements.
pub fn add_cumulants(kappa_a: &[f64], kappa_b: &[f64], kappa_sum: &mut [f64]) {
    let n = kappa_a.len().min(kappa_b.len()).min(kappa_sum.len());
    for i in 0..n {
        kappa_sum[i] = kappa_a[i] + kappa_b[i];
    }
}

/// Compute the Stieltjes transform from the R-transform via fixed-point iteration.
///
/// Solves `w = 1 / (z - R(w))` starting from `w₀ = 1/z`.
///
/// Returns the Stieltjes (Cauchy) transform `S(z)`.
pub fn stieltjes_from_r(cumulants: &[f64], z: f64, max_iter: usize, tol: f64) -> f64 {
    let mut w = 1.0 / z.abs().max(1e-15);

    for _ in 0..max_iter {
        let r_val = from_cumulants(cumulants, w);
        let w_new = 1.0 / (z - r_val);

        if (w_new - w).abs() < tol {
            w = w_new;
            break;
        }
        w = w_new;
    }

    w
}

/// Recover R(z) from a Stieltjes transform value.
///
/// Given `s = S(z)`, `R(s) = z - 1/s` (a single evaluation, not the inverse).
pub fn from_stieltjes(s: f64, z: f64) -> f64 {
    z - 1.0 / s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_r_transform_basic() {
        // Gaussian(0,1): κ = {0, 1, 0, 0} → R(z) = z
        let cumulants = vec![0.0, 1.0, 0.0, 0.0];

        assert!((from_cumulants(&cumulants, 0.0) - 0.0).abs() < 1e-12);
        assert!((from_cumulants(&cumulants, 1.0) - 1.0).abs() < 1e-12);
        assert!((from_cumulants(&cumulants, 2.0) - 2.0).abs() < 1e-12);
        assert!((from_cumulants(&cumulants, 0.5) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_r_transform_additivity_gaussian() {
        // A ~ Gaussian(1, 2): κ_A = {1, 2, 0, 0}
        // B ~ Gaussian(3, 4): κ_B = {3, 4, 0, 0}
        // A+B ~ Gaussian(4, 6): κ_{A+B} = {4, 6, 0, 0}
        let kappa_a = vec![1.0, 2.0, 0.0, 0.0];
        let kappa_b = vec![3.0, 4.0, 0.0, 0.0];
        let mut kappa_sum = vec![0.0_f64; 4];

        add_cumulants(&kappa_a, &kappa_b, &mut kappa_sum);

        assert!((kappa_sum[0] - 4.0).abs() < 1e-12);
        assert!((kappa_sum[1] - 6.0).abs() < 1e-12);
        assert!((kappa_sum[2] - 0.0).abs() < 1e-12);
        assert!((kappa_sum[3] - 0.0).abs() < 1e-12);

        // Verify R_{A+B}(z) = R_A(z) + R_B(z)
        let mut z = -1.0;
        while z <= 1.0 {
            let r_a = from_cumulants(&kappa_a, z);
            let r_b = from_cumulants(&kappa_b, z);
            let r_sum = from_cumulants(&kappa_sum, z);
            assert!((r_sum - (r_a + r_b)).abs() < 1e-12, "z={}", z);
            z += 0.5;
        }
    }

    #[test]
    fn test_r_transform_additivity_nonzero() {
        let kappa_a = vec![1.0, 2.0, 0.5, -0.3];
        let kappa_b = vec![2.0, 3.0, -0.5, 0.7];
        let mut kappa_sum = vec![0.0_f64; 4];

        add_cumulants(&kappa_a, &kappa_b, &mut kappa_sum);

        assert!((kappa_sum[0] - 3.0).abs() < 1e-12);
        assert!((kappa_sum[1] - 5.0).abs() < 1e-12);
        assert!((kappa_sum[2] - 0.0).abs() < 1e-12);
        assert!((kappa_sum[3] - 0.4).abs() < 1e-12);
    }

    #[test]
    fn test_stieltjes_roundtrip_semicircle() {
        // Semicircle(0, 1): R(z) = z
        // S(z) = (z - √(z²-4))/2 for real |z| >= 2
        let cumulants = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0];

        let z_test = 5.0;
        let s = stieltjes_from_r(&cumulants, z_test, 1000, 1e-12);

        let expected = (z_test - (z_test * z_test - 4.0).sqrt()) / 2.0;
        assert!((s - expected).abs() < 1e-6, "got {} expected {}", s, expected);
    }

    #[test]
    fn test_from_stieltjes() {
        let s = 0.3;
        let z = 5.0;
        let r = from_stieltjes(s, z);
        assert!((r - (z - 1.0 / s)).abs() < 1e-12);
    }
}
