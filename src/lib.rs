//! Free Probability for Rust.
//!
//! A port of [free-probability-c](https://github.com/SuperInstance/free-probability-c)
//! with corrections and enhancements.
//!
//! ## Why free probability?
//!
//! Xavier initialization. He initialization. Kaiming initialization.
//! They match Marchenko-Pastur. Free probability explains why.
//!
//! ## Modules
//!
//! - [`moments`] — Empirical moment computation, moment↔cumulant transforms
//! - [`r_transform`] — R-transform (free additive convolution)
//! - [`s_transform`] — S-transform (free multiplicative convolution)
//! - [`marchenko_pastur`] — Marchenko–Pastur density and moments
//! - [`prediction`] — Layer-combination prediction and gradient analysis

pub mod moments;
pub mod r_transform;
pub mod s_transform;
pub mod marchenko_pastur;
pub mod prediction;

/// Maximum supported moment/cumulant order.
pub const FP_MAX_ORDER: usize = 64;

/// An empirical distribution sampled at discrete points.
#[derive(Debug, Clone)]
pub struct EmpiricalDist<'a> {
    pub support_min: f64,
    pub support_max: f64,
    pub points: &'a [f64],
}
