#![forbid(unsafe_code)]
#![deny(missing_docs)]
//! Bounded, evidence-carrying dense matrix factorizations.
//!
//! [`qr_f64`] computes Householder QR, optionally with column pivoting.
//! [`symmetric_eigen_f64`] solves the real symmetric eigenproblem. Inputs are
//! borrowed and every successful result independently measures its certificate.

mod advanced;
pub use advanced::*;
mod matrix_functions;
pub use matrix_functions::*;

use sim_kernel::{
    AbiVersion, Export, Lib, LibManifest, LibTarget, Linker, Result as KernelResult, Symbol,
    Version,
};
use sim_lib_numbers_method::{
    CriterionId, ErrorMeasure, ExecutionIdentity, MethodEvidence, MethodId, PrecisionId,
    Termination, ToleranceSet, WorkLimit, WorkReceipt,
};
use std::{error::Error, fmt};

/// Whether an eigensolver materializes eigenvectors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VectorPolicy {
    /// Return values only.
    ValuesOnly,
    /// Return values and orthonormal vectors.
    Compute,
}

/// Admission and accuracy policy for QR.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QrPlan {
    /// Maximum admitted row or column count.
    pub max_dimension: usize,
    /// Maximum charged arithmetic work.
    pub max_work: u64,
    /// Relative diagonal threshold used for numerical rank.
    pub rank_threshold: f64,
    /// Select columns by remaining norm before each reflector.
    pub column_pivoting: bool,
    /// Maximum admitted reconstruction residual.
    pub reconstruction_tolerance: f64,
}
impl Default for QrPlan {
    fn default() -> Self {
        Self {
            max_dimension: 4096,
            max_work: u64::MAX,
            rank_threshold: 1e-12,
            column_pivoting: false,
            reconstruction_tolerance: 1e-10,
        }
    }
}

/// Admission and accuracy policy for a symmetric eigenproblem.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EigenPlan {
    /// Maximum admitted matrix dimension.
    pub max_dimension: usize,
    /// Maximum shifted iterations across all eigenvalues.
    pub max_iterations: usize,
    /// Relative convergence tolerance.
    pub tolerance: f64,
    /// Explicit maximum asymmetry. `None` requires bitwise symmetry.
    pub symmetry_tolerance: Option<f64>,
    /// Eigenvector materialization policy.
    pub vectors: VectorPolicy,
    /// Maximum admitted eigenpair residual.
    pub reconstruction_tolerance: f64,
}
impl Default for EigenPlan {
    fn default() -> Self {
        Self {
            max_dimension: 4096,
            max_iterations: 256,
            tolerance: 1e-12,
            symmetry_tolerance: None,
            vectors: VectorPolicy::Compute,
            reconstruction_tolerance: 1e-10,
        }
    }
}

/// Factor-specific QR certificate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QrEvidence {
    /// Frobenius norm of `A P - Q R`.
    pub reconstruction_residual: f64,
    /// Frobenius norm of `Q^T Q - I`.
    pub orthogonality_residual: f64,
}
/// Householder QR result, with row-major factors.
#[derive(Clone, Debug, PartialEq)]
pub struct QrFactorization {
    /// Row-major thin Q (`rows * min(rows, cols)`).
    pub q: Vec<f64>,
    /// Row-major R (`min(rows, cols) * cols`).
    pub r: Vec<f64>,
    /// Original-column index at each factorized column.
    pub permutation: Vec<usize>,
    /// Numerical rank under the plan threshold.
    pub rank: usize,
    /// Common bounded-method evidence.
    pub method: MethodEvidence,
    /// Factor-specific certificate.
    pub evidence: QrEvidence,
}
/// Factor-specific symmetric eigen certificate.
#[derive(Clone, Debug, PartialEq)]
pub struct EigenEvidence {
    /// Euclidean residual for each returned pair.
    pub pair_residuals: Vec<f64>,
    /// Frobenius norm of `V^T V - I`, or zero without vectors.
    pub orthogonality_residual: f64,
    /// Frobenius norm of `A V - V D`, or zero without vectors.
    pub reconstruction_residual: f64,
}
/// Ordered symmetric eigendecomposition.
#[derive(Clone, Debug, PartialEq)]
pub struct SymmetricEigen {
    /// Eigenvalues in descending order.
    pub eigenvalues: Vec<f64>,
    /// Row-major eigenvectors stored by columns when requested.
    pub eigenvectors: Option<Vec<f64>>,
    /// Common bounded-method evidence.
    pub method: MethodEvidence,
    /// Factor-specific certificate.
    pub evidence: EigenEvidence,
}

/// Rejection or bounded non-convergence from a factorization.
#[derive(Clone, Debug, PartialEq)]
pub enum DecompositionError {
    /// Matrix dimensions are empty, overflow, or do not match storage.
    InvalidDimensions,
    /// A plan parameter is invalid.
    InvalidPlan(&'static str),
    /// Input contains NaN or infinity.
    NonFinite {
        /// Flat input index.
        index: usize,
    },
    /// Dimension or projected work exceeds the plan.
    WorkLimit,
    /// Symmetry was not established under the caller's policy.
    Asymmetric {
        /// Largest absolute mismatch.
        mismatch: f64,
        /// Admitted tolerance.
        tolerance: f64,
    },
    /// Shifted iteration did not converge within its bound.
    NoConvergence,
    /// A computed certificate exceeded the plan tolerance.
    Reconstruction {
        /// Observed residual.
        residual: f64,
        /// Required maximum.
        tolerance: f64,
    },
    /// Common evidence construction failed.
    Evidence,
}
impl fmt::Display for DecompositionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::InvalidDimensions => "invalid matrix dimensions",
                Self::InvalidPlan(s) => s,
                Self::NonFinite { .. } => "matrix contains a non-finite value",
                Self::WorkLimit => "factorization exceeds its admitted work or dimension limit",
                Self::Asymmetric { .. } => "matrix is not symmetric under the explicit tolerance",
                Self::NoConvergence => "symmetric shifted iteration did not converge",
                Self::Reconstruction { .. } => "factorization failed its reconstruction check",
                Self::Evidence => "factorization evidence was internally inconsistent",
            }
        )
    }
}
impl Error for DecompositionError {}

include!("basic.rs");
/// Cookbook recipes embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

/// Loadable runtime surface advertising the dense decomposition contracts.
#[derive(Default)]
pub struct TensorDecompLib;
impl TensorDecompLib {
    /// Constructs the stateless decomposition library.
    pub fn new() -> Self {
        Self
    }
}
impl Lib for TensorDecompLib {
    fn manifest(&self) -> LibManifest {
        LibManifest {
            id: Symbol::qualified("numbers", "tensor-decomp"),
            version: Version(env!("CARGO_PKG_VERSION").to_owned()),
            abi: AbiVersion { major: 0, minor: 1 },
            target: LibTarget::HostRegistered,
            requires: Vec::new(),
            capabilities: Vec::new(),
            exports: vec![Export::Value {
                symbol: decomposition_schema_symbol(),
            }],
        }
    }
    fn load(&self, cx: &mut sim_kernel::LoadCx, linker: &mut Linker<'_>) -> KernelResult<()> {
        linker.value(decomposition_schema_symbol(), cx.factory().string("qr symmetric-eigen real-schur svd rank condition pseudoinverse least-squares null-space matrix-power matrix-exponential scaling-squaring pade sylvester lyapunov separation residual provider-admission plans factors permutation residual orthogonality method-evidence".to_owned())?)
    }
}
/// Symbol of the runtime decomposition inspection schema.
pub fn decomposition_schema_symbol() -> Symbol {
    Symbol::qualified("numbers/tensor-decomp", "schema")
}

#[cfg(test)]
mod tests;
