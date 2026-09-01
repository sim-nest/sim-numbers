//! Non-destructive projections from domain-owned reports.

use sim_lib_numbers_stats::{KMeansRestartEvidence, KMeansTermination};
use sim_lib_numbers_tensor_linalg::DenseSolveReport;

use crate::{
    CriterionId, ErrorMeasure, ExecutionIdentity, MethodError, MethodEvidence, MethodId,
    PrecisionId, Termination, ToleranceSet, WorkLimit, WorkReceipt,
};

/// Adapter for [`DenseSolveReport`].
pub struct DenseSolveAdapter;
impl DenseSolveAdapter {
    /// Projects residual and bounded cubic work while retaining the source report.
    pub fn evidence(
        report: &DenseSolveReport,
        residual_tolerance: f64,
        execution: ExecutionIdentity,
    ) -> Result<MethodEvidence, MethodError> {
        let requested = ToleranceSet::new([ErrorMeasure::new(
            CriterionId::ResidualNorm,
            residual_tolerance,
        )?])?;
        let achieved = ErrorMeasure::new(CriterionId::ResidualNorm, report.residual_l2)?;
        let dimension =
            u64::try_from(report.dimension).map_err(|_| MethodError::InvalidWorkLimit)?;
        let charged = dimension
            .checked_mul(dimension)
            .and_then(|v| v.checked_mul(dimension))
            .ok_or(MethodError::InvalidWorkLimit)?
            .max(1);
        let limit = WorkLimit::new(charged)?;
        MethodEvidence::new(
            MethodId::new(MethodId::DENSE_SCALED_PIVOT)?,
            if achieved.value() <= residual_tolerance {
                Termination::Converged {
                    criterion: CriterionId::ResidualNorm,
                }
            } else {
                Termination::IterationLimit
            },
            WorkReceipt::new(charged, limit, true)?,
            requested,
            vec![achieved],
            PrecisionId::Binary64,
            execution,
        )
    }
}

/// Adapter for one domain-owned k-means restart report.
pub struct StatsKMeansAdapter;
impl StatsKMeansAdapter {
    /// Projects objective, termination, and charged work without replacing clustering evidence.
    pub fn evidence(
        report: &KMeansRestartEvidence,
        objective_tolerance: f64,
        work_limit: u64,
        execution: ExecutionIdentity,
    ) -> Result<MethodEvidence, MethodError> {
        let requested = ToleranceSet::new([ErrorMeasure::new(
            CriterionId::ObjectiveValue,
            objective_tolerance,
        )?])?;
        let achieved = ErrorMeasure::new(CriterionId::ObjectiveValue, report.inertia)?;
        let limit = WorkLimit::new(work_limit)?;
        let receipt = WorkReceipt::new(
            report.work,
            limit,
            report.termination == KMeansTermination::WorkLimit,
        )?;
        let termination = match report.termination {
            KMeansTermination::Converged if achieved.value() <= objective_tolerance => {
                Termination::Converged {
                    criterion: CriterionId::ObjectiveValue,
                }
            }
            KMeansTermination::Converged => Termination::IterationLimit,
            KMeansTermination::IterationLimit => Termination::IterationLimit,
            KMeansTermination::WorkLimit => Termination::WorkLimit,
        };
        MethodEvidence::new(
            MethodId::new(MethodId::KMEANS_LLOYD)?,
            termination,
            receipt,
            requested,
            vec![achieved],
            PrecisionId::Binary64,
            execution,
        )
    }
}
