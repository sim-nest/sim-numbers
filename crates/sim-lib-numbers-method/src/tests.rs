use sim_kernel::Datum;
use sim_kernel::Lib;
#[cfg(feature = "adapters")]
use sim_lib_numbers_signal::{EstimatorEvidence, EstimatorKind, FrequencyGridPolicy, SpectrumSide};
#[cfg(feature = "adapters")]
use sim_lib_numbers_stats::{KMeansRestartEvidence, KMeansTermination};
#[cfg(feature = "adapters")]
use sim_lib_numbers_tensor_linalg::{DenseSolveOptions, solve_dense_f64};

use crate::*;

fn execution() -> ExecutionIdentity {
    ExecutionIdentity::new("test-engine", "method-tests-v1", "case-0001").unwrap()
}

#[test]
fn work_charging_saturates_without_overflow() {
    let limit = WorkLimit::new(17).unwrap();
    let receipt = WorkReceipt::empty(limit).charge(u64::MAX).charge(1);
    assert_eq!(receipt.charged(), 17);
    assert!(receipt.saturated());
    assert!(WorkReceipt::new(18, limit, false).is_err());
}

#[test]
fn criteria_are_not_substitutable_and_success_cannot_be_forged() {
    let requested = ToleranceSet::new([
        ErrorMeasure::new(CriterionId::ResidualNorm, 1e-8).unwrap(),
        ErrorMeasure::new(CriterionId::StepNorm, 1e-6).unwrap(),
    ])
    .unwrap();
    let error = MethodEvidence::new(
        MethodId::new(MethodId::DENSE_SCALED_PIVOT).unwrap(),
        Termination::Converged {
            criterion: CriterionId::ResidualNorm,
        },
        WorkReceipt::empty(WorkLimit::new(10).unwrap()),
        requested,
        vec![ErrorMeasure::new(CriterionId::StepNorm, 0.0).unwrap()],
        PrecisionId::Binary64,
        execution(),
    )
    .unwrap_err();
    assert_eq!(error, MethodError::ContradictoryEvidence);
}

#[test]
fn every_record_has_deterministic_exact_datum_identity() {
    let requested =
        ToleranceSet::new([ErrorMeasure::new(CriterionId::ResidualNorm, -0.0).unwrap()]).unwrap();
    let plan = MethodPlan::new(
        MethodId::new(MethodId::DENSE_SCALED_PIVOT).unwrap(),
        WorkLimit::new(8).unwrap(),
        requested.clone(),
    );
    let plan_datum = plan.to_datum();
    assert_eq!(MethodPlan::from_datum(&plan_datum).unwrap(), plan);
    assert_eq!(
        plan_datum.content_id().unwrap(),
        plan.to_datum().content_id().unwrap()
    );
    assert_eq!(
        plan.tolerances().measures()[0].value().to_bits(),
        (-0.0f64).to_bits()
    );

    let work = WorkReceipt::empty(WorkLimit::new(8).unwrap()).charge(8);
    assert_eq!(WorkReceipt::from_datum(&work.to_datum()).unwrap(), work);
    let termination = Termination::Refused {
        reason: RefusalId::Unsupported,
    };
    assert_eq!(
        Termination::from_datum(&termination.to_datum()).unwrap(),
        termination
    );
    assert_eq!(
        ExecutionIdentity::from_datum(&execution().to_datum()).unwrap(),
        execution()
    );

    let mut forged = ErrorMeasure::new(CriterionId::ResidualNorm, 0.0)
        .unwrap()
        .to_datum();
    if let Datum::Node { fields, .. } = &mut forged {
        fields[1].1 = Datum::String("binary64:7ff8000000000001".into());
    }
    assert!(ErrorMeasure::from_datum(&forged).is_err());
}

#[test]
#[cfg(feature = "adapters")]
fn dense_adapter_preserves_domain_report_and_common_queries() {
    let solution = solve_dense_f64(
        &[2.0, 0.0, 0.0, 4.0],
        &[2.0, 8.0],
        DenseSolveOptions::default(),
    )
    .unwrap();
    let retained = solution.report;
    let common = DenseSolveAdapter::evidence(&solution.report, 1e-12, execution()).unwrap();
    assert_eq!(solution.report, retained);
    assert_eq!(common.method().as_str(), MethodId::DENSE_SCALED_PIVOT);
    assert!(matches!(
        common.termination(),
        Termination::Converged {
            criterion: CriterionId::ResidualNorm
        }
    ));
    assert_eq!(common.achieved()[0].value(), retained.residual_l2);
    assert_eq!(
        MethodEvidence::from_datum(&common.to_datum()).unwrap(),
        common
    );
}

#[test]
#[cfg(feature = "adapters")]
fn stats_adapter_preserves_domain_evidence() {
    let stats = KMeansRestartEvidence {
        restart: 2,
        seed: 41,
        inertia: 0.25,
        iterations: 7,
        converged: true,
        empty_cluster_repairs: 1,
        work: 90,
        termination: KMeansTermination::Converged,
    };
    let retained_stats = stats.clone();
    let common = StatsKMeansAdapter::evidence(&stats, 0.5, 100, execution()).unwrap();
    assert_eq!(stats, retained_stats);
    assert_eq!(common.work().charged(), 90);
    assert_eq!(common.achieved()[0].value(), 0.25);
}

#[test]
fn ids_limits_and_measures_reject_invalid_boundaries() {
    assert!(MethodId::new("invented/method").is_err());
    assert!(WorkLimit::new(0).is_err());
    assert!(ErrorMeasure::new(CriterionId::BracketWidth, f64::NAN).is_err());
    assert!(ErrorMeasure::new(CriterionId::EmbeddedLocalError, -1.0).is_err());
}

#[test]
fn loadable_surface_advertises_the_inspection_schema() {
    let manifest = MethodNumbersLib::new().manifest();
    assert_eq!(
        manifest.id,
        sim_kernel::Symbol::qualified("numbers", "method")
    );
    assert_eq!(manifest.exports.len(), 1);
}
// conformance: method tests prove bounded plans and criterion-specific retained evidence.
