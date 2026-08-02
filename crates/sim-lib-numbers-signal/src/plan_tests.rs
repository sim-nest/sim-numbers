use crate::{
    DctType, LengthPolicy, PaddingPolicy, SignConvention, SignalError, SpectrumPacking, Stride,
    TransformKind, TransformPlan,
};

#[test]
fn default_plan_fixes_every_convention() {
    let plan = TransformPlan::new(TransformKind::Fft, 8);
    plan.validate().unwrap();
    assert_eq!(plan.sign, SignConvention::NegativeForward);
    assert_eq!(plan.stride, Stride::contiguous());
    assert_eq!(plan.packing, SpectrumPacking::Full);
}

#[test]
fn stride_is_nonzero_and_overflow_checked() {
    assert_eq!(Stride::new(2, 0), Err(SignalError::ZeroStride));
    let stride = Stride::new(usize::MAX - 1, 2).unwrap();
    assert_eq!(stride.physical_index(1), Err(SignalError::StrideOverflow));
}

#[test]
fn length_and_padding_policies_cannot_contradict() {
    let mut plan = TransformPlan::new(TransformKind::Fft, 4);
    plan.length = LengthPolicy::Pad;
    assert!(matches!(
        plan.validate(),
        Err(SignalError::InvalidPolicy {
            policy: "padding",
            ..
        })
    ));
    plan.padding = PaddingPolicy::Zero;
    plan.validate().unwrap();
}

#[test]
fn definition_lengths_and_packing_fail_closed() {
    assert!(matches!(
        TransformPlan::new(TransformKind::Fft, 0).validate(),
        Err(SignalError::InvalidLength { .. })
    ));
    assert!(matches!(
        TransformPlan::new(TransformKind::Dct(DctType::I), 1).validate(),
        Err(SignalError::InvalidLength { .. })
    ));
    let mut plan = TransformPlan::new(TransformKind::Fft, 4);
    plan.packing = SpectrumPacking::HermitianHalf;
    assert!(matches!(
        plan.validate(),
        Err(SignalError::InvalidPolicy {
            policy: "packing",
            ..
        })
    ));
}
