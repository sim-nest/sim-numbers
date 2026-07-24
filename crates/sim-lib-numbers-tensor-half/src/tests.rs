use std::sync::Arc;

use half::{bf16, f16};
use sim_kernel::Lib;
use sim_lib_numbers_tensor::{SpecTensor, domains, parse_f32_literal_cell};

use super::{
    Bf16Tensor, F16Tensor, HalfTensorLib, bf16_tensor_spec_symbol, f16_tensor_spec_symbol,
};

#[test]
fn f16_roundtrip_preserves_cells_and_storage_identity() {
    let tensor = F16Tensor::new(
        vec![3],
        vec![f16::from_f32(1.0), f16::NEG_ZERO, f16::INFINITY],
    )
    .unwrap();
    let uniform = tensor.to_uniform();
    assert!(Arc::ptr_eq(tensor.tensor.storage(), uniform.storage()));

    let roundtrip = F16Tensor::from_uniform(&uniform).unwrap();
    assert!(Arc::ptr_eq(roundtrip.tensor.storage(), uniform.storage()));
    assert_eq!(roundtrip.as_slice(), tensor.as_slice());
    assert!(roundtrip.as_slice()[1].is_sign_negative());
}

#[test]
fn bf16_roundtrip_preserves_cells_and_storage_identity() {
    let tensor = Bf16Tensor::new(vec![2], vec![bf16::from_f32(1.5), bf16::from_f32(-2.0)]).unwrap();
    let uniform = tensor.to_uniform();
    assert!(Arc::ptr_eq(tensor.tensor.storage(), uniform.storage()));

    let roundtrip = Bf16Tensor::from_uniform(&uniform).unwrap();
    assert!(Arc::ptr_eq(roundtrip.tensor.storage(), uniform.storage()));
    assert_eq!(roundtrip.as_slice(), tensor.as_slice());
}

#[test]
fn half_cpu_arithmetic_widens_to_f32() {
    let tensor = F16Tensor::new(
        vec![3],
        vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)],
    )
    .unwrap();
    let widened = tensor.add_f32_scalar(0.5);
    assert_eq!(widened.dtype(), &domains::f32());
    let cells = widened.cells().unwrap();
    assert_eq!(
        cells
            .iter()
            .map(parse_f32_literal_cell)
            .collect::<Option<Vec<_>>>()
            .unwrap(),
        vec![1.5, 2.5, 3.5]
    );
    assert_eq!(tensor.sum_f32(), 6.0);
}

#[test]
fn bf16_cpu_arithmetic_widens_to_f32() {
    let tensor = Bf16Tensor::new(vec![2], vec![bf16::from_f32(1.0), bf16::from_f32(2.0)]).unwrap();
    let widened = tensor.to_f32_uniform();
    assert_eq!(widened.dtype(), &domains::f32());
    assert_eq!(tensor.sum_f32(), 3.0);
}

#[test]
fn constructors_reject_overflowing_shapes() {
    assert!(F16Tensor::new(vec![usize::MAX, 2], Vec::new()).is_none());
    assert!(Bf16Tensor::new(vec![usize::MAX, 2], Vec::new()).is_none());
}

#[test]
fn lib_exports_half_spec_tensor_descriptors() {
    let exports = HalfTensorLib::new()
        .manifest()
        .exports
        .into_iter()
        .map(|export| export.symbol().clone())
        .collect::<Vec<_>>();
    assert_eq!(
        exports,
        vec![f16_tensor_spec_symbol(), bf16_tensor_spec_symbol()]
    );
}
