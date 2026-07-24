use std::sync::Arc;

use sim_kernel::{Factory, Lib};
use sim_lib_numbers_tensor::{SpecTensor, Tensor, domains};

use super::{F32Tensor, F32TensorLib, tensor_spec_symbol};

#[test]
fn f32_roundtrip_preserves_cells() {
    let tensor = F32Tensor::new(vec![3], vec![1.0, -0.0, f32::INFINITY]).unwrap();
    let shifted = tensor.add_scalar(0.5);
    assert_eq!(shifted.as_slice()[0], 1.5);
    let roundtrip = F32Tensor::from_uniform(&shifted.to_uniform()).unwrap();
    assert_eq!(roundtrip.as_slice(), shifted.as_slice());
}

#[test]
fn uniform_roundtrip_preserves_f32_storage_identity() {
    let tensor = F32Tensor::new(vec![2], vec![1.0, 2.0]).unwrap();
    let uniform = tensor.to_uniform();
    assert!(Arc::ptr_eq(tensor.tensor.storage(), uniform.storage()));

    let roundtrip = F32Tensor::from_uniform(&uniform).unwrap();
    assert!(Arc::ptr_eq(roundtrip.tensor.storage(), uniform.storage()));
    assert_eq!(roundtrip.as_slice(), &[1.0, 2.0]);
}

#[test]
fn from_uniform_accepts_boxed_f32_cells() {
    let uniform =
        Tensor::new_exact(vec![2], domains::f32(), vec![number("1.5"), number("2.5")]).unwrap();
    let typed = F32Tensor::from_uniform(&uniform).unwrap();
    assert_eq!(typed.as_slice(), &[1.5, 2.5]);
}

#[test]
fn constructor_rejects_overflowing_shape() {
    assert!(F32Tensor::new(vec![usize::MAX, 2], Vec::new()).is_none());
}

#[test]
fn lib_exports_spec_tensor_descriptor() {
    assert_eq!(
        F32TensorLib::new().manifest().exports[0].symbol(),
        &tensor_spec_symbol()
    );
}

fn number(canonical: &str) -> sim_kernel::Value {
    sim_kernel::DefaultFactory
        .number_literal(domains::f32(), canonical.to_owned())
        .unwrap()
}
