use std::sync::Arc;

use sim_kernel::Lib;
use sim_lib_numbers_tensor::SpecTensor;

use super::{Rat64Tensor, Rat64TensorLib, tensor_spec_symbol};

#[test]
fn rationals_are_normalized() {
    let tensor = Rat64Tensor::new(vec![2], vec![(2, 4), (-6, -8)]).unwrap();
    assert_eq!(tensor.to_uniform().cells().unwrap().len(), 2);
    let roundtrip = Rat64Tensor::from_uniform(&tensor.to_uniform()).unwrap();
    assert_eq!(roundtrip.as_slice(), &[(1, 2), (3, 4)]);
}

#[test]
fn uniform_roundtrip_preserves_rational_storage_identity() {
    let tensor = Rat64Tensor::new(vec![2], vec![(2, 4), (-6, -8)]).unwrap();
    let uniform = tensor.to_uniform();
    assert!(Arc::ptr_eq(tensor.tensor.storage(), uniform.storage()));

    let roundtrip = Rat64Tensor::from_uniform(&uniform).unwrap();
    assert!(Arc::ptr_eq(roundtrip.tensor.storage(), uniform.storage()));
    assert_eq!(roundtrip.as_slice(), &[(1, 2), (3, 4)]);
}

#[test]
fn constructor_rejects_overflowing_shape() {
    assert!(Rat64Tensor::new(vec![usize::MAX, 2], Vec::new()).is_none());
}

#[test]
fn lib_exports_spec_tensor_descriptor() {
    assert_eq!(
        Rat64TensorLib::new().manifest().exports[0].symbol(),
        &tensor_spec_symbol()
    );
}
