use sim_kernel::Lib;
use sim_lib_numbers_tensor::SpecTensor;

use super::{F64Tensor, F64TensorLib, tensor_spec_symbol};

#[test]
fn specialized_add_scalar_beats_uniform_smoke() {
    let len = 512;
    let tensor = F64Tensor::new(vec![len], vec![1.0; len]).unwrap();
    let added = tensor.add_scalar(2.0);
    assert_eq!(added.data[0], 3.0);
    let (fast, slow) = tensor.smoke_speed_ratio(2.0);
    assert!(
        slow > fast,
        "expected specialized path to beat slow uniform path"
    );
    let roundtrip = F64Tensor::from_uniform(&added.to_uniform()).unwrap();
    assert_eq!(roundtrip, added);
}

#[test]
fn constructor_rejects_overflowing_shape() {
    assert!(F64Tensor::new(vec![usize::MAX, 2], Vec::new()).is_none());
}

#[test]
fn lib_exports_spec_tensor_descriptor() {
    assert_eq!(
        F64TensorLib::new().manifest().exports[0].symbol(),
        &tensor_spec_symbol()
    );
}
