use sim_kernel::Lib;
use sim_lib_numbers_tensor::domains;

use super::{I64AddResult, I64Tensor, I64TensorLib, tensor_spec_symbol};

#[test]
fn overflow_widens_to_bigint_uniform_tensor() {
    let tensor = I64Tensor::new(vec![2], vec![i64::MAX, 3]).unwrap();
    let out = tensor.checked_add_scalar(1);
    match out {
        I64AddResult::Uniform(tensor) => {
            assert_eq!(tensor.dtype(), &domains::bigint());
            assert_eq!(tensor.shape(), &[2]);
        }
        I64AddResult::Specialized(_) => panic!("expected bigint widening"),
    }
}

#[test]
fn constructor_rejects_overflowing_shape() {
    assert!(I64Tensor::new(vec![usize::MAX, 2], Vec::new()).is_none());
}

#[test]
fn lib_exports_spec_tensor_descriptor() {
    assert_eq!(
        I64TensorLib::new().manifest().exports[0].symbol(),
        &tensor_spec_symbol()
    );
}
