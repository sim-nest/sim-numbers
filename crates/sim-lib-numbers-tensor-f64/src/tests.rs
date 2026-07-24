use std::any::Any;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use sim_kernel::Lib;
use sim_kernel::{DefaultFactory, Factory, Result, Symbol, Value};
use sim_lib_numbers_tensor::{SpecTensor, Tensor, TensorLocation, TensorStorage, domains};

use super::{F64Tensor, F64TensorLib, tensor_spec_symbol};

#[test]
fn specialized_add_scalar_beats_uniform_smoke() {
    let len = 512;
    let tensor = F64Tensor::new(vec![len], vec![1.0; len]).unwrap();
    let added = tensor.add_scalar(2.0);
    assert_eq!(added.as_slice()[0], 3.0);
    let (fast, slow) = tensor.smoke_speed_ratio(2.0);
    assert!(
        slow > fast,
        "expected specialized path to beat slow uniform path"
    );
    let roundtrip = F64Tensor::from_uniform(&added.to_uniform()).unwrap();
    assert_eq!(roundtrip, added);
}

#[test]
fn uniform_roundtrip_preserves_f64_storage_identity() {
    let tensor = F64Tensor::new(vec![2], vec![1.0, 2.0]).unwrap();
    let uniform = tensor.to_uniform();
    assert!(Arc::ptr_eq(tensor.tensor.storage(), uniform.storage()));

    let roundtrip = F64Tensor::from_uniform(&uniform).unwrap();
    assert!(Arc::ptr_eq(roundtrip.tensor.storage(), uniform.storage()));
    assert_eq!(roundtrip.as_slice(), &[1.0, 2.0]);
}

#[test]
fn from_uniform_materializes_opaque_storage_once() {
    let (uniform, readbacks) = resident_f64_tensor();
    let typed = F64Tensor::from_uniform(&uniform).unwrap();

    assert_eq!(typed.as_slice(), &[1.5, 2.5]);
    assert_eq!(readbacks.load(Ordering::SeqCst), 1);

    let typed_uniform = typed.to_uniform();
    assert!(Arc::ptr_eq(typed.tensor.storage(), typed_uniform.storage()));
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

struct CountingResidentStorage {
    host: Arc<dyn TensorStorage>,
    readbacks: Arc<AtomicUsize>,
    materialized: OnceLock<Result<Arc<dyn TensorStorage>>>,
}

impl TensorStorage for CountingResidentStorage {
    fn dtype(&self) -> &Symbol {
        self.host.dtype()
    }

    fn len(&self) -> usize {
        self.host.len()
    }

    fn location(&self) -> TensorLocation {
        TensorLocation::Resident {
            site: Symbol::qualified("test", "site"),
            allocation: Symbol::qualified("test", "f64"),
        }
    }

    fn cell(&self, index: usize) -> Result<Value> {
        self.materialize()?.cell(index)
    }

    fn materialize(&self) -> Result<Arc<dyn TensorStorage>> {
        self.materialized
            .get_or_init(|| {
                self.readbacks.fetch_add(1, Ordering::SeqCst);
                Ok(self.host.clone())
            })
            .clone()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn resident_f64_tensor() -> (Tensor, Arc<AtomicUsize>) {
    let host = Tensor::new_exact(
        vec![2],
        domains::f64(),
        vec![number(domains::f64(), "1.5"), number(domains::f64(), "2.5")],
    )
    .unwrap();
    let readbacks = Arc::new(AtomicUsize::new(0));
    let storage = Arc::new(CountingResidentStorage {
        host: host.storage().clone(),
        readbacks: readbacks.clone(),
        materialized: OnceLock::new(),
    });
    (
        Tensor::from_storage(vec![2], domains::f64(), storage).unwrap(),
        readbacks,
    )
}

fn number(domain: Symbol, canonical: &str) -> Value {
    DefaultFactory
        .number_literal(domain, canonical.to_owned())
        .unwrap()
}
