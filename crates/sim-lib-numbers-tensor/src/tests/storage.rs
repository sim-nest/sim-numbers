use std::any::Any;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use sim_kernel::{DefaultFactory, Error, Factory, Result, Symbol, Value};

use crate::{Tensor, TensorLocation, TensorStorage, build_tensor_value};

use super::{number, test_cx};

struct TestResidentStorage {
    dtype: Symbol,
    host: Arc<dyn TensorStorage>,
    readbacks: Arc<AtomicUsize>,
    fail: bool,
    materialized: OnceLock<Result<Arc<dyn TensorStorage>>>,
}

impl TensorStorage for TestResidentStorage {
    fn dtype(&self) -> &Symbol {
        &self.dtype
    }

    fn len(&self) -> usize {
        self.host.len()
    }

    fn location(&self) -> TensorLocation {
        TensorLocation::Resident {
            site: Symbol::qualified("test", "site"),
            allocation: Symbol::qualified("test", "allocation"),
        }
    }

    fn cell(&self, index: usize) -> Result<Value> {
        self.materialize()?.cell(index)
    }

    fn materialize(&self) -> Result<Arc<dyn TensorStorage>> {
        self.materialized
            .get_or_init(|| {
                self.readbacks.fetch_add(1, Ordering::SeqCst);
                if self.fail {
                    Err(Error::Eval("test tensor readback failed".to_owned()))
                } else {
                    Ok(self.host.clone())
                }
            })
            .clone()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn resident_tensor(fail: bool) -> (Tensor, Vec<Value>, Arc<AtomicUsize>) {
    let dtype = Symbol::qualified("numbers", "i64");
    let cells = vec![number("i64", "3"), number("i64", "5")];
    let host = Tensor::new_exact(vec![2], dtype.clone(), cells.clone()).unwrap();
    let readbacks = Arc::new(AtomicUsize::new(0));
    let storage = Arc::new(TestResidentStorage {
        dtype: dtype.clone(),
        host: host.storage().clone(),
        readbacks: readbacks.clone(),
        fail,
        materialized: OnceLock::new(),
    });
    (
        Tensor::from_storage(vec![2], dtype, storage).unwrap(),
        cells,
        readbacks,
    )
}

#[test]
fn tensor_observation_propagates_and_caches_storage_failure() {
    let (tensor, _, readbacks) = resident_tensor(true);
    assert!(tensor.cell(0).unwrap_err().to_string().contains("readback"));
    assert!(tensor.cells().unwrap_err().to_string().contains("readback"));
    assert_eq!(readbacks.load(Ordering::SeqCst), 1);
}

#[test]
fn tensor_materialization_is_idempotent_and_preserves_aliases() {
    let (tensor, source, readbacks) = resident_tensor(false);
    let clone = tensor.clone();
    assert!(Arc::ptr_eq(tensor.storage(), clone.storage()));

    let first_storage = tensor.materialize().unwrap();
    let second_storage = tensor.materialize().unwrap();
    assert!(Arc::ptr_eq(&first_storage, &second_storage));

    let first_cells = tensor.cells().unwrap();
    let second_cells = tensor.cells().unwrap();
    assert!(Arc::ptr_eq(&first_cells, &second_cells));
    assert_eq!(first_cells[0], source[0]);
    assert_eq!(first_cells[1], source[1]);
    assert_eq!(readbacks.load(Ordering::SeqCst), 1);

    let mut cx = test_cx();
    let value = DefaultFactory.opaque(Arc::new(tensor)).unwrap();
    sim_citizen::check_value_fixture(&mut cx, value).unwrap();
    assert_eq!(readbacks.load(Ordering::SeqCst), 1);
}

#[test]
fn zero_sized_tensor_keeps_shape_dtype_and_empty_observation() {
    let dtype = Symbol::qualified("numbers", "i64");
    let tensor = Tensor::new_exact(vec![2, 0, 4], dtype.clone(), Vec::new()).unwrap();
    assert_eq!(tensor.shape(), &[2, 0, 4]);
    assert_eq!(tensor.dtype(), &dtype);
    assert!(tensor.is_empty());
    assert!(tensor.cells().unwrap().is_empty());
    assert!(Tensor::coordinates(tensor.shape()).is_empty());

    let mut cx = test_cx();
    let value = build_tensor_value(&mut cx, vec![0], Some(dtype), Vec::new()).unwrap();
    sim_citizen::check_value_fixture(&mut cx, value).unwrap();
}

#[test]
fn concurrent_tensor_observation_performs_one_readback() {
    let (tensor, _, readbacks) = resident_tensor(false);
    let tensor = Arc::new(tensor);
    let observations = (0..8)
        .map(|_| {
            let tensor = tensor.clone();
            std::thread::spawn(move || tensor.cells().unwrap())
        })
        .collect::<Vec<_>>()
        .into_iter()
        .map(|thread| thread.join().unwrap())
        .collect::<Vec<_>>();

    for cells in observations.iter().skip(1) {
        assert!(Arc::ptr_eq(&observations[0], cells));
    }
    assert_eq!(readbacks.load(Ordering::SeqCst), 1);
}
