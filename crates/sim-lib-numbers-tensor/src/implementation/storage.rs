//! Open tensor storage and the boxed host-storage implementation.

use std::any::Any;
use std::sync::Arc;

use sim_kernel::{Error, Result, Symbol, Value};

/// The observable placement of tensor storage.
///
/// Host storage can be read directly. Resident storage belongs to a loadable
/// execution site and names an opaque allocation supplied by that site.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TensorLocation {
    /// Storage is directly observable by the host runtime.
    Host,
    /// Storage resides at a loadable execution site.
    Resident {
        /// The site that owns the allocation.
        site: Symbol,
        /// An opaque allocation name meaningful to that site.
        allocation: Symbol,
    },
}

/// Storage behind the canonical [`Tensor`](super::value::Tensor) value.
///
/// Implementations may keep cells in host-native or resident layouts.
/// Observation is fallible: a resident implementation performs and caches its
/// checked readback in [`materialize`](TensorStorage::materialize). A successful
/// materialization must return host storage with the same dtype and length.
pub trait TensorStorage: Send + Sync + 'static {
    /// The logical scalar domain of every cell.
    fn dtype(&self) -> &Symbol;

    /// The logical row-major cell count.
    fn len(&self) -> usize;

    /// Whether the storage has no logical cells.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The current storage placement.
    fn location(&self) -> TensorLocation;

    /// Observes one row-major cell.
    fn cell(&self, index: usize) -> Result<Value>;

    /// Returns a host-observable form of this storage.
    ///
    /// Resident implementations cache both success and failure so repeated or
    /// concurrent observations perform at most one readback.
    fn materialize(&self) -> Result<Arc<dyn TensorStorage>>;

    /// Exposes the concrete storage for safe downcasting by typed adapters.
    fn as_any(&self) -> &dyn Any;
}

/// Boxed host storage for scalar runtime values.
///
/// This is the default storage used by tensor constructors. Its cells are
/// reference counted so tensor clones and repeated observations preserve
/// value identity without copying the cell vector.
pub struct BoxedTensorStorage {
    dtype: Symbol,
    cells: Arc<[Value]>,
}

impl BoxedTensorStorage {
    pub(crate) fn new(dtype: Symbol, cells: Vec<Value>) -> Self {
        Self {
            dtype,
            cells: cells.into(),
        }
    }

    pub(crate) fn cells(&self) -> Arc<[Value]> {
        self.cells.clone()
    }
}

impl TensorStorage for BoxedTensorStorage {
    fn dtype(&self) -> &Symbol {
        &self.dtype
    }

    fn len(&self) -> usize {
        self.cells.len()
    }

    fn location(&self) -> TensorLocation {
        TensorLocation::Host
    }

    fn cell(&self, index: usize) -> Result<Value> {
        self.cells
            .get(index)
            .cloned()
            .ok_or_else(|| Error::Eval("tensor cell index was out of bounds".to_owned()))
    }

    fn materialize(&self) -> Result<Arc<dyn TensorStorage>> {
        Ok(Arc::new(Self {
            dtype: self.dtype.clone(),
            cells: self.cells.clone(),
        }))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
