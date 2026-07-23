//! Open tensor storage and the boxed host-storage implementation.

use std::any::Any;
use std::sync::Arc;

use sim_kernel::{DefaultFactory, Error, Factory, Result, Symbol, Value};
use sim_lib_numbers_core::domains;

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

/// A scalar cell type that can be held in typed host tensor storage.
pub trait TensorCell: Clone + Send + Sync + 'static {
    /// The scalar number domain this cell encodes.
    fn dtype() -> Symbol;

    /// Encodes the typed cell as a scalar runtime value.
    fn to_value(&self) -> Result<Value>;
}

/// Host storage for typed scalar cells behind the canonical
/// [`Tensor`](super::value::Tensor).
///
/// Typed tensor adapters use this storage so their public wrapper and the
/// uniform `Tensor` share one storage lifetime. The adapter still gets a native
/// cell slice for fast operations, while the runtime observes cells through the
/// ordinary [`TensorStorage`] contract.
pub struct TypedTensorStorage<T: TensorCell> {
    dtype: Symbol,
    cells: Arc<[T]>,
}

impl<T: TensorCell> TypedTensorStorage<T> {
    /// Builds typed host storage from a flat row-major cell buffer.
    pub fn new(cells: Vec<T>) -> Self {
        Self::from_shared(cells.into())
    }

    /// Builds typed host storage from an already shared cell buffer.
    pub fn from_shared(cells: Arc<[T]>) -> Self {
        Self {
            dtype: T::dtype(),
            cells,
        }
    }

    /// Borrows the typed row-major cells.
    pub fn cell_slice(&self) -> &[T] {
        &self.cells
    }

    /// Clones the shared typed row-major cells.
    pub fn cells(&self) -> Arc<[T]> {
        self.cells.clone()
    }
}

impl<T: TensorCell> TensorStorage for TypedTensorStorage<T> {
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
            .ok_or_else(|| Error::Eval("tensor cell index was out of bounds".to_owned()))?
            .to_value()
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

impl TensorCell for f64 {
    fn dtype() -> Symbol {
        domains::f64()
    }

    fn to_value(&self) -> Result<Value> {
        DefaultFactory.number_literal(domains::f64(), self.to_string())
    }
}

impl TensorCell for i64 {
    fn dtype() -> Symbol {
        domains::i64()
    }

    fn to_value(&self) -> Result<Value> {
        DefaultFactory.number_literal(domains::i64(), self.to_string())
    }
}

impl TensorCell for bool {
    fn dtype() -> Symbol {
        domains::bool()
    }

    fn to_value(&self) -> Result<Value> {
        DefaultFactory.number_literal(domains::bool(), self.to_string())
    }
}

impl TensorCell for (f64, f64) {
    fn dtype() -> Symbol {
        domains::complex()
    }

    fn to_value(&self) -> Result<Value> {
        DefaultFactory.number_literal(domains::complex(), format!("{}{:+}i", self.0, self.1))
    }
}

impl TensorCell for (i64, i64) {
    fn dtype() -> Symbol {
        domains::rational()
    }

    fn to_value(&self) -> Result<Value> {
        DefaultFactory.number_literal(domains::rational(), format!("{}/{}", self.0, self.1))
    }
}
