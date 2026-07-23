//! The uniform `Tensor` value type: its shape, dtype, and cell storage, with
//! indexing, construction, and number-value behavior backing the tensor domain.

use std::sync::Arc;

use sim_kernel::{
    ClassRef, Cx, DefaultFactory, Error, Expr, Factory, NumberValue, Object, ObjectCompat,
    ObjectEncode, ObjectEncoding, Result, Symbol, Value,
};

use super::citizen::tensor_value_class_symbol;
use super::domain::number_domain;
use super::storage::{BoxedTensorStorage, TensorLocation, TensorStorage};
use super::validation::{
    choose_dtype, validate_cells, validate_dtype_accepts_cells, validate_exact_cell_dtype,
    validate_shape_and_data_len,
};

/// The uniform tensor value: an n-dimensional array of scalar number cells.
///
/// A tensor is row-major (last axis varies fastest) and homogeneous: every cell
/// shares the [`dtype`](Tensor::dtype) number domain. An empty [`shape`](Tensor::shape)
/// denotes a rank-0 scalar holding a single cell. Tensors are the value backing
/// the `numbers/tensor` domain and are constructed through
/// [`build_tensor_value`] rather than parsed from literals.
#[derive(Clone)]
pub struct Tensor {
    /// Length of each axis, outermost first. Empty for a rank-0 scalar.
    shape: Arc<[usize]>,
    /// The shared scalar number domain of every cell (for example
    /// `numbers/i64` or `numbers/f64`).
    dtype: Symbol,
    /// Row-major host or resident storage.
    storage: Arc<dyn TensorStorage>,
}

impl Tensor {
    /// Builds a tensor after checking shape, scalar-cell, and promotion
    /// invariants against the loaded number-domain registry.
    pub fn new_checked(
        cx: &mut Cx,
        shape: Vec<usize>,
        dtype: Symbol,
        data: Vec<Value>,
    ) -> Result<Self> {
        validate_shape_and_data_len(&shape, data.len())?;
        validate_cells(cx, &data)?;
        validate_dtype_accepts_cells(cx, &dtype, &data)?;
        Self::from_storage(
            shape,
            dtype.clone(),
            Arc::new(BoxedTensorStorage::new(dtype, data)),
        )
    }

    /// Builds a tensor whose cells already exactly match `dtype`.
    ///
    /// Specialized tensor backends use this when converting packed storage into
    /// uniform tensor cells without needing a loaded registry. It is stricter
    /// than [`Tensor::new_checked`]: every cell must report exactly `dtype`.
    pub fn new_exact(shape: Vec<usize>, dtype: Symbol, data: Vec<Value>) -> Result<Self> {
        validate_shape_and_data_len(&shape, data.len())?;
        validate_exact_cell_dtype(&dtype, &data)?;
        Self::from_storage(
            shape,
            dtype.clone(),
            Arc::new(BoxedTensorStorage::new(dtype, data)),
        )
    }

    /// Builds a canonical tensor around externally supplied storage.
    ///
    /// This validates shape, dtype, and logical length without materializing
    /// resident storage. The storage implementation is responsible for
    /// returning scalar values consistent with its declared dtype.
    pub fn from_storage(
        shape: Vec<usize>,
        dtype: Symbol,
        storage: Arc<dyn TensorStorage>,
    ) -> Result<Self> {
        validate_shape_and_data_len(&shape, storage.len())?;
        if storage.dtype() != &dtype {
            return Err(Error::Eval(format!(
                "tensor dtype {dtype} does not match storage dtype {}",
                storage.dtype()
            )));
        }
        Ok(Self {
            shape: shape.into(),
            dtype,
            storage,
        })
    }

    /// The tensor shape, outermost axis first. Empty means a rank-0 scalar.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// The shared scalar number domain accepted by every tensor cell.
    pub fn dtype(&self) -> &Symbol {
        &self.dtype
    }

    /// The current host or resident storage location.
    pub fn location(&self) -> TensorLocation {
        self.storage.location()
    }

    /// Borrows the open storage implementation.
    ///
    /// Typed adapters can safely downcast through
    /// [`TensorStorage::as_any`], while execution providers can preserve
    /// storage identity by cloning this `Arc`.
    pub fn storage(&self) -> &Arc<dyn TensorStorage> {
        &self.storage
    }

    /// The logical row-major cell count.
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    /// Whether this tensor has no logical cells.
    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    /// Observes one row-major scalar cell.
    pub fn cell(&self, index: usize) -> Result<Value> {
        if index >= self.len() {
            return Err(Error::Eval(
                "tensor cell index was out of bounds".to_owned(),
            ));
        }
        self.storage.cell(index)
    }

    /// Materializes storage into a checked host-observable form.
    pub fn materialize(&self) -> Result<Arc<dyn TensorStorage>> {
        let storage = if self.storage.location() == TensorLocation::Host {
            self.storage.clone()
        } else {
            self.storage.materialize()?
        };
        if storage.location() != TensorLocation::Host {
            return Err(Error::Eval(
                "tensor materialization did not produce host storage".to_owned(),
            ));
        }
        if storage.dtype() != &self.dtype {
            return Err(Error::Eval(format!(
                "materialized tensor dtype {} does not match {}",
                storage.dtype(),
                self.dtype
            )));
        }
        if storage.len() != self.len() {
            return Err(Error::Eval(format!(
                "materialized tensor length {} does not match {}",
                storage.len(),
                self.len()
            )));
        }
        Ok(storage)
    }

    /// Observes all row-major scalar cells.
    ///
    /// Boxed host storage returns its shared cell slice directly. Other host
    /// layouts are read through the open storage contract after one checked
    /// materialization.
    pub fn cells(&self) -> Result<Arc<[Value]>> {
        let storage = self.materialize()?;
        if let Some(boxed) = storage.as_any().downcast_ref::<BoxedTensorStorage>() {
            return Ok(boxed.cells());
        }
        (0..storage.len())
            .map(|index| storage.cell(index))
            .collect::<Result<Vec<_>>>()
            .map(Arc::from)
    }

    /// The number of axes, i.e. the length of [`shape`](Tensor::shape). Zero
    /// for a scalar.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Computes the row-major flat offset into storage for a
    /// multi-dimensional `indices` coordinate against `shape`.
    ///
    /// Returns an error if the index rank does not match `shape` or any
    /// component is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use sim_lib_numbers_tensor::Tensor;
    ///
    /// // Row-major 2x3 tensor: element (1, 2) is at flat offset 5.
    /// assert_eq!(Tensor::flat_offset(&[2, 3], &[1, 2]).unwrap(), 5);
    /// assert_eq!(Tensor::flat_offset(&[2, 3], &[0, 0]).unwrap(), 0);
    /// // Out-of-bounds and rank-mismatched indices are rejected.
    /// assert!(Tensor::flat_offset(&[2, 3], &[2, 0]).is_err());
    /// assert!(Tensor::flat_offset(&[2, 3], &[0]).is_err());
    /// ```
    pub fn flat_offset(shape: &[usize], indices: &[usize]) -> Result<usize> {
        if shape.len() != indices.len() {
            return Err(Error::Eval("tensor index rank mismatch".to_owned()));
        }
        let mut stride = 1usize;
        let mut offset = 0usize;
        for (dim, index) in shape.iter().rev().zip(indices.iter().rev()) {
            if *index >= *dim {
                return Err(Error::Eval("tensor index was out of bounds".to_owned()));
            }
            offset += index * stride;
            stride = stride.saturating_mul(*dim);
        }
        Ok(offset)
    }

    /// Enumerates every multi-dimensional coordinate of `shape` in row-major
    /// order. An empty shape yields a single empty coordinate (the scalar cell).
    pub fn coordinates(shape: &[usize]) -> Vec<Vec<usize>> {
        if shape.is_empty() {
            return vec![Vec::new()];
        }
        if shape.contains(&0) {
            return Vec::new();
        }
        let mut out = Vec::new();
        let mut coord = vec![0usize; shape.len()];
        loop {
            out.push(coord.clone());
            let mut axis = shape.len();
            while axis > 0 {
                axis -= 1;
                coord[axis] += 1;
                if coord[axis] < shape[axis] {
                    break;
                }
                coord[axis] = 0;
                if axis == 0 {
                    return out;
                }
            }
        }
    }
}

impl Object for Tensor {
    fn display(&self, cx: &mut Cx) -> Result<String> {
        match self.as_expr(cx)? {
            Expr::Call { .. } => Ok(format!("{}<{:?}>", tensor_display_name(), self.shape)),
            expr => Ok(format!("{expr:?}")),
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl sim_kernel::ObjectCompat for Tensor {
    fn class(&self, cx: &mut Cx) -> Result<ClassRef> {
        if let Some(value) = cx.registry().class_by_symbol(&tensor_value_class_symbol()) {
            return Ok(value.clone());
        }
        if let Some(value) = cx
            .registry()
            .class_by_symbol(&Symbol::qualified("core", "Number"))
        {
            return Ok(value.clone());
        }
        DefaultFactory.class_stub(
            sim_kernel::CORE_NUMBER_CLASS_ID,
            Symbol::qualified("core", "Number"),
        )
    }
    fn as_expr(&self, cx: &mut Cx) -> Result<Expr> {
        let cells = self.cells()?;
        match self.rank() {
            0 => Ok(Expr::Call {
                operator: Box::new(Expr::Symbol(Symbol::new("scalar"))),
                args: vec![
                    cells
                        .first()
                        .ok_or_else(|| Error::Eval("scalar tensor is missing its cell".to_owned()))?
                        .object()
                        .as_expr(cx)?,
                ],
            }),
            1 => Ok(Expr::Vector(exprs(cx, &cells)?)),
            2 => {
                let width = self.shape[1];
                let rows = if width == 0 {
                    vec![Expr::Vector(Vec::new()); self.shape[0]]
                } else {
                    cells
                        .chunks(width)
                        .map(|row| exprs(cx, row).map(Expr::Vector))
                        .collect::<Result<Vec<_>>>()?
                };
                Ok(Expr::Vector(rows))
            }
            _ => Ok(Expr::Call {
                operator: Box::new(Expr::Symbol(Symbol::new("tensor"))),
                args: vec![
                    Expr::Vector(
                        self.shape
                            .iter()
                            .map(|dim| Expr::String(dim.to_string()))
                            .collect(),
                    ),
                    Expr::Symbol(self.dtype.clone()),
                    Expr::Vector(exprs(cx, &cells)?),
                ],
            }),
        }
    }
    fn as_table(&self, cx: &mut Cx) -> Result<Value> {
        let shape = cx.factory().list(
            self.shape
                .iter()
                .map(|dim| cx.factory().string(dim.to_string()))
                .collect::<Result<Vec<_>>>()?,
        )?;
        let data = cx.factory().list(self.cells()?.to_vec())?;
        cx.factory().table(vec![
            (
                Symbol::new("kind"),
                cx.factory().string("tensor".to_owned())?,
            ),
            (Symbol::new("shape"), shape),
            (
                Symbol::new("dtype"),
                cx.factory().symbol(self.dtype.clone())?,
            ),
            (Symbol::new("data"), data),
        ])
    }
    fn as_number_value(&self) -> Option<&dyn NumberValue> {
        Some(self)
    }

    fn as_object_encoder(&self) -> Option<&dyn ObjectEncode> {
        Some(self)
    }
}

impl NumberValue for Tensor {
    fn number_domain(&self, _cx: &mut Cx) -> Result<Symbol> {
        Ok(number_domain())
    }
}

impl ObjectEncode for Tensor {
    fn object_encoding(&self, cx: &mut Cx) -> Result<ObjectEncoding> {
        let cells = self.cells()?;
        Ok(ObjectEncoding::Constructor {
            class: tensor_value_class_symbol(),
            args: vec![
                Expr::Symbol(Symbol::new("v1")),
                Expr::List(
                    self.shape
                        .iter()
                        .map(|dim| {
                            Expr::Number(sim_kernel::NumberLiteral {
                                domain: Symbol::qualified("citizen", "int"),
                                canonical: dim.to_string(),
                            })
                        })
                        .collect(),
                ),
                Expr::List(exprs(cx, &cells)?),
                Expr::Symbol(self.dtype.clone()),
            ],
        })
    }
}

impl sim_citizen::Citizen for Tensor {
    fn citizen_symbol() -> Symbol {
        tensor_value_class_symbol()
    }

    fn citizen_version() -> u32 {
        1
    }

    fn citizen_arity() -> usize {
        3
    }

    fn citizen_fields() -> &'static [&'static str] {
        &["shape", "data", "domain"]
    }
}

/// Builds a tensor [`Value`] of the given `shape` from row-major `data` cells.
///
/// The cell count must equal the product of `shape` (one for an empty, scalar
/// shape). Every cell must be a scalar number value (not a nested tensor). When
/// `dtype_hint` is `Some`, all cells must promote to that domain; otherwise the
/// element domain is chosen as the cheapest join of the cell domains. Returns an
/// error on a cell-count mismatch, a non-scalar cell, or an impossible dtype.
pub fn build_tensor_value(
    cx: &mut Cx,
    shape: Vec<usize>,
    dtype_hint: Option<Symbol>,
    data: Vec<Value>,
) -> Result<Value> {
    validate_shape_and_data_len(&shape, data.len())?;
    let dtype = choose_dtype(cx, dtype_hint, &data)?;
    let tensor = Tensor::new_checked(cx, shape, dtype, data)?;
    cx.factory().opaque(Arc::new(tensor))
}

/// Builds a rank-0 scalar tensor wrapping a single scalar number `value`.
pub fn build_scalar_tensor_value(cx: &mut Cx, value: Value) -> Result<Value> {
    build_tensor_value(cx, Vec::new(), None, vec![value])
}

/// Borrows the [`Tensor`] backing a value, or `None` if it is not a tensor.
pub fn tensor_value_ref(value: &Value) -> Option<&Tensor> {
    value.object().downcast_ref::<Tensor>()
}

/// The shared element number domain (dtype) of a tensor's cells.
pub fn tensor_dtype(tensor: &Tensor) -> &Symbol {
    tensor.dtype()
}

/// Observes a tensor's row-major scalar cells as a shared flat slice.
pub fn flatten_tensor_scalar_cells(tensor: &Tensor) -> Result<Arc<[Value]>> {
    tensor.cells()
}

pub fn tensor_display_name() -> &'static str {
    "tensor"
}

fn exprs(cx: &mut Cx, data: &[Value]) -> Result<Vec<Expr>> {
    data.iter()
        .map(|value| value.object().as_expr(cx))
        .collect()
}
