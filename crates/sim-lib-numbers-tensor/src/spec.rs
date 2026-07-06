//! The `SpecTensor` interface and descriptor types that let specialized
//! element-type backends convert to and from the uniform `Tensor` storage, plus
//! literal-cell parsing helpers shared across those backends.

use std::sync::Arc;

use sim_kernel::{
    Cx, DefaultFactory, Factory, NoopEvalPolicy, NumberLiteral, Result, Symbol, Value,
};

use crate::Tensor;
use sim_lib_numbers_core::domains;

/// Interface a specialized element-type tensor backend implements to bridge its
/// own storage and the uniform [`Tensor`] value.
///
/// Typed backends (for example dense `f64` or `i64` tensors) keep their own
/// packed representation and use this trait to convert to and from the shared
/// uniform storage the `numbers/tensor` domain operates on.
pub trait SpecTensor: Send + Sync + 'static {
    /// The length of each axis of the specialized tensor, outermost first.
    fn shape(&self) -> &[usize];
    /// The element number domain (dtype) of the specialized tensor's cells.
    fn dtype(&self) -> Symbol;
    /// Converts this specialized tensor into the uniform [`Tensor`] storage.
    fn to_uniform(&self) -> Tensor;
    /// Rebuilds a specialized tensor from uniform storage, or `None` if the
    /// uniform tensor's dtype or shape does not fit this backend.
    fn from_uniform(tensor: &Tensor) -> Option<Self>
    where
        Self: Sized;
}

/// Metadata describing one registered `SpecTensor` backend, surfaced as a
/// descriptor value so the registry can advertise the specialized tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpecTensorDescriptor {
    /// The symbol under which the backend's descriptor value is installed.
    pub symbol: Symbol,
    /// The element number domain (dtype) the backend specializes on.
    pub dtype: Symbol,
    /// Human-readable name of the implementing crate or strategy.
    pub implementation: &'static str,
    /// Human-readable description of the backend's storage layout.
    pub storage: &'static str,
}

/// Builds a descriptor symbol (`numbers/tensor-spec/<name>`) for a specialized
/// tensor backend.
///
/// # Examples
///
/// ```
/// use sim_lib_numbers_tensor::spec_tensor_symbol;
///
/// let symbol = spec_tensor_symbol("dense-f64");
/// assert_eq!(symbol.to_string(), "numbers/tensor-spec/dense-f64");
/// ```
pub fn spec_tensor_symbol(name: &str) -> Symbol {
    Symbol::qualified("numbers/tensor-spec", name)
}

/// Encodes a [`SpecTensorDescriptor`] as a registry descriptor table value with
/// `kind`, `symbol`, `dtype`, `implementation`, and `storage` entries.
pub fn spec_tensor_descriptor_value(
    factory: &dyn Factory,
    descriptor: SpecTensorDescriptor,
) -> Result<Value> {
    factory.table(vec![
        (
            Symbol::new("kind"),
            factory.string("spec-tensor".to_owned())?,
        ),
        (Symbol::new("symbol"), factory.symbol(descriptor.symbol)?),
        (Symbol::new("dtype"), factory.symbol(descriptor.dtype)?),
        (
            Symbol::new("implementation"),
            factory.string(descriptor.implementation.to_owned())?,
        ),
        (
            Symbol::new("storage"),
            factory.string(descriptor.storage.to_owned())?,
        ),
    ])
}

/// The number of cells in a tensor of the given shape. An empty shape is a
/// scalar (one cell). This is the one home for the `element_count` helper that
/// the generic, broadcast, linalg, and every typed tensor crate re-grew.
///
/// # Examples
///
/// ```
/// use sim_lib_numbers_tensor::element_count;
///
/// assert_eq!(element_count(&[]), 1); // rank-0 scalar
/// assert_eq!(element_count(&[3]), 3); // length-3 vector
/// assert_eq!(element_count(&[2, 3]), 6); // 2x3 matrix
/// ```
pub fn element_count(shape: &[usize]) -> usize {
    if shape.is_empty() {
        1
    } else {
        shape.iter().product()
    }
}

/// The number of cells in a tensor of the given shape, failing closed when the
/// dimension product overflows `usize` instead of wrapping (release) or
/// panicking (debug).
///
/// [`element_count`] assumes an already-validated shape; this is the form to use
/// at the untrusted-input boundary -- for example a user-supplied `reshape`
/// shape parsed from arbitrary dimensions -- where a hostile dimension product
/// would otherwise overflow.
///
/// # Examples
///
/// ```
/// use sim_lib_numbers_tensor::checked_element_count;
///
/// assert_eq!(checked_element_count(&[]).unwrap(), 1); // rank-0 scalar
/// assert_eq!(checked_element_count(&[2, 3]).unwrap(), 6); // 2x3 matrix
/// assert!(checked_element_count(&[usize::MAX, 2]).is_err()); // overflow
/// ```
pub fn checked_element_count(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1_usize, |acc, &dim| {
        acc.checked_mul(dim).ok_or_else(|| {
            sim_kernel::Error::Eval(format!("tensor shape {shape:?} cell count overflows usize"))
        })
    })
}

/// The largest number of cells a tensor operation will materialize in one
/// allocation. A dimension product can be far below `usize::MAX` and still be
/// hopeless to allocate (a `[1_000_000, 1_000_000]` broadcast is `1e12` cells);
/// this ceiling is the line past which the input is rejected rather than driven
/// into an out-of-memory abort.
pub const MAX_TENSOR_CELLS: usize = 1 << 28;

/// The number of cells in a tensor of the given shape, failing closed both when
/// the dimension product overflows `usize` (via [`checked_element_count`]) and
/// when it exceeds [`MAX_TENSOR_CELLS`].
///
/// This is the form to use before sizing an allocation from untrusted
/// dimensions -- a broadcast result shape, a `zeros`/`ones`/`eye` size -- where a
/// legal-but-hostile shape whose product still fits in `usize` would otherwise
/// OOM the process.
///
/// # Examples
///
/// ```
/// use sim_lib_numbers_tensor::bounded_element_count;
///
/// assert_eq!(bounded_element_count(&[2, 3]).unwrap(), 6); // 2x3 matrix
/// assert!(bounded_element_count(&[usize::MAX, 2]).is_err()); // overflow
/// assert!(bounded_element_count(&[1_000_000, 1_000_000]).is_err()); // over ceiling
/// ```
pub fn bounded_element_count(shape: &[usize]) -> Result<usize> {
    let cells = checked_element_count(shape)?;
    if cells > MAX_TENSOR_CELLS {
        return Err(sim_kernel::Error::Eval(format!(
            "tensor shape {shape:?} has {cells} cells, exceeding the {MAX_TENSOR_CELLS}-cell limit"
        )));
    }
    Ok(cells)
}

/// Extracts the canonical [`NumberLiteral`] of a scalar tensor cell `value`, or
/// `None` if the value is not a number. Shared backing for the typed
/// literal-cell parsers below.
pub fn number_literal_for_tensor_cell(value: &Value) -> Option<NumberLiteral> {
    let mut cx = Cx::new(Arc::new(NoopEvalPolicy), Arc::new(DefaultFactory));
    value
        .object()
        .as_number_value()?
        .number_literal(&mut cx)
        .ok()?
}

/// Parses a tensor cell as an `i64`, returning `None` unless it is a number in
/// the `numbers/i64` domain whose canonical form parses cleanly.
pub fn parse_i64_literal_cell(value: &Value) -> Option<i64> {
    let literal = number_literal_for_tensor_cell(value)?;
    (literal.domain == domains::i64())
        .then(|| literal.canonical.parse::<i64>().ok())
        .flatten()
}

/// Parses a tensor cell as an `f64`, returning `None` unless it is a number in
/// the `numbers/f64` domain whose canonical form parses cleanly.
pub fn parse_f64_literal_cell(value: &Value) -> Option<f64> {
    let literal = number_literal_for_tensor_cell(value)?;
    (literal.domain == domains::f64())
        .then(|| literal.canonical.parse::<f64>().ok())
        .flatten()
}

/// Parses a tensor cell as a `(numerator, denominator)` rational pair,
/// returning `None` unless it is a number in the `numbers/rational` domain
/// whose canonical `num/den` form parses cleanly.
pub fn parse_rational_literal_cell(value: &Value) -> Option<(i64, i64)> {
    let literal = number_literal_for_tensor_cell(value)?;
    if literal.domain != domains::rational() {
        return None;
    }
    let (num, den) = literal.canonical.split_once('/')?;
    Some((num.parse::<i64>().ok()?, den.parse::<i64>().ok()?))
}

/// Parses a tensor cell as a `(real, imaginary)` pair, returning `None` unless
/// it is a number in the `numbers/complex` domain whose canonical `a+bi` form
/// parses cleanly.
pub fn parse_complex_literal_cell(value: &Value) -> Option<(f64, f64)> {
    let literal = number_literal_for_tensor_cell(value)?;
    if literal.domain != domains::complex() {
        return None;
    }
    let text = literal.canonical.strip_suffix('i')?;
    let split = text
        .char_indices()
        .skip(1)
        .find(|(_, ch)| *ch == '+' || *ch == '-')
        .map(|(index, _)| index)?;
    let (real, imag) = text.split_at(split);
    Some((real.parse::<f64>().ok()?, imag.parse::<f64>().ok()?))
}
