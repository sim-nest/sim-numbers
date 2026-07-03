//! The uniform `Tensor` value type: its shape, dtype, and cell storage, with
//! indexing, construction, and number-value behavior backing the tensor domain.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};
use std::sync::Arc;

use sim_kernel::{
    ClassRef, Cx, DefaultFactory, Error, Expr, Factory, NumberValue, Object, ObjectCompat,
    ObjectEncode, ObjectEncoding, Result, Symbol, Value,
};

use super::citizen::tensor_value_class_symbol;
use super::domain::number_domain;

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
    pub shape: Vec<usize>,
    /// The shared scalar number domain of every cell (for example
    /// `numbers/i64` or `numbers/f64`).
    pub dtype: Symbol,
    /// Row-major cell storage; its length equals the product of `shape`
    /// (one for a scalar).
    pub data: Vec<Value>,
}

impl Tensor {
    /// The number of axes, i.e. the length of [`shape`](Tensor::shape). Zero
    /// for a scalar.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Computes the row-major flat offset into [`data`](Tensor::data) for a
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
        match self.rank() {
            0 => Ok(Expr::Call {
                operator: Box::new(Expr::Symbol(Symbol::new("scalar"))),
                args: vec![self.data[0].object().as_expr(cx)?],
            }),
            1 => Ok(Expr::Vector(exprs(cx, &self.data)?)),
            2 => {
                let width = self.shape[1];
                let rows = self
                    .data
                    .chunks(width)
                    .map(|row| exprs(cx, row).map(Expr::Vector))
                    .collect::<Result<Vec<_>>>()?;
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
                    Expr::Vector(exprs(cx, &self.data)?),
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
        let data = cx.factory().list(self.data.clone())?;
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
                Expr::List(exprs(cx, &self.data)?),
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
    let expected = checked_element_count(&shape)?;
    if data.len() != expected {
        return Err(Error::Eval(format!(
            "tensor shape {:?} expects {expected} cells, found {}",
            shape,
            data.len()
        )));
    }
    validate_cells(cx, &data)?;
    let dtype = choose_dtype(cx, dtype_hint, &data)?;
    cx.factory().opaque(Arc::new(Tensor { shape, dtype, data }))
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
    &tensor.dtype
}

/// Clones a tensor's row-major cell values as a flat vector.
pub fn flatten_tensor_scalar_cells(tensor: &Tensor) -> Vec<Value> {
    tensor.data.clone()
}

pub fn tensor_display_name() -> &'static str {
    "tensor"
}

fn exprs(cx: &mut Cx, data: &[Value]) -> Result<Vec<Expr>> {
    data.iter()
        .map(|value| value.object().as_expr(cx))
        .collect()
}

use crate::spec::checked_element_count;

fn validate_cells(cx: &mut Cx, data: &[Value]) -> Result<()> {
    for cell in data {
        let Some(number) = cx.number_value_ref(cell.clone())? else {
            return Err(Error::Eval(
                "tensor cells must all be scalar number values".to_owned(),
            ));
        };
        if number.domain == number_domain() {
            return Err(Error::Eval(
                "tensor cells must be scalar numbers, not nested tensors".to_owned(),
            ));
        }
    }
    Ok(())
}

fn choose_dtype(cx: &mut Cx, dtype_hint: Option<Symbol>, data: &[Value]) -> Result<Symbol> {
    let domains = data
        .iter()
        .map(|value| {
            cx.number_value_ref(value.clone())?
                .map(|number| number.domain)
                .ok_or_else(|| {
                    Error::Eval("tensor cells must all be scalar number values".to_owned())
                })
        })
        .collect::<Result<Vec<_>>>()?;
    let Some(first) = domains.first() else {
        return Err(Error::Eval("tensor requires at least one cell".to_owned()));
    };
    if let Some(dtype) = dtype_hint {
        if domains
            .iter()
            .all(|domain| promotion_cost(cx, domain, &dtype).is_some())
        {
            return Ok(dtype);
        }
        return Err(Error::Eval(format!(
            "tensor dtype {dtype} is not a valid join for cell domains {domains:?}"
        )));
    }
    let candidates = cx
        .registry()
        .number_domains()
        .keys()
        .filter(|symbol| **symbol != number_domain())
        .cloned()
        .collect::<Vec<_>>();
    let mut best = None::<(u32, Symbol)>;
    for candidate in candidates {
        let mut total = 0u32;
        let mut valid = true;
        for domain in &domains {
            let Some(cost) = promotion_cost(cx, domain, &candidate) else {
                valid = false;
                break;
            };
            total += cost;
        }
        if !valid {
            continue;
        }
        match &best {
            Some((best_cost, best_symbol))
                if total > *best_cost || (total == *best_cost && candidate >= *best_symbol) => {}
            _ => best = Some((total, candidate)),
        }
    }
    best.map(|(_, symbol)| symbol)
        .ok_or_else(|| {
            Error::Eval(format!(
                "no join domain exists for tensor cells {domains:?}"
            ))
        })
        .or_else(|_| Ok(first.clone()))
}

fn promotion_cost(cx: &Cx, from: &Symbol, to: &Symbol) -> Option<u32> {
    if from == to {
        return Some(0);
    }

    #[derive(Clone, Eq, PartialEq)]
    struct State {
        cost: u32,
        symbol: Symbol,
    }

    impl Ord for State {
        fn cmp(&self, other: &Self) -> Ordering {
            other
                .cost
                .cmp(&self.cost)
                .then_with(|| other.symbol.cmp(&self.symbol))
        }
    }

    impl PartialOrd for State {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    let mut best = BTreeMap::<Symbol, u32>::new();
    let mut heap = BinaryHeap::new();
    best.insert(from.clone(), 0);
    heap.push(State {
        cost: 0,
        symbol: from.clone(),
    });

    while let Some(State { cost, symbol }) = heap.pop() {
        if &symbol == to {
            return Some(cost);
        }
        if best.get(&symbol).copied().unwrap_or(u32::MAX) < cost {
            continue;
        }
        for rule in cx
            .registry()
            .value_promotion_rules()
            .iter()
            .filter(|rule| rule.from_domain == symbol)
        {
            let next = cost + rule.cost as u32;
            let entry = best.entry(rule.to_domain.clone()).or_insert(u32::MAX);
            if next < *entry {
                *entry = next;
                heap.push(State {
                    cost: next,
                    symbol: rule.to_domain.clone(),
                });
            }
        }
    }
    None
}
