//! Tensor constructor operations: the `tensor`, `scalar`, `vec`, `mat`,
//! `reshape`, `index`, `slice`, and `map` callables over tensor values.

use std::any::Any;

use sim_kernel::{
    Args, Callable, ClassRef, Cx, DefaultFactory, Error, Expr, Factory, Object, QuoteMode, Result,
    Symbol, Value, force_list_to_vec,
};

use super::cast::{cast_function_impl, cast_symbol};
use super::dimension::{extract_dims, extract_usize};
use super::value::{
    Tensor, build_scalar_tensor_value, build_tensor_value, tensor_dtype, tensor_value_ref,
};

pub fn tensor_symbol() -> Symbol {
    Symbol::new("tensor")
}

pub fn scalar_symbol() -> Symbol {
    Symbol::new("scalar")
}

pub fn vec_symbol() -> Symbol {
    Symbol::new("vec")
}

pub fn mat_symbol() -> Symbol {
    Symbol::new("mat")
}

pub fn index_symbol() -> Symbol {
    Symbol::new("index")
}

pub fn reshape_symbol() -> Symbol {
    Symbol::new("reshape")
}

pub fn slice_symbol() -> Symbol {
    Symbol::new("slice")
}

pub fn map_symbol() -> Symbol {
    Symbol::new("map")
}

#[derive(Clone)]
pub struct TensorFunction {
    pub symbol: Symbol,
}

impl Object for TensorFunction {
    fn display(&self, _cx: &mut Cx) -> Result<String> {
        Ok(format!("#<function {}>", self.symbol))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl sim_kernel::ObjectCompat for TensorFunction {
    fn class(&self, cx: &mut Cx) -> Result<ClassRef> {
        if let Some(value) = cx
            .registry()
            .class_by_symbol(&Symbol::qualified("core", "Function"))
        {
            return Ok(value.clone());
        }
        DefaultFactory.class_stub(
            sim_kernel::CORE_FUNCTION_CLASS_ID,
            Symbol::qualified("core", "Function"),
        )
    }
    fn as_expr(&self, _cx: &mut Cx) -> Result<Expr> {
        Ok(Expr::Symbol(self.symbol.clone()))
    }
    fn as_callable(&self) -> Option<&dyn Callable> {
        Some(self)
    }
}

impl Callable for TensorFunction {
    fn call(&self, cx: &mut Cx, args: Args) -> Result<Value> {
        match self.symbol.clone() {
            symbol if symbol == tensor_symbol() => tensor_impl(cx, args.into_vec()),
            symbol if symbol == scalar_symbol() => scalar_impl(cx, args.into_vec()),
            symbol if symbol == vec_symbol() => vec_impl(cx, args.into_vec()),
            symbol if symbol == mat_symbol() => mat_impl(cx, args.into_vec()),
            symbol if symbol == index_symbol() => index_impl(cx, args.into_vec()),
            symbol if symbol == reshape_symbol() => reshape_impl(cx, args.into_vec()),
            symbol if symbol == slice_symbol() => slice_impl(cx, args.into_vec()),
            symbol if symbol == map_symbol() => map_impl(cx, args.into_vec()),
            symbol if symbol == cast_symbol() => cast_function_impl(cx, args.into_vec()),
            _ => Err(Error::Eval(format!(
                "unsupported tensor helper function {}",
                self.symbol
            ))),
        }
    }
}

fn tensor_impl(cx: &mut Cx, values: Vec<Value>) -> Result<Value> {
    let [shape_value, dtype_value, cells_value] = values.as_slice() else {
        return Err(Error::Eval(
            "tensor expects exactly three arguments: shape dtype values".to_owned(),
        ));
    };
    let shape = extract_dims(cx, shape_value, "tensor shape")?;
    let dtype = extract_optional_symbol(cx, dtype_value)?;
    let cells = extract_list_values(cx, cells_value, "tensor values")?;
    build_tensor_value(cx, shape, dtype, cells)
}

fn scalar_impl(cx: &mut Cx, values: Vec<Value>) -> Result<Value> {
    let [value] = values.as_slice() else {
        return Err(Error::Eval(
            "scalar expects exactly one argument".to_owned(),
        ));
    };
    build_scalar_tensor_value(cx, value.clone())
}

fn vec_impl(cx: &mut Cx, values: Vec<Value>) -> Result<Value> {
    build_tensor_value(cx, vec![values.len()], None, values)
}

fn mat_impl(cx: &mut Cx, values: Vec<Value>) -> Result<Value> {
    let [rows_value] = values.as_slice() else {
        return Err(Error::Eval(
            "mat expects exactly one rows argument".to_owned(),
        ));
    };
    let rows = extract_list_values(cx, rows_value, "mat rows")?;
    let mut shape = vec![rows.len()];
    let mut cells = Vec::new();
    let mut width = None;
    for row in rows {
        let row_values = extract_list_values(cx, &row, "mat row")?;
        match width {
            Some(expected) if expected != row_values.len() => {
                return Err(Error::Eval(
                    "mat expects rows with equal lengths".to_owned(),
                ));
            }
            None => width = Some(row_values.len()),
            _ => {}
        }
        cells.extend(row_values);
    }
    shape.push(width.unwrap_or(0));
    build_tensor_value(cx, shape, None, cells)
}

fn index_impl(cx: &mut Cx, values: Vec<Value>) -> Result<Value> {
    let [tensor_value, indices @ ..] = values.as_slice() else {
        return Err(Error::Eval(
            "index expects a tensor followed by one or more indices".to_owned(),
        ));
    };
    let tensor = tensor_value_ref(tensor_value)
        .ok_or_else(|| Error::Eval("index expects a tensor as its first argument".to_owned()))?;
    let shape = tensor.shape();
    if indices.len() != shape.len() {
        return Err(Error::Eval(format!(
            "index expected {} indices, found {}",
            shape.len(),
            indices.len()
        )));
    }
    let offsets = indices
        .iter()
        .map(|value| extract_usize(cx, value, "tensor index"))
        .collect::<Result<Vec<_>>>()?;
    let flat = Tensor::flat_offset(shape, &offsets)?;
    tensor.cell(flat)
}

fn reshape_impl(cx: &mut Cx, values: Vec<Value>) -> Result<Value> {
    let [tensor_value, shape_value] = values.as_slice() else {
        return Err(Error::Eval(
            "reshape expects exactly two arguments: tensor shape".to_owned(),
        ));
    };
    let tensor = tensor_value_ref(tensor_value)
        .ok_or_else(|| Error::Eval("reshape expects a tensor value".to_owned()))?;
    let shape = extract_dims(cx, shape_value, "reshape shape")?;
    build_tensor_value(
        cx,
        shape,
        Some(tensor_dtype(tensor).clone()),
        tensor.cells()?.to_vec(),
    )
}

fn slice_impl(cx: &mut Cx, values: Vec<Value>) -> Result<Value> {
    let [tensor_value, starts_value, lens_value] = values.as_slice() else {
        return Err(Error::Eval(
            "slice expects exactly three arguments: tensor starts lengths".to_owned(),
        ));
    };
    let tensor = tensor_value_ref(tensor_value)
        .ok_or_else(|| Error::Eval("slice expects a tensor value".to_owned()))?;
    let starts = extract_dims(cx, starts_value, "slice starts")?;
    let lens = extract_dims(cx, lens_value, "slice lengths")?;
    if starts.len() != tensor.shape().len() || lens.len() != tensor.shape().len() {
        return Err(Error::Eval(
            "slice starts and lengths must match tensor rank".to_owned(),
        ));
    }
    let mut cells = Vec::new();
    for coord in Tensor::coordinates(&lens) {
        let absolute = coord
            .iter()
            .zip(starts.iter())
            .map(|(offset, start)| offset + start)
            .collect::<Vec<_>>();
        let flat = Tensor::flat_offset(tensor.shape(), &absolute)?;
        cells.push(tensor.cell(flat)?);
    }
    build_tensor_value(cx, lens, Some(tensor_dtype(tensor).clone()), cells)
}

fn map_impl(cx: &mut Cx, values: Vec<Value>) -> Result<Value> {
    let [function, tensor_value] = values.as_slice() else {
        return Err(Error::Eval(
            "map expects exactly two arguments: function tensor".to_owned(),
        ));
    };
    let tensor = tensor_value_ref(tensor_value)
        .ok_or_else(|| Error::Eval("map expects a tensor value".to_owned()))?;
    let tensor_cells = tensor.cells()?;
    let mut cells = Vec::with_capacity(tensor_cells.len());
    for cell in tensor_cells.iter() {
        cells.push(cx.call_value(function.clone(), Args::new(vec![cell.clone()]))?);
    }
    build_tensor_value(cx, tensor.shape().to_vec(), None, cells)
}

fn extract_optional_symbol(cx: &mut Cx, value: &Value) -> Result<Option<Symbol>> {
    match value.object().as_expr(cx)? {
        Expr::Nil => Ok(None),
        Expr::Symbol(symbol) => Ok(Some(symbol)),
        Expr::Quote {
            mode: QuoteMode::Quote,
            expr,
        } => match *expr {
            Expr::Symbol(symbol) => Ok(Some(symbol)),
            _ => Err(Error::Eval("expected a symbol for tensor dtype".to_owned())),
        },
        _ => Err(Error::Eval(
            "expected a symbol or nil for tensor dtype".to_owned(),
        )),
    }
}

fn extract_list_values(cx: &mut Cx, value: &Value, context: &str) -> Result<Vec<Value>> {
    let list = value
        .object()
        .as_list()
        .ok_or_else(|| Error::Eval(format!("{context} must be a list or vector")))?;
    force_list_to_vec(cx, list, context)
}
