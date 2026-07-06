//! The broadcasting library: the `TensorBroadcastLib` that registers
//! element-wise tensor operations and their shape-broadcasting promotion rules.

use sim_kernel::{
    AbiVersion, Cx, Dependency, Error, Lib, LibManifest, LibTarget, Linker, Result, Symbol, Value,
    ValueNumberBinaryOp, ValueNumberUnaryOp, ValuePromotionRule, Version,
};
use sim_lib_numbers_core::domains;
use sim_lib_numbers_tensor::{
    Tensor, bounded_element_count, build_scalar_tensor_value, build_tensor_value,
    flatten_tensor_scalar_cells, number_domain, tensor_dtype, tensor_value_ref,
};

/// Registered library that installs NumPy-style broadcasting for the
/// `numbers/tensor` domain.
///
/// Loading this [`Lib`] registers the `math` element-wise binary operators
/// (`add`, `sub`, `mul`, `div`, `rem`, `pow`) and the unary `neg` over tensor
/// values, plus a promotion rule that lifts any scalar number domain into a
/// rank-0 tensor. Binary operators broadcast their operand shapes following the
/// trailing-axis rule (a dimension of `1` stretches to match its partner) and
/// dispatch each cell through the kernel's per-domain scalar operations.
pub struct TensorBroadcastLib;

impl TensorBroadcastLib {
    /// Creates the broadcasting library. The value is stateless; the promotion
    /// rule and element-wise operators are installed when it is loaded into a
    /// [`Cx`].
    pub fn new() -> Self {
        Self
    }
}

impl Default for TensorBroadcastLib {
    fn default() -> Self {
        Self::new()
    }
}

impl Lib for TensorBroadcastLib {
    fn manifest(&self) -> LibManifest {
        LibManifest {
            id: domains::tensor_bcast(),
            version: Version(env!("CARGO_PKG_VERSION").to_owned()),
            abi: AbiVersion { major: 0, minor: 1 },
            target: LibTarget::HostRegistered,
            requires: Vec::<Dependency>::new(),
            capabilities: Vec::new(),
            exports: Vec::new(),
        }
    }

    fn load(&self, cx: &mut sim_kernel::LoadCx, linker: &mut Linker<'_>) -> Result<()> {
        for domain in cx.registry().number_domains().keys() {
            if *domain == number_domain() {
                continue;
            }
            linker.value_promotion_rule(ValuePromotionRule {
                from_domain: domain.clone(),
                to_domain: number_domain(),
                cost: 20,
                convert: promote_scalar_to_tensor,
            });
        }

        for operator in [
            Symbol::qualified("math", "add"),
            Symbol::qualified("math", "sub"),
            Symbol::qualified("math", "mul"),
            Symbol::qualified("math", "div"),
            Symbol::qualified("math", "rem"),
            Symbol::qualified("math", "pow"),
        ] {
            linker.value_number_binary_op(ValueNumberBinaryOp {
                operator: operator.clone(),
                left_domain: number_domain(),
                right_domain: number_domain(),
                cost: 1,
                apply: binary_apply_for(&operator),
            });
        }

        linker.value_number_unary_op(ValueNumberUnaryOp {
            operator: Symbol::qualified("math", "neg"),
            operand_domain: number_domain(),
            cost: 1,
            apply: apply_tensor_neg,
        });
        Ok(())
    }
}

fn promote_scalar_to_tensor(cx: &mut Cx, value: Value) -> Result<Value> {
    if tensor_value_ref(&value).is_some() {
        return Ok(value);
    }
    build_scalar_tensor_value(cx, value)
}

fn binary_apply_for(operator: &Symbol) -> fn(&mut Cx, Value, Value) -> Result<Value> {
    if *operator == Symbol::qualified("math", "add") {
        apply_tensor_add
    } else if *operator == Symbol::qualified("math", "sub") {
        apply_tensor_sub
    } else if *operator == Symbol::qualified("math", "mul") {
        apply_tensor_mul
    } else if *operator == Symbol::qualified("math", "div") {
        apply_tensor_div
    } else if *operator == Symbol::qualified("math", "rem") {
        apply_tensor_rem
    } else {
        apply_tensor_pow
    }
}

fn apply_tensor_add(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    apply_tensor_binary(cx, Symbol::qualified("math", "add"), left, right)
}

fn apply_tensor_sub(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    apply_tensor_binary(cx, Symbol::qualified("math", "sub"), left, right)
}

fn apply_tensor_mul(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    apply_tensor_binary(cx, Symbol::qualified("math", "mul"), left, right)
}

fn apply_tensor_div(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    apply_tensor_binary(cx, Symbol::qualified("math", "div"), left, right)
}

fn apply_tensor_rem(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    apply_tensor_binary(cx, Symbol::qualified("math", "rem"), left, right)
}

fn apply_tensor_pow(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    apply_tensor_binary(cx, Symbol::qualified("math", "pow"), left, right)
}

fn apply_tensor_binary(cx: &mut Cx, operator: Symbol, left: Value, right: Value) -> Result<Value> {
    let left = tensor_value_ref(&left)
        .ok_or_else(|| Error::Eval("left operand was not a tensor value".to_owned()))?;
    let right = tensor_value_ref(&right)
        .ok_or_else(|| Error::Eval("right operand was not a tensor value".to_owned()))?;
    let shape = broadcast_shape(&left.shape, &right.shape)?;
    // Size and materialize the result from the checked, ceiling-bounded cell
    // count: a legal pair like [1_000_000, 1] x [1, 1_000_000] broadcasts to a
    // 1e12-cell shape that fits in usize but would OOM if allocated. Fail closed
    // before building the coordinate list or the result vector.
    let cells_len = bounded_element_count(&shape)?;
    let mut cells = Vec::with_capacity(cells_len);
    for coord in Tensor::coordinates(&shape) {
        let left_cell = select_cell(left, &coord, &shape)?;
        let right_cell = select_cell(right, &coord, &shape)?;
        cells.push(cx.apply_value_number_binary_op(&operator, left_cell, right_cell)?);
    }
    build_tensor_value(cx, shape, None, cells)
}

fn apply_tensor_neg(cx: &mut Cx, value: Value) -> Result<Value> {
    let tensor = tensor_value_ref(&value)
        .ok_or_else(|| Error::Eval("neg expects a tensor value".to_owned()))?;
    let mut cells = Vec::with_capacity(tensor.data.len());
    for cell in flatten_tensor_scalar_cells(tensor) {
        cells.push(cx.apply_value_number_unary_op(&Symbol::qualified("math", "neg"), cell)?);
    }
    build_tensor_value(
        cx,
        tensor.shape.clone(),
        Some(tensor_dtype(tensor).clone()),
        cells,
    )
}

fn broadcast_shape(left: &[usize], right: &[usize]) -> Result<Vec<usize>> {
    let rank = left.len().max(right.len());
    let mut out = Vec::with_capacity(rank);
    for axis in 0..rank {
        let left_dim = *left
            .get(left.len().wrapping_sub(rank - axis))
            .unwrap_or(&1usize);
        let right_dim = *right
            .get(right.len().wrapping_sub(rank - axis))
            .unwrap_or(&1usize);
        if left_dim == right_dim || left_dim == 1 || right_dim == 1 {
            out.push(left_dim.max(right_dim));
        } else {
            return Err(Error::Eval(format!(
                "cannot broadcast tensor shapes {left:?} and {right:?}"
            )));
        }
    }
    Ok(out)
}

fn select_cell(tensor: &Tensor, coord: &[usize], result_shape: &[usize]) -> Result<Value> {
    let rank_gap = result_shape.len().saturating_sub(tensor.shape.len());
    let mut local = Vec::with_capacity(tensor.shape.len());
    for (axis, dim) in tensor.shape.iter().enumerate() {
        let result_axis = axis + rank_gap;
        let coord_value = coord
            .get(result_axis)
            .copied()
            .ok_or_else(|| Error::Eval("tensor broadcast axis mismatch".to_owned()))?;
        local.push(if *dim == 1 { 0 } else { coord_value });
    }
    let flat = Tensor::flat_offset(&tensor.shape, &local)?;
    tensor
        .data
        .get(flat)
        .cloned()
        .ok_or_else(|| Error::Eval("tensor broadcast selected an invalid cell".to_owned()))
}
