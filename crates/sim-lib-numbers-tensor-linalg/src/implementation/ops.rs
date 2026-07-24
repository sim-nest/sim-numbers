//! The linear-algebra operation implementations: dispatch plus the concrete
//! `dot`, `matmul`, `det`, `inv`, and related routines over tensor values.

use sim_kernel::{Error, Result, Symbol, Value};
use sim_lib_numbers_tensor::{
    Tensor, build_tensor_value, cos_op_symbol, execute_tensor_dot, execute_tensor_matmul,
    execute_tensor_norm, execute_tensor_reduction, execute_tensor_transcendental,
    execute_tensor_transpose, exp_op_symbol, max_op_symbol, min_op_symbol, sin_op_symbol,
    sqrt_op_symbol, sum_op_symbol, tensor_dtype,
};

use super::support::{
    add, bounded_element_count, div, expect_matrix, expect_tensor, expect_vector,
    extract_optional_symbol, extract_shape, extract_usize, i64_number, mul, neg, sub,
};

pub fn dispatch(cx: &mut sim_kernel::Cx, symbol: &Symbol, values: Vec<Value>) -> Result<Value> {
    if *symbol == Symbol::new("sum") {
        reduction(cx, &values, sum_op_symbol())
    } else if *symbol == Symbol::new("min") {
        reduction(cx, &values, min_op_symbol())
    } else if *symbol == Symbol::new("max") {
        reduction(cx, &values, max_op_symbol())
    } else if *symbol == Symbol::new("dot") {
        dot(cx, &values)
    } else if *symbol == Symbol::new("matmul") {
        matmul(cx, &values)
    } else if *symbol == Symbol::new("cross") {
        cross(cx, &values)
    } else if *symbol == Symbol::new("transpose") {
        transpose(cx, &values)
    } else if *symbol == Symbol::new("det") {
        det(cx, &values)
    } else if *symbol == Symbol::new("inv") {
        inv(cx, &values)
    } else if *symbol == Symbol::new("trace") {
        trace(cx, &values)
    } else if *symbol == Symbol::new("norm") {
        norm(cx, &values)
    } else if *symbol == Symbol::new("eye") {
        eye(cx, &values)
    } else if *symbol == Symbol::new("zeros") {
        zeros(cx, &values)
    } else if *symbol == Symbol::new("ones") {
        ones(cx, &values)
    } else if *symbol == Symbol::new("sqrt") {
        transcendental(cx, &values, sqrt_op_symbol())
    } else if *symbol == Symbol::new("exp") {
        transcendental(cx, &values, exp_op_symbol())
    } else if *symbol == Symbol::new("sin") {
        transcendental(cx, &values, sin_op_symbol())
    } else if *symbol == Symbol::new("cos") {
        transcendental(cx, &values, cos_op_symbol())
    } else {
        Err(Error::Eval(format!(
            "unsupported tensor linalg function {symbol}"
        )))
    }
}

fn reduction(cx: &mut sim_kernel::Cx, values: &[Value], operator: Symbol) -> Result<Value> {
    let [value] = values else {
        return Err(Error::Eval(
            "tensor reduction expects exactly one tensor".to_owned(),
        ));
    };
    let tensor = expect_tensor(value)?;
    scalar_tensor_cell(&execute_tensor_reduction(cx, operator, tensor)?)
}

fn transcendental(cx: &mut sim_kernel::Cx, values: &[Value], operator: Symbol) -> Result<Value> {
    let [value] = values else {
        return Err(Error::Eval(
            "tensor transcendental expects exactly one tensor".to_owned(),
        ));
    };
    let tensor = expect_tensor(value)?;
    let out = execute_tensor_transcendental(cx, operator, tensor)?;
    build_tensor_value(
        cx,
        out.shape().to_vec(),
        Some(out.dtype().clone()),
        out.cells()?.to_vec(),
    )
}

fn dot(cx: &mut sim_kernel::Cx, values: &[Value]) -> Result<Value> {
    let [left_value, right_value] = values else {
        return Err(Error::Eval(
            "dot expects exactly two vector tensors".to_owned(),
        ));
    };
    let left = expect_vector(left_value)?;
    let right = expect_vector(right_value)?;
    if left.shape() != right.shape() {
        return Err(Error::Eval(
            "dot expects vectors with matching lengths".to_owned(),
        ));
    }
    scalar_tensor_cell(&execute_tensor_dot(cx, left, right)?)
}

fn matmul(cx: &mut sim_kernel::Cx, values: &[Value]) -> Result<Value> {
    let [left_value, right_value] = values else {
        return Err(Error::Eval(
            "matmul expects exactly two tensor arguments".to_owned(),
        ));
    };
    let left = expect_tensor(left_value)?;
    let right = expect_tensor(right_value)?;
    match (left.shape(), right.shape()) {
        ([m], [n]) => {
            if m != n {
                return Err(Error::Eval("matmul vector lengths must match".to_owned()));
            }
            scalar_tensor_cell(&execute_tensor_matmul(cx, left, right)?)
        }
        ([_, _], [_, _]) | ([_, _], [_]) | ([_], [_, _]) => {
            let tensor = execute_tensor_matmul(cx, left, right)?;
            build_tensor_value(
                cx,
                tensor.shape().to_vec(),
                Some(tensor.dtype().clone()),
                tensor.cells()?.to_vec(),
            )
        }
        _ => Err(Error::Eval(
            "matmul currently supports rank-1 and rank-2 tensors only".to_owned(),
        )),
    }
}

fn cross(cx: &mut sim_kernel::Cx, values: &[Value]) -> Result<Value> {
    let [left_value, right_value] = values else {
        return Err(Error::Eval(
            "cross expects exactly two vector tensors".to_owned(),
        ));
    };
    let left = expect_vector(left_value)?;
    let right = expect_vector(right_value)?;
    if left.len() != 3 || right.len() != 3 {
        return Err(Error::Eval("cross expects 3-vectors".to_owned()));
    }
    let a = left.cells()?;
    let b = right.cells()?;
    let c0_left = mul(cx, a[1].clone(), b[2].clone())?;
    let c0_right = mul(cx, a[2].clone(), b[1].clone())?;
    let c1_left = mul(cx, a[2].clone(), b[0].clone())?;
    let c1_right = mul(cx, a[0].clone(), b[2].clone())?;
    let c2_left = mul(cx, a[0].clone(), b[1].clone())?;
    let c2_right = mul(cx, a[1].clone(), b[0].clone())?;
    let cells = vec![
        sub(cx, c0_left, c0_right)?,
        sub(cx, c1_left, c1_right)?,
        sub(cx, c2_left, c2_right)?,
    ];
    build_tensor_value(cx, vec![3], None, cells)
}

fn transpose(cx: &mut sim_kernel::Cx, values: &[Value]) -> Result<Value> {
    let [value] = values else {
        return Err(Error::Eval(
            "transpose expects exactly one tensor".to_owned(),
        ));
    };
    let tensor = expect_matrix(value)?;
    let rows = tensor.shape()[0];
    let cols = tensor.shape()[1];
    let out = execute_tensor_transpose(cx, tensor)?;
    build_tensor_value(
        cx,
        vec![cols, rows],
        Some(out.dtype().clone()),
        out.cells()?.to_vec(),
    )
}

fn det(cx: &mut sim_kernel::Cx, values: &[Value]) -> Result<Value> {
    let [value] = values else {
        return Err(Error::Eval(
            "det expects exactly one matrix tensor".to_owned(),
        ));
    };
    let tensor = expect_matrix(value)?;
    if tensor.shape()[0] != tensor.shape()[1] {
        return Err(Error::Eval("det expects a square matrix".to_owned()));
    }
    determinant(cx, tensor)
}

fn inv(cx: &mut sim_kernel::Cx, values: &[Value]) -> Result<Value> {
    let [value] = values else {
        return Err(Error::Eval(
            "inv expects exactly one matrix tensor".to_owned(),
        ));
    };
    let tensor = expect_matrix(value)?;
    match tensor.shape() {
        [1, 1] => {
            let denom = tensor.cell(0)?;
            let one = i64_number(1)?;
            let value = div(cx, one, denom)?;
            build_tensor_value(
                cx,
                vec![1, 1],
                Some(tensor_dtype(tensor).clone()),
                vec![value],
            )
        }
        [2, 2] => {
            let det_value = determinant(cx, tensor)?;
            let one_over_det = div(cx, i64_number(1)?, det_value)?;
            let a = tensor.cell(0)?;
            let b = tensor.cell(1)?;
            let c = tensor.cell(2)?;
            let d = tensor.cell(3)?;
            let minus_b = neg(cx, b)?;
            let minus_c = neg(cx, c)?;
            let cells = vec![
                mul(cx, d, one_over_det.clone())?,
                mul(cx, minus_b, one_over_det.clone())?,
                mul(cx, minus_c, one_over_det.clone())?,
                mul(cx, a, one_over_det)?,
            ];
            build_tensor_value(cx, vec![2, 2], Some(tensor_dtype(tensor).clone()), cells)
        }
        _ => Err(Error::Eval(
            "inv currently supports 1x1 and 2x2 matrices only".to_owned(),
        )),
    }
}

fn trace(cx: &mut sim_kernel::Cx, values: &[Value]) -> Result<Value> {
    let [value] = values else {
        return Err(Error::Eval(
            "trace expects exactly one matrix tensor".to_owned(),
        ));
    };
    let tensor = expect_matrix(value)?;
    if tensor.shape()[0] != tensor.shape()[1] {
        return Err(Error::Eval("trace expects a square matrix".to_owned()));
    }
    let cols = tensor.shape()[1];
    if tensor.shape()[0] == 0 {
        return i64_number(0);
    }
    let mut acc = tensor.cell(0)?;
    for row in 1..tensor.shape()[0] {
        acc = add(cx, acc, tensor.cell(row * cols + row)?)?;
    }
    Ok(acc)
}

fn norm(cx: &mut sim_kernel::Cx, values: &[Value]) -> Result<Value> {
    let (tensor_value, ord) = match values {
        [value] => (value, 2usize),
        [value, ord] => (value, extract_usize(ord, "norm ord")?),
        _ => {
            return Err(Error::Eval(
                "norm expects a tensor and an optional positive ord".to_owned(),
            ));
        }
    };
    if ord != 2 {
        return Err(Error::Eval(
            "norm currently supports only Euclidean ord 2".to_owned(),
        ));
    }
    let tensor = expect_tensor(tensor_value)?;
    scalar_tensor_cell(&execute_tensor_norm(cx, tensor)?)
}

fn eye(cx: &mut sim_kernel::Cx, values: &[Value]) -> Result<Value> {
    let [value] = values else {
        return Err(Error::Eval(
            "eye expects exactly one size argument".to_owned(),
        ));
    };
    let n = extract_usize(value, "eye size")?;
    // Bound n*n before allocating: a hostile size overflows the capacity
    // argument or OOMs long before the identity is built.
    let count = bounded_element_count(&[n, n])?;
    let mut cells = Vec::with_capacity(count);
    for row in 0..n {
        for col in 0..n {
            cells.push(if row == col {
                i64_number(1)?
            } else {
                i64_number(0)?
            });
        }
    }
    build_tensor_value(cx, vec![n, n], None, cells)
}

fn zeros(cx: &mut sim_kernel::Cx, values: &[Value]) -> Result<Value> {
    fill_tensor(cx, values, false)
}

fn ones(cx: &mut sim_kernel::Cx, values: &[Value]) -> Result<Value> {
    fill_tensor(cx, values, true)
}

fn fill_tensor(cx: &mut sim_kernel::Cx, values: &[Value], ones: bool) -> Result<Value> {
    let (shape_value, dtype) = match values {
        [shape] => (shape, None),
        [shape, dtype] => (shape, extract_optional_symbol(cx, dtype)?),
        _ => {
            return Err(Error::Eval(
                "zeros/ones expect shape and optional dtype".to_owned(),
            ));
        }
    };
    let shape = extract_shape(cx, shape_value)?;
    let cell = if ones { i64_number(1)? } else { i64_number(0)? };
    // Bound the cell count before allocating: zeros(["1000000000000"]) parses to
    // a shape whose product fits in usize but would OOM if filled.
    let size = bounded_element_count(&shape)?;
    build_tensor_value(cx, shape, dtype, vec![cell; size])
}

/// Above this order the determinant switches from Laplace cofactor expansion
/// (O(n!)) to fraction-free Gaussian elimination (O(n^3)). Cofactor stays cheap
/// and allocation-light for tiny matrices; a 7x7 cofactor expansion already
/// costs 5040 recursive minors and grows factorially, so anything larger must
/// take the elimination path or it hangs.
const DET_COFACTOR_MAX: usize = 6;

fn determinant(cx: &mut sim_kernel::Cx, tensor: &Tensor) -> Result<Value> {
    let n = tensor.shape()[0];
    match n {
        0 => i64_number(1),
        1 => tensor.cell(0),
        2 => {
            let ad = mul(cx, tensor.cell(0)?, tensor.cell(3)?)?;
            let bc = mul(cx, tensor.cell(1)?, tensor.cell(2)?)?;
            sub(cx, ad, bc)
        }
        _ if n <= DET_COFACTOR_MAX => determinant_cofactor(cx, tensor),
        _ => determinant_bareiss(cx, tensor),
    }
}

fn determinant_cofactor(cx: &mut sim_kernel::Cx, tensor: &Tensor) -> Result<Value> {
    let n = tensor.shape()[0];
    let mut acc = None;
    for col in 0..n {
        let sign = if col % 2 == 0 {
            i64_number(1)?
        } else {
            i64_number(-1)?
        };
        let factor = mul(cx, sign, tensor.cell(col)?)?;
        let minor = minor_tensor(cx, tensor, 0, col)?;
        let subdet = determinant(cx, &minor)?;
        let term = mul(cx, factor, subdet)?;
        acc = Some(match acc {
            Some(current) => add(cx, current, term)?,
            None => term,
        });
    }
    Ok(acc.unwrap_or(i64_number(0)?))
}

/// Bareiss fraction-free Gaussian elimination: an O(n^3) determinant that keeps
/// large matrices tractable where cofactor expansion would hang.
///
/// Every division in the recurrence
/// `M[i][j] <- (M[i][j]*pivot - M[i][k]*M[k][j]) / prev_pivot` is exact -- Bareiss
/// guarantees a zero remainder -- so on an exact element domain (integers,
/// rationals) the result stays exact and integer-valued, and even a truncating
/// integer division would land on the right answer. Floating-point element
/// domains follow the ordinary numeric path. Row swaps to reach a nonzero pivot
/// flip the sign; an all-zero pivot column means the matrix is singular
/// (determinant 0).
fn determinant_bareiss(cx: &mut sim_kernel::Cx, tensor: &Tensor) -> Result<Value> {
    let n = tensor.shape()[0];
    let mut m = tensor.cells()?.to_vec();
    let at = |row: usize, col: usize| row * n + col;
    let mut prev = i64_number(1)?;
    let mut negate = false;
    for k in 0..n - 1 {
        if cell_is_zero(cx, &m[at(k, k)])? {
            let mut pivot_row = None;
            for row in (k + 1)..n {
                if !cell_is_zero(cx, &m[at(row, k)])? {
                    pivot_row = Some(row);
                    break;
                }
            }
            let Some(row) = pivot_row else {
                return i64_number(0);
            };
            for col in 0..n {
                m.swap(at(k, col), at(row, col));
            }
            negate = !negate;
        }
        let pivot = m[at(k, k)].clone();
        for i in (k + 1)..n {
            for j in (k + 1)..n {
                let scaled = mul(cx, m[at(i, j)].clone(), pivot.clone())?;
                let cross = mul(cx, m[at(i, k)].clone(), m[at(k, j)].clone())?;
                let numerator = sub(cx, scaled, cross)?;
                m[at(i, j)] = div(cx, numerator, prev.clone())?;
            }
            m[at(i, k)] = i64_number(0)?;
        }
        prev = pivot;
    }
    let det = m[at(n - 1, n - 1)].clone();
    if negate { neg(cx, det) } else { Ok(det) }
}

/// True when a scalar tensor cell is exactly the number zero. Used only to find
/// a usable Bareiss pivot; an unrecognized canonical form is treated as nonzero
/// so a real pivot is never skipped.
fn cell_is_zero(cx: &mut sim_kernel::Cx, value: &Value) -> Result<bool> {
    match value.object().as_expr(cx)? {
        sim_kernel::Expr::Number(literal) => Ok(number_canonical_is_zero(&literal.canonical)),
        _ => Ok(false),
    }
}

fn number_canonical_is_zero(canonical: &str) -> bool {
    let text = canonical.trim();
    if let Ok(value) = text.parse::<f64>() {
        return value == 0.0;
    }
    // Rational "num/den" is zero exactly when the numerator is zero.
    if let Some((num, _)) = text.split_once('/') {
        return num.trim().parse::<f64>().map(|v| v == 0.0).unwrap_or(false);
    }
    false
}

fn scalar_tensor_cell(tensor: &Tensor) -> Result<Value> {
    if tensor.rank() != 0 {
        return Err(Error::Eval(
            "executor returned a non-scalar tensor where a scalar result was expected".to_owned(),
        ));
    }
    tensor.cell(0)
}

fn minor_tensor(
    cx: &mut sim_kernel::Cx,
    tensor: &Tensor,
    skip_row: usize,
    skip_col: usize,
) -> Result<Tensor> {
    let n = tensor.shape()[0];
    let mut data = Vec::with_capacity((n - 1) * (n - 1));
    for row in 0..n {
        if row == skip_row {
            continue;
        }
        for col in 0..n {
            if col == skip_col {
                continue;
            }
            data.push(tensor.cell(row * n + col)?);
        }
    }
    Tensor::new_checked(cx, vec![n - 1, n - 1], tensor.dtype().clone(), data)
}
