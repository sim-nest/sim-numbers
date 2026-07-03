//! The linear-algebra operation implementations: dispatch plus the concrete
//! `dot`, `matmul`, `det`, `inv`, and related routines over tensor values.

use sim_kernel::{DefaultFactory, Error, Factory, Result, Symbol, Value};
use sim_lib_numbers_core::domains;
use sim_lib_numbers_tensor::{Tensor, build_tensor_value, tensor_dtype};

use super::support::{
    add, div, element_count, expect_matrix, expect_tensor, expect_vector, extract_optional_symbol,
    extract_shape, extract_usize, i64_number, mul, neg, pow, sub,
};

pub fn dispatch(cx: &mut sim_kernel::Cx, symbol: &Symbol, values: Vec<Value>) -> Result<Value> {
    if *symbol == Symbol::new("dot") {
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
    } else {
        Err(Error::Eval(format!(
            "unsupported tensor linalg function {symbol}"
        )))
    }
}

fn dot(cx: &mut sim_kernel::Cx, values: &[Value]) -> Result<Value> {
    let [left_value, right_value] = values else {
        return Err(Error::Eval(
            "dot expects exactly two vector tensors".to_owned(),
        ));
    };
    let left = expect_vector(left_value)?;
    let right = expect_vector(right_value)?;
    if left.shape != right.shape {
        return Err(Error::Eval(
            "dot expects vectors with matching lengths".to_owned(),
        ));
    }
    sum_products(
        cx,
        left.data
            .iter()
            .cloned()
            .zip(right.data.iter().cloned())
            .collect(),
    )
}

fn matmul(cx: &mut sim_kernel::Cx, values: &[Value]) -> Result<Value> {
    let [left_value, right_value] = values else {
        return Err(Error::Eval(
            "matmul expects exactly two tensor arguments".to_owned(),
        ));
    };
    let left = expect_tensor(left_value)?;
    let right = expect_tensor(right_value)?;
    match (left.shape.as_slice(), right.shape.as_slice()) {
        ([m], [n]) => {
            if m != n {
                return Err(Error::Eval("matmul vector lengths must match".to_owned()));
            }
            dot(cx, values)
        }
        ([rows, inner_left], [inner_right, cols]) => {
            if inner_left != inner_right {
                return Err(Error::Eval("matmul inner dimensions must match".to_owned()));
            }
            let mut out = Vec::with_capacity(rows * cols);
            for row in 0..*rows {
                for col in 0..*cols {
                    let mut terms = Vec::with_capacity(*inner_left);
                    for inner in 0..*inner_left {
                        let left_cell = left.data[row * inner_left + inner].clone();
                        let right_cell = right.data[inner * cols + col].clone();
                        terms.push((left_cell, right_cell));
                    }
                    out.push(sum_products(cx, terms)?);
                }
            }
            build_tensor_value(cx, vec![*rows, *cols], None, out)
        }
        ([rows, inner_left], [inner_right]) => {
            if inner_left != inner_right {
                return Err(Error::Eval("matmul inner dimensions must match".to_owned()));
            }
            let mut out = Vec::with_capacity(*rows);
            for row in 0..*rows {
                let mut terms = Vec::with_capacity(*inner_left);
                for inner in 0..*inner_left {
                    let left_cell = left.data[row * inner_left + inner].clone();
                    let right_cell = right.data[inner].clone();
                    terms.push((left_cell, right_cell));
                }
                out.push(sum_products(cx, terms)?);
            }
            build_tensor_value(cx, vec![*rows], None, out)
        }
        ([inner_left], [inner_right, cols]) => {
            if inner_left != inner_right {
                return Err(Error::Eval("matmul inner dimensions must match".to_owned()));
            }
            let mut out = Vec::with_capacity(*cols);
            for col in 0..*cols {
                let mut terms = Vec::with_capacity(*inner_left);
                for inner in 0..*inner_left {
                    let left_cell = left.data[inner].clone();
                    let right_cell = right.data[inner * cols + col].clone();
                    terms.push((left_cell, right_cell));
                }
                out.push(sum_products(cx, terms)?);
            }
            build_tensor_value(cx, vec![*cols], None, out)
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
    if left.data.len() != 3 || right.data.len() != 3 {
        return Err(Error::Eval("cross expects 3-vectors".to_owned()));
    }
    let a = &left.data;
    let b = &right.data;
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
    let rows = tensor.shape[0];
    let cols = tensor.shape[1];
    let mut out = Vec::with_capacity(tensor.data.len());
    for col in 0..cols {
        for row in 0..rows {
            out.push(tensor.data[row * cols + col].clone());
        }
    }
    build_tensor_value(
        cx,
        vec![cols, rows],
        Some(tensor_dtype(tensor).clone()),
        out,
    )
}

fn det(cx: &mut sim_kernel::Cx, values: &[Value]) -> Result<Value> {
    let [value] = values else {
        return Err(Error::Eval(
            "det expects exactly one matrix tensor".to_owned(),
        ));
    };
    let tensor = expect_matrix(value)?;
    if tensor.shape[0] != tensor.shape[1] {
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
    match tensor.shape.as_slice() {
        [1, 1] => {
            let denom = tensor.data[0].clone();
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
            let a = tensor.data[0].clone();
            let b = tensor.data[1].clone();
            let c = tensor.data[2].clone();
            let d = tensor.data[3].clone();
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
    if tensor.shape[0] != tensor.shape[1] {
        return Err(Error::Eval("trace expects a square matrix".to_owned()));
    }
    let cols = tensor.shape[1];
    if tensor.shape[0] == 0 {
        return i64_number(0);
    }
    let mut acc = tensor.data[0].clone();
    for row in 1..tensor.shape[0] {
        acc = add(cx, acc, tensor.data[row * cols + row].clone())?;
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
    let mut acc = i64_number(0)?;
    for cell in &tensor.data {
        let square = mul(cx, cell.clone(), cell.clone())?;
        acc = add(cx, acc, square)?;
    }
    let half = DefaultFactory.number_literal(domains::rational(), "1/2".to_owned())?;
    pow(cx, acc, half)
}

fn eye(cx: &mut sim_kernel::Cx, values: &[Value]) -> Result<Value> {
    let [value] = values else {
        return Err(Error::Eval(
            "eye expects exactly one size argument".to_owned(),
        ));
    };
    let n = extract_usize(value, "eye size")?;
    let mut cells = Vec::with_capacity(n * n);
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
    let size = element_count(&shape);
    build_tensor_value(cx, shape, dtype, vec![cell; size])
}

fn determinant(cx: &mut sim_kernel::Cx, tensor: &Tensor) -> Result<Value> {
    let n = tensor.shape[0];
    match n {
        0 => i64_number(1),
        1 => Ok(tensor.data[0].clone()),
        2 => {
            let ad = mul(cx, tensor.data[0].clone(), tensor.data[3].clone())?;
            let bc = mul(cx, tensor.data[1].clone(), tensor.data[2].clone())?;
            sub(cx, ad, bc)
        }
        _ => {
            let mut acc = None;
            for col in 0..n {
                let sign = if col % 2 == 0 {
                    i64_number(1)?
                } else {
                    i64_number(-1)?
                };
                let factor = mul(cx, sign, tensor.data[col].clone())?;
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
    }
}

fn sum_products(cx: &mut sim_kernel::Cx, terms: Vec<(Value, Value)>) -> Result<Value> {
    let mut terms = terms.into_iter();
    let Some((first_left, first_right)) = terms.next() else {
        return i64_number(0);
    };
    let mut acc = mul(cx, first_left, first_right)?;
    for (left, right) in terms {
        let product = mul(cx, left, right)?;
        acc = add(cx, acc, product)?;
    }
    Ok(acc)
}

fn minor_tensor(
    _cx: &mut sim_kernel::Cx,
    tensor: &Tensor,
    skip_row: usize,
    skip_col: usize,
) -> Result<Tensor> {
    let n = tensor.shape[0];
    let mut data = Vec::with_capacity((n - 1) * (n - 1));
    for row in 0..n {
        if row == skip_row {
            continue;
        }
        for col in 0..n {
            if col == skip_col {
                continue;
            }
            data.push(tensor.data[row * n + col].clone());
        }
    }
    Ok(Tensor {
        shape: vec![n - 1, n - 1],
        dtype: tensor.dtype.clone(),
        data,
    })
}
