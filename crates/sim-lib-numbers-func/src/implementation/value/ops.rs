//! Arithmetic operator support for function values.

use std::sync::Arc;

use sim_kernel::{Cx, Error, Result, Symbol, Value, ValueNumberBinaryOp, ValueNumberUnaryOp};
use sim_lib_numbers_cas::{CasExpr, simplify_expr};

use super::super::domain::func_domain_symbol;
use super::{Func, FuncMetadata, NativeFn, SymbolicStatus, build_func_value};

pub(crate) fn register_value_ops(linker: &mut sim_kernel::Linker<'_>) {
    linker.value_number_binary_op(binary_op(
        Symbol::qualified("math", "add"),
        apply_add_func_op,
    ));
    linker.value_number_binary_op(binary_op(
        Symbol::qualified("math", "sub"),
        apply_sub_func_op,
    ));
    linker.value_number_binary_op(binary_op(
        Symbol::qualified("math", "mul"),
        apply_mul_func_op,
    ));
    linker.value_number_binary_op(binary_op(
        Symbol::qualified("math", "div"),
        apply_div_func_op,
    ));
    linker.value_number_binary_op(binary_op(
        Symbol::qualified("math", "pow"),
        apply_pow_func_op,
    ));
    linker.value_number_binary_op(binary_op(
        Symbol::qualified("math", "rem"),
        apply_rem_func_op,
    ));
    linker.value_number_unary_op(ValueNumberUnaryOp {
        operator: Symbol::qualified("math", "neg"),
        operand_domain: func_domain_symbol(),
        cost: 1,
        apply: apply_unary_func_op,
    });
}

fn binary_op(
    operator: Symbol,
    apply: fn(&mut Cx, Value, Value) -> Result<Value>,
) -> ValueNumberBinaryOp {
    ValueNumberBinaryOp {
        operator,
        left_domain: func_domain_symbol(),
        right_domain: func_domain_symbol(),
        cost: 1,
        apply,
    }
}

fn apply_add_func_op(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    apply_binary_func_op(cx, Symbol::qualified("math", "add"), left, right)
}

fn apply_sub_func_op(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    apply_binary_func_op(cx, Symbol::qualified("math", "sub"), left, right)
}

fn apply_mul_func_op(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    apply_binary_func_op(cx, Symbol::qualified("math", "mul"), left, right)
}

fn apply_div_func_op(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    apply_binary_func_op(cx, Symbol::qualified("math", "div"), left, right)
}

fn apply_pow_func_op(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    apply_binary_func_op(cx, Symbol::qualified("math", "pow"), left, right)
}

fn apply_rem_func_op(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    apply_binary_func_op(cx, Symbol::qualified("math", "rem"), left, right)
}

fn apply_binary_func_op(cx: &mut Cx, operator: Symbol, left: Value, right: Value) -> Result<Value> {
    let left_func = left
        .object()
        .downcast_ref::<Func>()
        .ok_or_else(|| Error::Eval("left operand was not a function value".to_owned()))?
        .clone();
    let right_func = right
        .object()
        .downcast_ref::<Func>()
        .ok_or_else(|| Error::Eval("right operand was not a function value".to_owned()))?
        .clone();
    let vars = union_vars(&left_func.vars, &right_func.vars);
    let closure_vars = vars.clone();
    let body_cas = match (left_func.body_cas(), right_func.body_cas()) {
        (Some(left_body), Some(right_body)) => Some(simplify_expr(
            cx,
            CasExpr::Op(
                operator.clone(),
                vec![left_body.clone(), right_body.clone()],
            ),
        )?),
        _ => None,
    };
    let native: NativeFn = Arc::new(move |cx: &mut Cx, args: &[Value]| {
        let left_args = project_args(&closure_vars, &left_func.vars, args)?;
        let right_args = project_args(&closure_vars, &right_func.vars, args)?;
        let left_value = left_func.invoke(cx, &left_args)?;
        let right_value = right_func.invoke(cx, &right_args)?;
        cx.apply_value_number_binary_op(&operator, left_value, right_value)
    });
    let func = match body_cas {
        Some(body_cas) => Func::dual_with(vars, body_cas, native, FuncMetadata::default()),
        None => Func::native_with_status(
            vars,
            native,
            FuncMetadata::default(),
            SymbolicStatus::mixed_native(),
        ),
    };
    build_func_value(cx, func)
}

fn apply_unary_func_op(cx: &mut Cx, value: Value) -> Result<Value> {
    let func = value
        .object()
        .downcast_ref::<Func>()
        .ok_or_else(|| Error::Eval("operand was not a function value".to_owned()))?
        .clone();
    let body_cas = func
        .body_cas()
        .cloned()
        .map(|body| {
            simplify_expr(
                cx,
                CasExpr::Op(Symbol::qualified("math", "neg"), vec![body]),
            )
        })
        .transpose()?;
    let native_func = func.clone();
    let native: NativeFn = Arc::new(move |cx: &mut Cx, args: &[Value]| {
        let out = native_func.invoke(cx, args)?;
        cx.apply_value_number_unary_op(&Symbol::qualified("math", "neg"), out)
    });
    let func = match body_cas {
        Some(body_cas) => {
            Func::dual_with(func.vars.clone(), body_cas, native, FuncMetadata::default())
        }
        None => Func::native(func.vars.clone(), native),
    };
    build_func_value(cx, func)
}

fn union_vars(left: &[Symbol], right: &[Symbol]) -> Vec<Symbol> {
    let mut vars = left.to_vec();
    for var in right {
        if !vars.contains(var) {
            vars.push(var.clone());
        }
    }
    vars
}

fn project_args(union: &[Symbol], target: &[Symbol], args: &[Value]) -> Result<Vec<Value>> {
    target
        .iter()
        .map(|var| {
            let index = union
                .iter()
                .position(|candidate| candidate == var)
                .ok_or_else(|| {
                    Error::Eval(format!(
                        "function variable {var} missing from projected call"
                    ))
                })?;
            args.get(index).cloned().ok_or_else(|| {
                Error::Eval(format!(
                    "function variable {var} missing from call arguments"
                ))
            })
        })
        .collect()
}
