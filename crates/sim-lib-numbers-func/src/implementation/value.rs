//! The `Func` function value: variables plus an optional CAS or native body,
//! with its metadata and arithmetic over function values.

use std::{any::Any, sync::Arc};

use sim_kernel::{
    Args, Callable, ClassRef, Cx, DefaultFactory, Error, Expr, Factory, NumberValue, Object,
    ObjectEncode, Result, ShapeRef, Symbol, Value, ValueNumberBinaryOp, ValueNumberUnaryOp,
};
use sim_lib_numbers_cas::{CasExpr, cas_expr_to_surface_expr, simplify_expr};
use sim_lib_numbers_cas_eval::eval_cas;
use sim_shape::{AnyShape, ListShape, Shape, shape_value};

use super::domain::{func_class_symbol, func_domain_symbol};
use super::function::{child_env_with_args, vars_expr};

/// A native (Rust-backed) function body: a closure invoked with the runtime
/// context and the evaluated argument values, used when a `Func` has no CAS body.
pub type NativeFn = Arc<dyn Fn(&mut Cx, &[Value]) -> Result<Value> + Send + Sync>;

/// Out-of-band annotations attached to a [`Func`] value.
#[derive(Clone, Default)]
pub struct FuncMetadata {
    /// Symbol identifying where this function came from (for example, an
    /// elementary-function name), when known.
    pub source: Option<Symbol>,
    /// Optional hint naming the differentiator that should handle `grad`/`diff`
    /// for this function.
    pub differentiator_hint: Option<Symbol>,
    /// Arbitrary caller-supplied value carried alongside the function.
    pub payload: Option<Value>,
}

/// A callable function value in the `Func` number domain: its bound variables
/// plus an optional symbolic (CAS) body and/or native body.
#[derive(Clone)]
pub struct Func {
    /// The ordered parameter symbols bound when the function is invoked.
    pub vars: Vec<Symbol>,
    /// The symbolic body, when the function can be expressed as a CAS expression.
    pub body_cas: Option<CasExpr>,
    /// The native body, used when no symbolic body is available.
    pub body_native: Option<NativeFn>,
    /// Out-of-band metadata describing the function.
    pub metadata: FuncMetadata,
}

impl Func {
    /// Builds a function from its variables, optional symbolic and native
    /// bodies, and metadata.
    pub fn new(
        vars: Vec<Symbol>,
        body_cas: Option<CasExpr>,
        body_native: Option<NativeFn>,
        metadata: FuncMetadata,
    ) -> Self {
        Self {
            vars,
            body_cas,
            body_native,
            metadata,
        }
    }

    /// Builds a function with a symbolic (CAS) body and default metadata.
    pub fn symbolic(vars: Vec<Symbol>, body_cas: CasExpr) -> Self {
        Self::symbolic_with(vars, body_cas, FuncMetadata::default())
    }

    /// Builds a function with a symbolic (CAS) body and caller-supplied metadata.
    pub fn symbolic_with(vars: Vec<Symbol>, body_cas: CasExpr, metadata: FuncMetadata) -> Self {
        Self::new(vars, Some(body_cas), None, metadata)
    }

    /// Builds a function with a native (Rust closure) body and default metadata.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use sim_kernel::Symbol;
    /// use sim_lib_numbers_func::Func;
    ///
    /// let func = Func::native(
    ///     vec![Symbol::new("x")],
    ///     Arc::new(|_cx, args| Ok(args[0].clone())),
    /// );
    /// assert_eq!(func.vars, vec![Symbol::new("x")]);
    /// assert!(func.body_cas.is_none());
    /// assert!(func.body_native.is_some());
    /// ```
    pub fn native(vars: Vec<Symbol>, body_native: NativeFn) -> Self {
        Self::native_with(vars, body_native, FuncMetadata::default())
    }

    /// Builds a function with a native body and caller-supplied metadata.
    pub fn native_with(vars: Vec<Symbol>, body_native: NativeFn, metadata: FuncMetadata) -> Self {
        Self::new(vars, None, Some(body_native), metadata)
    }

    /// Returns the symbolic body advertised by this function, when available.
    pub fn body_cas(&self) -> Option<&CasExpr> {
        self.body_cas.as_ref()
    }

    fn invoke(&self, cx: &mut Cx, args: &[Value]) -> Result<Value> {
        if args.len() != self.vars.len() {
            return Err(Error::Eval(format!(
                "function expected {} arguments but received {}",
                self.vars.len(),
                args.len()
            )));
        }
        if let Some(body_native) = &self.body_native {
            return body_native(cx, args);
        }
        let Some(body_cas) = &self.body_cas else {
            return Err(Error::Eval(
                "function has neither symbolic nor native body".to_owned(),
            ));
        };
        let env = child_env_with_args(cx.env(), &self.vars, args)?;
        cx.with_env(env.clone(), |cx| eval_cas(cx, body_cas, &env))
    }
}

impl Object for Func {
    fn display(&self, cx: &mut Cx) -> Result<String> {
        if let Some(body_cas) = &self.body_cas {
            return Ok(format!(
                "#<func {:?} -> {:?}>",
                self.vars,
                cas_expr_to_surface_expr(cx, body_cas)?
            ));
        }
        Ok(format!("#<native-func {:?}>", self.vars))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl sim_kernel::ObjectCompat for Func {
    fn class(&self, cx: &mut Cx) -> Result<ClassRef> {
        if let Some(value) = cx.registry().class_by_symbol(&func_class_symbol()) {
            return Ok(value.clone());
        }
        DefaultFactory.class_stub(
            sim_kernel::CORE_NUMBER_CLASS_ID,
            Symbol::qualified("core", "Number"),
        )
    }
    fn as_expr(&self, cx: &mut Cx) -> Result<Expr> {
        let Some(body_cas) = &self.body_cas else {
            return Ok(Expr::Extension {
                tag: func_class_symbol(),
                payload: Box::new(Expr::String("#<native-func>".to_owned())),
            });
        };
        Ok(Expr::Call {
            operator: Box::new(Expr::Symbol(Symbol::new("fn"))),
            args: vec![
                vars_expr(&self.vars),
                cas_expr_to_surface_expr(cx, body_cas)?,
            ],
        })
    }
    fn as_table(&self, cx: &mut Cx) -> Result<Value> {
        let vars = cx.factory().list(
            self.vars
                .iter()
                .cloned()
                .map(|var| cx.factory().symbol(var))
                .collect::<Result<Vec<_>>>()?,
        )?;
        let body_expr = self
            .body_cas
            .as_ref()
            .map(|body| cas_expr_to_surface_expr(cx, body))
            .transpose()?;
        let body = match &self.body_cas {
            Some(_) => cx
                .factory()
                .expr(body_expr.expect("body expr should exist when body_cas is present"))?,
            None => cx.factory().nil()?,
        };
        let native = cx.factory().bool(self.body_native.is_some())?;
        cx.factory().table(vec![
            (Symbol::new("kind"), cx.factory().string("func".to_owned())?),
            (Symbol::new("vars"), vars),
            (Symbol::new("body"), body),
            (Symbol::new("native"), native),
        ])
    }
    fn as_callable(&self) -> Option<&dyn Callable> {
        Some(self)
    }
    fn as_number_value(&self) -> Option<&dyn NumberValue> {
        Some(self)
    }
    fn as_object_encoder(&self) -> Option<&dyn ObjectEncode> {
        Some(self)
    }
}

impl Callable for Func {
    fn call(&self, cx: &mut Cx, args: Args) -> Result<Value> {
        self.invoke(cx, args.values())
    }

    fn browse_args_shape(&self, _cx: &mut Cx) -> Result<Option<ShapeRef>> {
        let items = self
            .vars
            .iter()
            .map(|_| Arc::new(AnyShape) as Arc<dyn Shape>)
            .collect();
        Ok(Some(shape_value(
            Symbol::qualified(func_class_symbol().to_string(), "args"),
            Arc::new(ListShape::new(items)),
        )))
    }
}

impl NumberValue for Func {
    fn number_domain(&self, _cx: &mut Cx) -> Result<Symbol> {
        Ok(func_domain_symbol())
    }
}

impl sim_citizen::Citizen for Func {
    fn citizen_symbol() -> Symbol {
        func_class_symbol()
    }

    fn citizen_version() -> u32 {
        0
    }

    fn citizen_arity() -> usize {
        2
    }

    fn citizen_fields() -> &'static [&'static str] {
        &["vars", "body"]
    }
}

/// Wraps a [`Func`] into a runtime [`Value`] in the `Func` number domain.
pub fn build_func_value(cx: &mut Cx, func: Func) -> Result<Value> {
    cx.factory().opaque(Arc::new(func))
}

pub(crate) fn build_constant_func_value(cx: &mut Cx, value: Value) -> Result<Value> {
    build_func_value(
        cx,
        Func::new(
            Vec::new(),
            Some(CasExpr::Num(value)),
            None,
            FuncMetadata::default(),
        ),
    )
}

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
    let body_cas = match (&left_func.body_cas, &right_func.body_cas) {
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
    let body_native = body_cas.is_none().then_some(native);
    build_func_value(
        cx,
        Func::new(vars, body_cas, body_native, FuncMetadata::default()),
    )
}

fn apply_unary_func_op(cx: &mut Cx, value: Value) -> Result<Value> {
    let func = value
        .object()
        .downcast_ref::<Func>()
        .ok_or_else(|| Error::Eval("operand was not a function value".to_owned()))?
        .clone();
    let body_cas = func
        .body_cas
        .clone()
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
    let body_native = body_cas.is_none().then_some(native);
    build_func_value(
        cx,
        Func::new(
            func.vars.clone(),
            body_cas,
            body_native,
            FuncMetadata::default(),
        ),
    )
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
