//! Function operations: the `fn`, `call`, and `grad` callables and the
//! function class builder backing the `Func` domain.

use std::{any::Any, sync::Arc};

use sim_kernel::{
    Args, Callable, Class, ClassId, ClassRef, Cx, DefaultFactory, Env, Error, Expr, Factory,
    Object, ObjectEncode, ObjectEncoding, ReadConstructor, ReadConstructorRef, Result, ShapeRef,
    Symbol, TableRef, Value,
};
use sim_lib_numbers_cas::{cas_expr_to_surface_expr, expr_to_cas_expr};

use super::domain::{func_class_symbol, value_shape_symbol};
use super::value::{Func, FuncMetadata, build_func_value};

/// Returns the symbol bound to the `fn` function-builder callable.
///
/// # Examples
///
/// ```
/// use sim_lib_numbers_func::{call_symbol, fn_symbol, grad_symbol};
///
/// assert_eq!(fn_symbol().to_string(), "fn");
/// assert_eq!(call_symbol().to_string(), "call");
/// assert_eq!(grad_symbol().to_string(), "grad");
/// ```
pub fn fn_symbol() -> Symbol {
    Symbol::new("fn")
}

/// Returns the symbol bound to the `call` apply-a-function callable.
pub fn call_symbol() -> Symbol {
    Symbol::new("call")
}

/// Returns the symbol bound to the `grad` gradient-of-a-function callable.
pub fn grad_symbol() -> Symbol {
    Symbol::new("grad")
}

#[derive(Clone)]
pub struct FnBuilder;

impl Object for FnBuilder {
    fn display(&self, _cx: &mut Cx) -> Result<String> {
        Ok("#<function fn>".to_owned())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl sim_kernel::ObjectCompat for FnBuilder {
    fn class(&self, cx: &mut Cx) -> Result<ClassRef> {
        function_class(cx)
    }
    fn as_expr(&self, _cx: &mut Cx) -> Result<Expr> {
        Ok(Expr::Symbol(fn_symbol()))
    }
    fn as_callable(&self) -> Option<&dyn Callable> {
        Some(self)
    }
}

impl Callable for FnBuilder {
    fn call(&self, _cx: &mut Cx, _args: Args) -> Result<Value> {
        Err(Error::Eval(
            "fn must be called with unevaluated parameters and a body".to_owned(),
        ))
    }

    fn call_exprs(&self, cx: &mut Cx, args: sim_kernel::RawArgs) -> Result<Value> {
        let args = args.into_exprs();
        let [vars_expr, body_expr] = args.as_slice() else {
            return Err(Error::Eval(
                "fn expects exactly a parameter list and one body expression".to_owned(),
            ));
        };
        let vars = parse_vars_expr(vars_expr)?;
        let body_cas = expr_to_cas_expr(cx, body_expr)?
            .ok_or_else(|| Error::Eval("fn body must be CAS-compatible".to_owned()))?;
        build_func_value(
            cx,
            Func::new(vars, Some(body_cas), None, FuncMetadata::default()),
        )
    }
}

#[derive(Clone)]
pub struct CallFunction;

impl Object for CallFunction {
    fn display(&self, _cx: &mut Cx) -> Result<String> {
        Ok("#<function call>".to_owned())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl sim_kernel::ObjectCompat for CallFunction {
    fn class(&self, cx: &mut Cx) -> Result<ClassRef> {
        function_class(cx)
    }
    fn as_expr(&self, _cx: &mut Cx) -> Result<Expr> {
        Ok(Expr::Symbol(call_symbol()))
    }
    fn as_callable(&self) -> Option<&dyn Callable> {
        Some(self)
    }
}

impl Callable for CallFunction {
    fn call(&self, cx: &mut Cx, args: Args) -> Result<Value> {
        let mut values = args.into_vec();
        if values.is_empty() {
            return Err(Error::Eval(
                "call expects a callable value and at least zero arguments".to_owned(),
            ));
        }
        let callable = values.remove(0);
        cx.call_value(callable, Args::new(values))
    }
}

#[derive(Clone)]
pub struct GradFunction;

impl Object for GradFunction {
    fn display(&self, _cx: &mut Cx) -> Result<String> {
        Ok("#<function grad>".to_owned())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl sim_kernel::ObjectCompat for GradFunction {
    fn class(&self, cx: &mut Cx) -> Result<ClassRef> {
        function_class(cx)
    }
    fn as_expr(&self, _cx: &mut Cx) -> Result<Expr> {
        Ok(Expr::Symbol(grad_symbol()))
    }
    fn as_callable(&self) -> Option<&dyn Callable> {
        Some(self)
    }
}

impl Callable for GradFunction {
    fn call(&self, cx: &mut Cx, args: Args) -> Result<Value> {
        let values = args.into_vec();
        let [value] = values.as_slice() else {
            return Err(Error::Eval(
                "grad expects exactly one function value".to_owned(),
            ));
        };
        let func = expect_func(value)?;
        let mut grads = Vec::with_capacity(func.vars.len());
        for var in &func.vars {
            let var_value = cx.factory().symbol(var.clone())?;
            grads.push(cx.call_function(
                &Symbol::new("diff"),
                Args::new(vec![value.clone(), var_value]),
            )?);
        }
        cx.factory().list(grads)
    }
}

pub(crate) struct FuncValueClass {
    id: std::sync::atomic::AtomicU32,
}

pub(crate) fn build_func_class() -> Arc<FuncValueClass> {
    Arc::new(FuncValueClass {
        id: std::sync::atomic::AtomicU32::new(0),
    })
}

impl FuncValueClass {
    pub(crate) fn set_id(&self, id: ClassId) {
        self.id.store(id.0, std::sync::atomic::Ordering::Relaxed);
    }
}

impl Object for FuncValueClass {
    fn display(&self, _cx: &mut Cx) -> Result<String> {
        Ok(format!("#<class {}>", func_class_symbol()))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl sim_kernel::ObjectCompat for FuncValueClass {
    fn class(&self, cx: &mut Cx) -> Result<ClassRef> {
        if let Some(value) = cx
            .registry()
            .class_by_symbol(&Symbol::qualified("core", "Class"))
        {
            return Ok(value.clone());
        }
        DefaultFactory.class_stub(
            sim_kernel::CORE_CLASS_CLASS_ID,
            Symbol::qualified("core", "Class"),
        )
    }
    fn as_expr(&self, _cx: &mut Cx) -> Result<Expr> {
        Ok(Expr::Symbol(func_class_symbol()))
    }
    fn as_callable(&self) -> Option<&dyn Callable> {
        Some(self)
    }
    fn as_class(&self) -> Option<&dyn Class> {
        Some(self)
    }
    fn as_read_constructor(&self) -> Option<&dyn ReadConstructor> {
        Some(self)
    }
}

impl Callable for FuncValueClass {
    fn call(&self, cx: &mut Cx, args: Args) -> Result<Value> {
        let values = args.into_vec();
        let [vars_value, body_value] = values.as_slice() else {
            return Err(Error::Eval(format!(
                "class {} expects exactly two arguments",
                func_class_symbol()
            )));
        };
        let vars = parse_vars_value(cx, vars_value)?;
        let body_expr = body_value.object().as_expr(cx)?;
        let body_cas = expr_to_cas_expr(cx, &body_expr)?
            .ok_or_else(|| Error::Eval("numbers/Func body must be CAS-compatible".to_owned()))?;
        build_func_value(
            cx,
            Func::new(vars, Some(body_cas), None, FuncMetadata::default()),
        )
    }
}

impl Class for FuncValueClass {
    fn id(&self) -> ClassId {
        ClassId(self.id.load(std::sync::atomic::Ordering::Relaxed))
    }

    fn symbol(&self) -> Symbol {
        func_class_symbol()
    }

    fn constructor_shape(&self, cx: &mut Cx) -> Result<ShapeRef> {
        cx.factory().nil()
    }

    fn instance_shape(&self, cx: &mut Cx) -> Result<ShapeRef> {
        Ok(cx
            .registry()
            .shape_by_symbol(&value_shape_symbol())
            .cloned()
            .unwrap_or(cx.factory().symbol(value_shape_symbol())?))
    }

    fn read_constructor(&self, cx: &mut Cx) -> Result<Option<ReadConstructorRef>> {
        Ok(cx.registry().class_by_symbol(&func_class_symbol()).cloned())
    }

    fn members(&self, cx: &mut Cx) -> Result<TableRef> {
        cx.factory().table(Vec::new())
    }
}

impl ReadConstructor for FuncValueClass {
    fn symbol(&self) -> Symbol {
        func_class_symbol()
    }

    fn args_shape(&self, cx: &mut Cx) -> Result<ShapeRef> {
        cx.factory().nil()
    }

    fn construct_read(&self, cx: &mut Cx, args: Vec<Value>) -> Result<Value> {
        self.call(cx, Args::new(args))
    }
}

impl ObjectEncode for Func {
    fn object_encoding(&self, cx: &mut Cx) -> Result<ObjectEncoding> {
        let Some(body_cas) = &self.body_cas else {
            return Err(Error::Eval(
                "native-only functions do not have a read-construct encoding".to_owned(),
            ));
        };
        Ok(ObjectEncoding::Constructor {
            class: func_class_symbol(),
            args: vec![
                vars_expr(&self.vars),
                cas_expr_to_surface_expr(cx, body_cas)?,
            ],
        })
    }
}

pub(crate) fn parse_vars_expr(expr: &Expr) -> Result<Vec<Symbol>> {
    let Expr::List(items) = expr else {
        return Err(Error::Eval(
            "function parameter list must be a list of symbols".to_owned(),
        ));
    };
    items
        .iter()
        .map(|item| match item {
            Expr::Symbol(symbol) => Ok(symbol.clone()),
            _ => Err(Error::Eval(
                "function parameter list must contain only symbols".to_owned(),
            )),
        })
        .collect()
}

fn parse_vars_value(cx: &mut Cx, value: &Value) -> Result<Vec<Symbol>> {
    parse_vars_expr(&value.object().as_expr(cx)?)
}

pub(crate) fn vars_expr(vars: &[Symbol]) -> Expr {
    Expr::List(vars.iter().cloned().map(Expr::Symbol).collect())
}

pub(crate) fn function_class(cx: &mut Cx) -> Result<ClassRef> {
    if let Some(value) = cx
        .registry()
        .class_by_symbol(&Symbol::qualified("core", "Function"))
    {
        return Ok(value.clone());
    }
    cx.factory().class_stub(
        sim_kernel::CORE_FUNCTION_CLASS_ID,
        Symbol::qualified("core", "Function"),
    )
}

pub(crate) fn expect_func(value: &Value) -> Result<&Func> {
    value
        .object()
        .downcast_ref::<Func>()
        .ok_or_else(|| Error::Eval("expected a numbers/func value".to_owned()))
}

pub(crate) fn child_env_with_args(parent: &Env, vars: &[Symbol], args: &[Value]) -> Result<Env> {
    if vars.len() != args.len() {
        return Err(Error::Eval(format!(
            "function expected {} arguments but received {}",
            vars.len(),
            args.len()
        )));
    }
    let mut env = Env::child(Arc::new(parent.clone()));
    for (var, value) in vars.iter().cloned().zip(args.iter().cloned()) {
        env.define(var, value);
    }
    Ok(env)
}
