//! The CAS runtime functions: the `cas/var` and `cas/simplify` symbols and the
//! callable `CasFunction` object that dispatches them.

use std::any::Any;

use sim_kernel::{
    Args, Callable, ClassRef, Cx, DefaultFactory, Error, Expr, Factory, Object, QuoteMode, Result,
    Symbol, Value,
};

use super::simplify::simplify_value;

/// The `cas/var` symbol: constructs a free symbolic variable.
pub fn cas_var_symbol() -> Symbol {
    Symbol::qualified("cas", "var")
}

/// The `cas/simplify` symbol: simplifies a symbolic CAS value.
pub fn cas_simplify_symbol() -> Symbol {
    Symbol::qualified("cas", "simplify")
}

#[derive(Clone)]
pub(crate) struct CasFunction {
    pub(crate) symbol: Symbol,
}

impl Object for CasFunction {
    fn display(&self, _cx: &mut Cx) -> Result<String> {
        Ok(format!("#<function {}>", self.symbol))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl sim_kernel::ObjectCompat for CasFunction {
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

impl Callable for CasFunction {
    fn call(&self, cx: &mut Cx, args: Args) -> Result<Value> {
        match self.symbol.clone() {
            symbol if symbol == cas_var_symbol() => build_var(cx, args.into_vec()),
            symbol if symbol == cas_simplify_symbol() => simplify(cx, args.into_vec()),
            _ => Err(Error::Eval(format!(
                "unsupported CAS helper function {}",
                self.symbol
            ))),
        }
    }
}

fn build_var(cx: &mut Cx, values: Vec<Value>) -> Result<Value> {
    let [value] = values.as_slice() else {
        return Err(Error::Eval(format!(
            "{} expects exactly one argument",
            cas_var_symbol()
        )));
    };
    let Some(symbol) = extract_symbolish(cx, value)? else {
        return Err(Error::Eval(
            "cas/var expects a quoted symbol or symbol value".to_owned(),
        ));
    };
    simplify_value(cx, cx.factory().symbol(symbol)?)
}

fn simplify(cx: &mut Cx, values: Vec<Value>) -> Result<Value> {
    let [value] = values.as_slice() else {
        return Err(Error::Eval(format!(
            "{} expects exactly one argument",
            cas_simplify_symbol()
        )));
    };
    simplify_value(cx, value.clone())
}

/// Extract a bare or singly-quoted symbol from a CAS value, if it is one.
/// Shared by the differentiation and integration function rules (OVERLAP6.10).
pub fn extract_symbolish(cx: &mut Cx, value: &Value) -> Result<Option<Symbol>> {
    match value.object().as_expr(cx)? {
        Expr::Symbol(symbol) => Ok(Some(symbol)),
        Expr::Quote {
            mode: QuoteMode::Quote,
            expr,
        } => match *expr {
            Expr::Symbol(symbol) => Ok(Some(symbol)),
            _ => Ok(None),
        },
        _ => Ok(None),
    }
}
