//! Inspectable composed numeric pipeline values and their constructor.

use std::{collections::BTreeMap, sync::Arc};

use sim_kernel::{
    Args, ClassRef, Cx, DefaultFactory, Error, Expr, Factory, HandleStore, Object, Ref, Result,
    Symbol, Term, Value,
};
use sim_lib_numbers_func::Func;

use super::{options::parse_symbolish_value, registry::global_numeric_registry};

/// Numeric pipeline domain selected for a composed pipeline.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PipelineKind {
    /// Ordinary differential equation solve.
    OdeSolve,
    /// Quadrature / integration pipeline.
    Quadrature,
}

impl PipelineKind {
    /// Symbol exposed in the pipeline inspection table.
    pub fn symbol(&self) -> Symbol {
        match self {
            Self::OdeSolve => Symbol::new("ode-solve"),
            Self::Quadrature => Symbol::new("quadrature"),
        }
    }
}

/// Runtime state representation selected for a composed pipeline.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StateKind {
    /// Scalar `f64` state.
    F64,
    /// Tensor state promoted through the numeric domain graph.
    Tensor,
}

impl StateKind {
    /// Symbol exposed in the pipeline inspection table.
    pub fn symbol(&self) -> Symbol {
        match self {
            Self::F64 => Symbol::new("f64"),
            Self::Tensor => Symbol::new("tensor"),
        }
    }
}

/// First-class value describing a numeric pipeline composition.
#[derive(Clone, Debug)]
pub struct ComposedPipeline {
    /// Reference to the function value the pipeline runs.
    pub func_ref: Ref,
    /// Pipeline domain.
    pub kind: PipelineKind,
    /// Backend method selected for the pipeline.
    pub method: Symbol,
    /// Runtime state representation.
    pub state: StateKind,
}

impl ComposedPipeline {
    /// Builds a composed pipeline descriptor.
    pub fn new(func_ref: Ref, kind: PipelineKind, method: Symbol, state: StateKind) -> Self {
        Self {
            func_ref,
            kind,
            method,
            state,
        }
    }

    /// Projects this pipeline to an inspectable SIM table value.
    pub fn table_value(&self, factory: &dyn Factory) -> Result<Value> {
        factory.table(vec![
            (
                Symbol::new("kind"),
                factory.string("composed-pipeline".to_owned())?,
            ),
            (Symbol::new("domain"), factory.symbol(self.kind.symbol())?),
            (Symbol::new("method"), factory.symbol(self.method.clone())?),
            (Symbol::new("state"), factory.symbol(self.state.symbol())?),
            (
                Symbol::new("func"),
                factory.expr(Term::Ref(self.func_ref.clone()).into())?,
            ),
        ])
    }
}

impl Object for ComposedPipeline {
    fn display(&self, _cx: &mut Cx) -> Result<String> {
        Ok(format!(
            "#<composed-pipeline {} {} {}>",
            self.kind.symbol(),
            self.method,
            self.state.symbol()
        ))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl sim_kernel::ObjectCompat for ComposedPipeline {
    fn class(&self, cx: &mut Cx) -> Result<ClassRef> {
        if let Some(value) = cx
            .registry()
            .class_by_symbol(&Symbol::qualified("core", "Table"))
        {
            return Ok(value.clone());
        }
        DefaultFactory.class_stub(
            sim_kernel::CORE_TABLE_CLASS_ID,
            Symbol::qualified("core", "Table"),
        )
    }

    fn as_expr(&self, cx: &mut Cx) -> Result<Expr> {
        self.as_table(cx)?.object().as_expr(cx)
    }

    fn as_table(&self, cx: &mut Cx) -> Result<Value> {
        self.table_value(cx.factory())
    }
}

pub fn call_numeric_compose(cx: &mut Cx, args: Args) -> Result<Value> {
    let values = args.into_vec();
    let pipeline = compose_from_values(cx, &values)?;
    pipeline_value(cx, pipeline)
}

pub fn call_numeric_compose_exprs(cx: &mut Cx, args: Vec<Expr>) -> Result<Value> {
    let pipeline = compose_from_exprs(cx, &args)?;
    pipeline_value(cx, pipeline)
}

pub fn call_numeric_run_composed(cx: &mut Cx, args: Args) -> Result<Value> {
    super::pipeline_run::call_numeric_run_composed(cx, args)
}

pub fn call_numeric_run_composed_exprs(cx: &mut Cx, args: Vec<Expr>) -> Result<Value> {
    super::pipeline_run::call_numeric_run_composed_exprs(cx, args)
}

fn compose_from_values(cx: &mut Cx, values: &[Value]) -> Result<ComposedPipeline> {
    match values {
        [func, kind, method, state] if !is_compose_key_value(cx, kind)? => {
            let func_ref = require_func_ref(cx, "numeric/compose", func)?;
            let kind = require_pipeline_kind_value(cx, "numeric/compose", kind)?;
            let method = require_symbol_value(cx, "numeric/compose", method)?;
            let state = require_state_kind_value(cx, "numeric/compose", state)?;
            finish_compose(func_ref, kind, method, state)
        }
        [func, rest @ ..] if rest.len().is_multiple_of(2) => {
            let func_ref = require_func_ref(cx, "numeric/compose", func)?;
            let mut options = BTreeMap::<String, Value>::new();
            for pair in rest.chunks(2) {
                let key = require_compose_key_value(cx, &pair[0])?;
                options.insert(key, pair[1].clone());
            }
            let kind = require_compose_kind_value(cx, &options)?;
            let method = require_compose_symbol_value(cx, &options, "method")?;
            let state = require_compose_state_value(cx, &options)?;
            finish_compose(func_ref, kind, method, state)
        }
        _ => Err(Error::Eval(
            "numeric/compose expects func, kind, method, state or keyword pairs".to_owned(),
        )),
    }
}

fn compose_from_exprs(cx: &mut Cx, args: &[Expr]) -> Result<ComposedPipeline> {
    let Some((func_expr, rest)) = args.split_first() else {
        return Err(Error::Eval(
            "numeric/compose expects func, kind, method, state or keyword pairs".to_owned(),
        ));
    };
    let func = cx.eval_expr(func_expr.clone())?;
    let func_ref = require_func_ref(cx, "numeric/compose", &func)?;
    if let [kind_expr, method_expr, state_expr] = rest
        && !is_compose_key_expr(kind_expr)
    {
        let kind = require_pipeline_kind_expr("numeric/compose", kind_expr)?;
        let method = require_symbol_expr("numeric/compose", method_expr)?;
        let state = require_state_kind_expr("numeric/compose", state_expr)?;
        return finish_compose(func_ref, kind, method, state);
    }
    if !rest.len().is_multiple_of(2) {
        return Err(Error::Eval(
            "numeric/compose keyword arguments must be key/value pairs".to_owned(),
        ));
    }
    let mut options = BTreeMap::<String, Symbol>::new();
    for pair in rest.chunks(2) {
        options.insert(
            require_compose_key_expr(&pair[0])?,
            require_symbol_expr("numeric/compose", &pair[1])?,
        );
    }
    let kind = require_compose_kind_symbol(&options)?;
    let method = require_compose_symbol(&options, "method")?;
    let state = parse_state_kind(&require_compose_symbol(&options, "state")?).ok_or_else(|| {
        Error::Eval("numeric/compose expected state kind f64 or tensor".to_owned())
    })?;
    finish_compose(func_ref, kind, method, state)
}

fn require_func_ref(cx: &mut Cx, name: &str, value: &Value) -> Result<Ref> {
    value.object().downcast_ref::<Func>().ok_or_else(|| {
        Error::Eval(format!(
            "{name} expects its first argument to be a Func value"
        ))
    })?;
    Ok(Ref::Handle(cx.handles_mut().intern(value.clone())))
}

fn pipeline_value(cx: &mut Cx, pipeline: ComposedPipeline) -> Result<Value> {
    cx.factory().opaque(Arc::new(pipeline))
}

fn require_pipeline_kind_value(cx: &mut Cx, name: &str, value: &Value) -> Result<PipelineKind> {
    let symbol = require_symbol_value(cx, name, value)?;
    parse_pipeline_kind(&symbol).ok_or_else(|| {
        Error::Eval(format!(
            "{name} expected pipeline kind ode-solve or quadrature"
        ))
    })
}

fn require_state_kind_value(cx: &mut Cx, name: &str, value: &Value) -> Result<StateKind> {
    let symbol = require_symbol_value(cx, name, value)?;
    parse_state_kind(&symbol)
        .ok_or_else(|| Error::Eval(format!("{name} expected state kind f64 or tensor")))
}

fn require_symbol_value(cx: &mut Cx, name: &str, value: &Value) -> Result<Symbol> {
    parse_symbolish_value(cx, value)?
        .ok_or_else(|| Error::Eval(format!("{name} expected a symbol argument")))
}

fn finish_compose(
    func_ref: Ref,
    kind: PipelineKind,
    method: Symbol,
    state: StateKind,
) -> Result<ComposedPipeline> {
    if kind == PipelineKind::Quadrature {
        validate_quadrature_method(&method)?;
    }
    Ok(ComposedPipeline::new(func_ref, kind, method, state))
}

fn require_compose_kind_value(
    cx: &mut Cx,
    options: &BTreeMap<String, Value>,
) -> Result<PipelineKind> {
    let symbol = options
        .get("domain")
        .or_else(|| options.get("kind"))
        .ok_or_else(|| Error::Eval("numeric/compose missing :domain".to_owned()))
        .and_then(|value| require_symbol_value(cx, "numeric/compose", value))?;
    parse_pipeline_kind(&symbol).ok_or_else(|| {
        Error::Eval("numeric/compose expected domain ode-solve or quadrature".to_owned())
    })
}

fn require_compose_kind_symbol(options: &BTreeMap<String, Symbol>) -> Result<PipelineKind> {
    let symbol = options
        .get("domain")
        .or_else(|| options.get("kind"))
        .ok_or_else(|| Error::Eval("numeric/compose missing :domain".to_owned()))?;
    parse_pipeline_kind(symbol).ok_or_else(|| {
        Error::Eval("numeric/compose expected domain ode-solve or quadrature".to_owned())
    })
}

fn require_compose_state_value(
    cx: &mut Cx,
    options: &BTreeMap<String, Value>,
) -> Result<StateKind> {
    let symbol = require_compose_symbol_value(cx, options, "state")?;
    parse_state_kind(&symbol)
        .ok_or_else(|| Error::Eval("numeric/compose expected state kind f64 or tensor".to_owned()))
}

fn require_compose_symbol_value(
    cx: &mut Cx,
    options: &BTreeMap<String, Value>,
    key: &str,
) -> Result<Symbol> {
    let value = options
        .get(key)
        .ok_or_else(|| Error::Eval(format!("numeric/compose missing :{key}")))?;
    require_symbol_value(cx, "numeric/compose", value)
}

fn require_compose_symbol(options: &BTreeMap<String, Symbol>, key: &str) -> Result<Symbol> {
    options
        .get(key)
        .cloned()
        .ok_or_else(|| Error::Eval(format!("numeric/compose missing :{key}")))
}

fn require_pipeline_kind_expr(name: &str, expr: &Expr) -> Result<PipelineKind> {
    let symbol = require_symbol_expr(name, expr)?;
    parse_pipeline_kind(&symbol).ok_or_else(|| {
        Error::Eval(format!(
            "{name} expected pipeline kind ode-solve or quadrature"
        ))
    })
}

fn require_state_kind_expr(name: &str, expr: &Expr) -> Result<StateKind> {
    let symbol = require_symbol_expr(name, expr)?;
    parse_state_kind(&symbol)
        .ok_or_else(|| Error::Eval(format!("{name} expected state kind f64 or tensor")))
}

fn require_symbol_expr(name: &str, expr: &Expr) -> Result<Symbol> {
    match expr {
        Expr::Symbol(symbol) => Ok(symbol.clone()),
        Expr::Quote { expr, .. } => match expr.as_ref() {
            Expr::Symbol(symbol) => Ok(symbol.clone()),
            _ => Err(Error::Eval(format!("{name} expected a symbol argument"))),
        },
        _ => Err(Error::Eval(format!("{name} expected a symbol argument"))),
    }
}

fn parse_pipeline_kind(symbol: &Symbol) -> Option<PipelineKind> {
    match keyword_name(symbol).as_str() {
        "ode-solve" => Some(PipelineKind::OdeSolve),
        "quadrature" => Some(PipelineKind::Quadrature),
        _ => None,
    }
}

fn parse_state_kind(symbol: &Symbol) -> Option<StateKind> {
    match keyword_name(symbol).as_str() {
        "f64" => Some(StateKind::F64),
        "tensor" => Some(StateKind::Tensor),
        _ => None,
    }
}

fn keyword_name(symbol: &Symbol) -> String {
    symbol
        .name
        .strip_prefix(':')
        .unwrap_or(&symbol.name)
        .to_owned()
}

fn is_compose_key_value(cx: &mut Cx, value: &Value) -> Result<bool> {
    Ok(parse_symbolish_value(cx, value)?
        .as_ref()
        .is_some_and(|symbol| is_compose_key_name(&keyword_name(symbol))))
}

fn require_compose_key_value(cx: &mut Cx, value: &Value) -> Result<String> {
    parse_symbolish_value(cx, value)?
        .map(|symbol| keyword_name(&symbol))
        .filter(|key| is_compose_key_name(key))
        .ok_or_else(|| Error::Eval("numeric/compose expected keyword argument".to_owned()))
}

fn is_compose_key_expr(expr: &Expr) -> bool {
    let Expr::Symbol(symbol) = expr else {
        return false;
    };
    is_compose_key_name(&keyword_name(symbol))
}

fn require_compose_key_expr(expr: &Expr) -> Result<String> {
    let Expr::Symbol(symbol) = expr else {
        return Err(Error::Eval(
            "numeric/compose expected keyword argument".to_owned(),
        ));
    };
    let key = keyword_name(symbol);
    if is_compose_key_name(&key) {
        Ok(key)
    } else {
        Err(Error::Eval(format!(
            "numeric/compose: unknown option :{key}"
        )))
    }
}

fn is_compose_key_name(key: &str) -> bool {
    matches!(key, "domain" | "kind" | "method" | "state")
}

fn validate_quadrature_method(method: &Symbol) -> Result<()> {
    let method = resolve_quad_method(method);
    let registry = global_numeric_registry()
        .read()
        .map_err(|_| Error::PoisonedLock("numeric registry"))?;
    if registry.quadrature_fixed(&method).is_some()
        || registry.quadrature_adaptive(&method).is_some()
    {
        Ok(())
    } else {
        Err(unknown_numeric_method("quadrature", &method))
    }
}

fn resolve_quad_method(method: &Symbol) -> Symbol {
    if *method != Symbol::new("auto") {
        return method.clone();
    }
    Symbol::new("simpson")
}

fn unknown_numeric_method(kind: &str, method: &Symbol) -> Error {
    Error::Eval(format!("UnknownNumericMethod: {kind} method {method}"))
}
