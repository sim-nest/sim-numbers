//! The `Func` function value: variables plus a labelled CAS/native body, with
//! its metadata and arithmetic over function values.

use std::{any::Any, sync::Arc};

use sim_kernel::{
    Args, Callable, ClassRef, Cx, DefaultFactory, Error, Expr, Factory, NumberValue, Object,
    ObjectEncode, Result, ShapeRef, Symbol, Value,
};
use sim_lib_numbers_cas::{CasExpr, cas_expr_to_surface_expr, free_vars, value_to_cas_expr};
use sim_lib_numbers_cas_eval::eval_cas;
use sim_shape::{AnyShape, ListShape, Shape, shape_value};

use super::domain::{func_class_symbol, func_domain_symbol};
use super::function::{child_env_with_args, vars_expr};

mod ops;

pub(crate) use ops::register_value_ops;

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

#[derive(Clone)]
enum FuncBody {
    Symbolic(CasExpr),
    Native {
        native: NativeFn,
        symbolic_status: SymbolicStatus,
    },
    Dual {
        cas: CasExpr,
        native: NativeFn,
    },
}

/// Whether a [`Func`] exposes an exact symbolic body, is differentiable through
/// a named hint, or has lost its symbolic body for a named reason.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SymbolicStatus {
    /// A CAS body is available and can be inspected by symbolic tools.
    Available,
    /// The function is native-only, but metadata names a differentiator that
    /// can handle it numerically or exactly outside the CAS path.
    ProvidedByHint,
    /// The function is native-only because symbolic information was lost.
    Lost {
        /// The machine-readable reason for the loss.
        reason: Symbol,
    },
}

impl SymbolicStatus {
    /// Reason used for ordinary native functions with no symbolic body.
    pub fn native_only() -> Self {
        Self::Lost {
            reason: Symbol::qualified("numbers/func", "native-only"),
        }
    }

    /// Reason used when arithmetic combines a symbolic body with a native-only
    /// body and must keep only the executable native result.
    pub fn mixed_native() -> Self {
        Self::Lost {
            reason: Symbol::qualified("numbers/func", "mixed-native"),
        }
    }

    fn status_symbol(&self) -> Symbol {
        match self {
            Self::Available => Symbol::qualified("numbers/func", "available"),
            Self::ProvidedByHint => Symbol::qualified("numbers/func", "provided-by-hint"),
            Self::Lost { .. } => Symbol::qualified("numbers/func", "lost"),
        }
    }

    fn reason(&self) -> Option<&Symbol> {
        match self {
            Self::Lost { reason } => Some(reason),
            _ => None,
        }
    }
}

/// A callable function value in the `Func` number domain: its bound variables
/// plus one labelled symbolic, native, or internal dual body.
#[derive(Clone)]
pub struct Func {
    /// The ordered parameter symbols bound when the function is invoked.
    pub vars: Vec<Symbol>,
    body: FuncBody,
    /// Out-of-band metadata describing the function.
    pub metadata: FuncMetadata,
}

impl Func {
    fn new(vars: Vec<Symbol>, body: FuncBody, metadata: FuncMetadata) -> Self {
        Self {
            vars,
            body,
            metadata,
        }
    }

    /// Builds a function with a symbolic (CAS) body and default metadata.
    pub fn symbolic(vars: Vec<Symbol>, body_cas: CasExpr) -> Self {
        Self::symbolic_with(vars, body_cas, FuncMetadata::default())
    }

    /// Builds a function with a symbolic (CAS) body and caller-supplied metadata.
    pub fn symbolic_with(vars: Vec<Symbol>, body_cas: CasExpr, metadata: FuncMetadata) -> Self {
        Self::new(vars, FuncBody::Symbolic(body_cas), metadata)
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
    /// assert!(func.body_cas().is_none());
    /// assert!(func.is_native());
    /// ```
    pub fn native(vars: Vec<Symbol>, body_native: NativeFn) -> Self {
        Self::native_with(vars, body_native, FuncMetadata::default())
    }

    /// Builds a function with a native body and caller-supplied metadata.
    pub fn native_with(vars: Vec<Symbol>, body_native: NativeFn, metadata: FuncMetadata) -> Self {
        let status = if metadata.differentiator_hint.is_some() {
            SymbolicStatus::ProvidedByHint
        } else {
            SymbolicStatus::native_only()
        };
        Self::native_with_status(vars, body_native, metadata, status)
    }

    fn native_with_status(
        vars: Vec<Symbol>,
        body_native: NativeFn,
        metadata: FuncMetadata,
        symbolic_status: SymbolicStatus,
    ) -> Self {
        Self::new(
            vars,
            FuncBody::Native {
                native: body_native,
                symbolic_status,
            },
            metadata,
        )
    }

    /// Builds an internal dual-body function whose native body is derived from
    /// the same operation as the symbolic body.
    pub(crate) fn dual_with(
        vars: Vec<Symbol>,
        body_cas: CasExpr,
        body_native: NativeFn,
        metadata: FuncMetadata,
    ) -> Self {
        Self::new(
            vars,
            FuncBody::Dual {
                cas: body_cas,
                native: body_native,
            },
            metadata,
        )
    }

    /// Returns the symbolic body advertised by this function, when available.
    pub fn body_cas(&self) -> Option<&CasExpr> {
        match &self.body {
            FuncBody::Symbolic(body) | FuncBody::Dual { cas: body, .. } => Some(body),
            FuncBody::Native { .. } => None,
        }
    }

    fn body_native(&self) -> Option<&NativeFn> {
        match &self.body {
            FuncBody::Native { native: body, .. } | FuncBody::Dual { native: body, .. } => {
                Some(body)
            }
            FuncBody::Symbolic(_) => None,
        }
    }

    /// Returns whether this function carries a native body.
    pub fn is_native(&self) -> bool {
        self.body_native().is_some()
    }

    /// Returns the symbolic-body status for this function.
    pub fn symbolic_status(&self) -> SymbolicStatus {
        match &self.body {
            FuncBody::Symbolic(_) | FuncBody::Dual { .. } => SymbolicStatus::Available,
            FuncBody::Native {
                symbolic_status, ..
            } => symbolic_status.clone(),
        }
    }

    fn native_extension_payload(&self) -> Expr {
        let status = self.symbolic_status();
        let mut fields = vec![(
            Expr::Symbol(Symbol::new("symbolic-status")),
            Expr::Symbol(status.status_symbol()),
        )];
        if let Some(reason) = status.reason() {
            fields.push((
                Expr::Symbol(Symbol::new("symbolic-loss-reason")),
                Expr::Symbol(reason.clone()),
            ));
        }
        if let Some(hint) = &self.metadata.differentiator_hint {
            fields.push((
                Expr::Symbol(Symbol::new("differentiator-hint")),
                Expr::Symbol(hint.clone()),
            ));
        }
        Expr::Map(fields)
    }

    fn invoke(&self, cx: &mut Cx, args: &[Value]) -> Result<Value> {
        if args.len() != self.vars.len() {
            return Err(Error::Eval(format!(
                "function expected {} arguments but received {}",
                self.vars.len(),
                args.len()
            )));
        }
        if let Some(body_native) = self.body_native() {
            return body_native(cx, args);
        }
        let body_cas = self
            .body_cas()
            .expect("FuncBody always contains a symbolic or native body");
        let env = child_env_with_args(cx.env(), &self.vars, args)?;
        cx.with_env(env.clone(), |cx| eval_cas(cx, body_cas, &env))
    }
}

impl Object for Func {
    fn display(&self, cx: &mut Cx) -> Result<String> {
        if let Some(body_cas) = self.body_cas() {
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
        let Some(body_cas) = self.body_cas() else {
            return Ok(Expr::Extension {
                tag: func_class_symbol(),
                payload: Box::new(self.native_extension_payload()),
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
            .body_cas()
            .map(|body| cas_expr_to_surface_expr(cx, body))
            .transpose()?;
        let body = match self.body_cas() {
            Some(_) => cx
                .factory()
                .expr(body_expr.expect("body expr should exist when body_cas is present"))?,
            None => cx.factory().nil()?,
        };
        let native = cx.factory().bool(self.is_native())?;
        let symbolic_status = self.symbolic_status();
        let mut fields = vec![
            (Symbol::new("kind"), cx.factory().string("func".to_owned())?),
            (Symbol::new("vars"), vars),
            (Symbol::new("body"), body),
            (Symbol::new("native"), native),
            (
                Symbol::new("symbolic-status"),
                cx.factory().symbol(symbolic_status.status_symbol())?,
            ),
        ];
        if let Some(reason) = symbolic_status.reason() {
            fields.push((
                Symbol::new("symbolic-loss-reason"),
                cx.factory().symbol(reason.clone())?,
            ));
        }
        cx.factory().table(fields)
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
    let body = value_to_cas_expr(cx, value)?;
    let vars = free_vars(&body);
    build_func_value(cx, Func::symbolic(vars, body))
}
