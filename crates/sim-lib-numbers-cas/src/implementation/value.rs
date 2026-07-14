//! The symbolic expression tree `CasExpr` and its `CasValue` object, plus the
//! conversions from a `CasExpr` back into a runtime value or surface `Expr`.

use std::{any::Any, cmp::Ordering, sync::Arc};

use sim_kernel::{
    ClassRef, Cx, DefaultFactory, Error, Expr, Factory, LengthResult, ListValue, NumberLiteral,
    NumberValue, Object, ObjectCompat, ObjectEncode, ObjectEncoding, Result, Symbol, Value,
};

use super::{citizen::cas_value_class_symbol, domain::cas_domain_symbol};

/// A node in the symbolic expression tree.
///
/// A `CasExpr` is either a concrete number [`Value`], a free variable, or an
/// operator applied to argument subtrees. This is the internal algebraic form;
/// surface `Expr`/`Value` conversions go through [`expr_to_cas_expr`],
/// [`value_to_cas_expr`], [`cas_expr_to_value`], and [`cas_expr_to_surface_expr`].
///
/// [`expr_to_cas_expr`]: crate::expr_to_cas_expr
/// [`value_to_cas_expr`]: crate::value_to_cas_expr
#[derive(Clone, Debug)]
pub enum CasExpr {
    /// A concrete number value carried unchanged through the CAS.
    Num(Value),
    /// A free symbolic variable, e.g. `x`.
    Var(Symbol),
    /// An operator (such as `math/add`) applied to argument subtrees.
    Op(Symbol, Vec<CasExpr>),
}

impl CasExpr {
    /// Builds a concrete number leaf.
    ///
    /// Returns an error when `value` does not implement the number-value
    /// protocol. `CasExpr::Num` is a number leaf, not a generic value escape
    /// hatch.
    pub fn num(cx: &mut Cx, value: Value) -> Result<Self> {
        if cx.number_value_ref(value.clone())?.is_some() {
            Ok(Self::Num(value))
        } else {
            Err(Error::Eval(format!(
                "CasExpr::Num expected a number value, found {}",
                value.object().display(cx)?
            )))
        }
    }
}

/// Returns free variables in deterministic first-seen order.
pub fn free_vars(expr: &CasExpr) -> Vec<Symbol> {
    fn walk(expr: &CasExpr, out: &mut Vec<Symbol>) {
        match expr {
            CasExpr::Num(_) => {}
            CasExpr::Var(symbol) => {
                if !out.contains(symbol) {
                    out.push(symbol.clone());
                }
            }
            CasExpr::Op(_, args) => {
                for arg in args {
                    walk(arg, out);
                }
            }
        }
    }

    let mut out = Vec::new();
    walk(expr, &mut out);
    out
}

/// Compares two CAS trees by their canonical surface expression.
///
/// This compares the algebraic surface form instead of relying on runtime
/// `Value` identity inside number leaves.
pub fn canonical_eq(cx: &mut Cx, left: &CasExpr, right: &CasExpr) -> Result<bool> {
    let left = cas_expr_to_surface_expr(cx, left)?;
    let right = cas_expr_to_surface_expr(cx, right)?;
    Ok(left.canonical_eq(&right))
}

#[derive(Clone, Debug)]
pub struct CasValue {
    pub(crate) expr: CasExpr,
}

impl CasValue {
    pub(crate) fn new(expr: CasExpr) -> Self {
        Self { expr }
    }

    pub(crate) fn expr(&self) -> &CasExpr {
        &self.expr
    }
}

impl Object for CasValue {
    fn display(&self, cx: &mut Cx) -> Result<String> {
        Ok(format!("#<cas {:?}>", self.as_expr(cx)?))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl sim_kernel::ObjectCompat for CasValue {
    fn class(&self, cx: &mut Cx) -> Result<ClassRef> {
        if let Some(value) = cx.registry().class_by_symbol(&cas_value_class_symbol()) {
            return Ok(value.clone());
        }
        DefaultFactory.class_stub(
            sim_kernel::CORE_NUMBER_CLASS_ID,
            Symbol::qualified("core", "Number"),
        )
    }
    fn as_expr(&self, cx: &mut Cx) -> Result<Expr> {
        cas_expr_to_surface_expr(cx, &self.expr)
    }
    fn as_table(&self, cx: &mut Cx) -> Result<Value> {
        let expr = self.as_expr(cx)?;
        let expr_value = cx.factory().expr(expr)?;
        cx.factory().table(vec![
            (Symbol::new("kind"), cx.factory().string("cas".to_owned())?),
            (
                Symbol::new("domain"),
                cx.factory().symbol(cas_domain_symbol())?,
            ),
            (Symbol::new("expr"), expr_value),
        ])
    }
    fn as_number_value(&self) -> Option<&dyn NumberValue> {
        Some(self)
    }
    fn as_list(&self) -> Option<&dyn ListValue> {
        matches!(self.expr, CasExpr::Op(_, _)).then_some(self)
    }

    fn as_object_encoder(&self) -> Option<&dyn ObjectEncode> {
        Some(self)
    }
}

impl ObjectEncode for CasValue {
    fn object_encoding(&self, cx: &mut Cx) -> Result<ObjectEncoding> {
        Ok(ObjectEncoding::Constructor {
            class: cas_value_class_symbol(),
            args: vec![
                Expr::Symbol(Symbol::new("v1")),
                cas_expr_to_surface_expr(cx, &self.expr)?,
            ],
        })
    }
}

impl sim_citizen::Citizen for CasValue {
    fn citizen_symbol() -> Symbol {
        cas_value_class_symbol()
    }

    fn citizen_version() -> u32 {
        1
    }

    fn citizen_arity() -> usize {
        1
    }

    fn citizen_fields() -> &'static [&'static str] {
        &["expr"]
    }
}

impl NumberValue for CasValue {
    fn number_domain(&self, _cx: &mut Cx) -> Result<Symbol> {
        Ok(cas_domain_symbol())
    }

    fn number_literal(&self, _cx: &mut Cx) -> Result<Option<NumberLiteral>> {
        Ok(None)
    }
}

impl ListValue for CasValue {
    fn is_empty(&self, _cx: &mut Cx) -> Result<bool> {
        Ok(false)
    }

    fn car(&self, cx: &mut Cx) -> Result<Option<Value>> {
        Ok(self.list_values(cx)?.first().cloned())
    }

    fn cdr(&self, cx: &mut Cx) -> Result<Option<Value>> {
        let tail = self
            .list_values(cx)
            .map(|values| values.into_iter().skip(1).collect())?;
        Ok(Some(cx.factory().list(tail)?))
    }

    fn len(&self, _cx: &mut Cx) -> Result<LengthResult> {
        Ok(LengthResult::Known(self.list_len()))
    }

    fn len_cmp(&self, _cx: &mut Cx, n: usize) -> Result<Ordering> {
        Ok(self.list_len().cmp(&n))
    }

    fn get(&self, cx: &mut Cx, index: usize) -> Result<Option<Value>> {
        Ok(self.list_values(cx)?.get(index).cloned())
    }
}

impl CasValue {
    fn list_len(&self) -> usize {
        match &self.expr {
            CasExpr::Num(_) | CasExpr::Var(_) => 1,
            CasExpr::Op(_, args) => args.len() + 1,
        }
    }

    fn list_values(&self, cx: &mut Cx) -> Result<Vec<Value>> {
        match &self.expr {
            CasExpr::Num(value) => Ok(vec![value.clone()]),
            CasExpr::Var(symbol) => Ok(vec![cx.factory().symbol(symbol.clone())?]),
            CasExpr::Op(operator, args) => {
                let mut values = Vec::with_capacity(args.len() + 1);
                values.push(cx.factory().symbol(display_operator(operator))?);
                for arg in args {
                    values.push(cas_expr_to_value(cx, arg.clone())?);
                }
                Ok(values)
            }
        }
    }
}

/// Materialize a [`CasExpr`] as a runtime [`Value`].
///
/// A bare [`CasExpr::Num`] unwraps to its underlying number value; every other
/// node is boxed as an opaque CAS value object.
pub fn cas_expr_to_value(cx: &mut Cx, expr: CasExpr) -> Result<Value> {
    match expr {
        CasExpr::Num(value) => Ok(value),
        other => cx.factory().opaque(Arc::new(CasValue::new(other))),
    }
}

/// Lower a [`CasExpr`] into a surface [`Expr`] for display and encoding.
///
/// Operator symbols are rendered in their infix surface spelling (`math/add`
/// becomes `+`, and so on); variables and numbers map to their direct surface
/// forms.
pub fn cas_expr_to_surface_expr(cx: &mut Cx, expr: &CasExpr) -> Result<Expr> {
    match expr {
        CasExpr::Num(value) => value.object().as_expr(cx),
        CasExpr::Var(symbol) => Ok(Expr::Symbol(symbol.clone())),
        CasExpr::Op(operator, args) => Ok(Expr::List(
            std::iter::once(Expr::Symbol(display_operator(operator)))
                .chain(
                    args.iter()
                        .map(|arg| cas_expr_to_surface_expr(cx, arg))
                        .collect::<Result<Vec<_>>>()?,
                )
                .collect(),
        )),
    }
}

fn display_operator(operator: &Symbol) -> Symbol {
    match (operator.namespace.as_deref(), operator.name.as_ref()) {
        (Some("math"), "add") => Symbol::new("+"),
        (Some("math"), "sub") => Symbol::new("-"),
        (Some("math"), "mul") => Symbol::new("*"),
        (Some("math"), "div") => Symbol::new("/"),
        (Some("math"), "rem") => Symbol::new("%"),
        (Some("math"), "neg") => Symbol::new("-"),
        (Some("math"), "pow") => Symbol::new("^"),
        _ => operator.clone(),
    }
}
