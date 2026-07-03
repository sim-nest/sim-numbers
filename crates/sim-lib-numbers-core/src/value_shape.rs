//! The domain number-value shape and the number-domain browse table.

use sim_kernel::{Cx, Expr, NumberValueRef, Result, Symbol, Value};
use sim_shape::{MatchScore, Shape, ShapeDoc, ShapeMatch};

/// The stable symbol for a domain's number-value shape.
pub fn value_shape_symbol(domain: &Symbol) -> Symbol {
    Symbol::qualified(domain.to_string(), "value-shape")
}

/// Inputs for the shared number-domain browse table.
pub struct NumberDomainTableSpec<'a> {
    domain: Symbol,
    numeric_family: &'a str,
    canonical_form: &'a str,
    parse_priority: i32,
    literal_class: Value,
    instance_shape: Value,
    value_shape: Value,
}

impl<'a> NumberDomainTableSpec<'a> {
    /// Assemble the inputs for one domain's browse table from its identity,
    /// parse priority, and the literal-class/instance-shape/value-shape values.
    pub fn new(
        domain: Symbol,
        numeric_family: &'a str,
        canonical_form: &'a str,
        parse_priority: i32,
        literal_class: Value,
        instance_shape: Value,
        value_shape: Value,
    ) -> Self {
        Self {
            domain,
            numeric_family,
            canonical_form,
            parse_priority,
            literal_class,
            instance_shape,
            value_shape,
        }
    }
}

/// Build the shared number-domain browse table value.
pub fn number_domain_table(cx: &mut Cx, spec: NumberDomainTableSpec<'_>) -> Result<Value> {
    cx.factory().table(vec![
        (Symbol::new("symbol"), cx.factory().symbol(spec.domain)?),
        (
            Symbol::new("kind"),
            cx.factory().string("number-domain".to_owned())?,
        ),
        (
            Symbol::new("numeric-family"),
            cx.factory().string(spec.numeric_family.to_owned())?,
        ),
        (
            Symbol::new("canonical-form"),
            cx.factory().string(spec.canonical_form.to_owned())?,
        ),
        (
            Symbol::new("parse-priority"),
            cx.factory().string(spec.parse_priority.to_string())?,
        ),
        (Symbol::new("literal-class"), spec.literal_class),
        (Symbol::new("instance-shape"), spec.instance_shape),
        (Symbol::new("value-shape"), spec.value_shape),
    ])
}

/// A shape matching number *values* (not just literals) in one domain.
pub struct DomainNumberValueShape {
    domain: Symbol,
    name: &'static str,
    details: Vec<&'static str>,
}

impl DomainNumberValueShape {
    /// Build a value shape for `domain` with a browse `name` and detail lines.
    pub fn new(
        domain: Symbol,
        name: &'static str,
        details: impl IntoIterator<Item = &'static str>,
    ) -> Self {
        Self {
            domain,
            name,
            details: details.into_iter().collect(),
        }
    }

    fn matches_number(&self, number: &NumberValueRef) -> bool {
        number.domain == self.domain
    }
}

impl Shape for DomainNumberValueShape {
    fn check_value(&self, cx: &mut Cx, value: Value) -> Result<ShapeMatch> {
        match cx.number_value_ref(value)? {
            Some(number) if self.matches_number(&number) => {
                Ok(ShapeMatch::accept(MatchScore::exact(25)))
            }
            _ => Ok(ShapeMatch::reject(format!(
                "expected number value in {}",
                self.domain
            ))),
        }
    }

    fn check_expr(&self, _cx: &mut Cx, expr: &Expr) -> Result<ShapeMatch> {
        match expr {
            Expr::Number(number) if number.domain == self.domain => {
                Ok(ShapeMatch::accept(MatchScore::exact(20)))
            }
            _ => Ok(ShapeMatch::reject(format!(
                "expected number value in {}",
                self.domain
            ))),
        }
    }

    fn describe(&self, _cx: &mut Cx) -> Result<ShapeDoc> {
        let mut doc = ShapeDoc::new(self.name);
        for detail in &self.details {
            doc = doc.with_detail(*detail);
        }
        Ok(doc)
    }
}

/// Assert the domain's value-shape symbol matches the shared helper (test aid).
pub fn assert_value_shape_symbol(domain: Symbol, actual: Symbol) {
    assert_eq!(actual, value_shape_symbol(&domain));
}
