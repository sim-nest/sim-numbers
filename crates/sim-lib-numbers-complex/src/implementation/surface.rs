//! The complex number-value shape: a `Shape` matching opaque complex values by
//! domain for dispatch and browsing.

use sim_kernel::{Cx, Expr, NumberValueRef, Result, Symbol, Value};
use sim_shape::{MatchScore, Shape, ShapeDoc, ShapeMatch};

pub(super) struct NumberValueShape {
    domain: Symbol,
    name: &'static str,
    details: Vec<&'static str>,
}

impl NumberValueShape {
    pub(super) fn new(
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

impl Shape for NumberValueShape {
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
