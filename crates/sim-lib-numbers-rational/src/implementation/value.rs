//! The `Rational` value (a numerator/denominator pair of integer values) and
//! its value class: reduction, citizen encoding, and the read-constructor that
//! rebuilds rationals.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use num_bigint::{BigInt, Sign};
use sim_kernel::{
    Args, Callable, Class, ClassId, ClassRef, Cx, DefaultFactory, Error, Expr, Factory,
    NumberLiteral, NumberValue, Object, ObjectEncode, ObjectEncoding, ReadConstructor,
    ReadConstructorRef, Result, ShapeRef, Symbol, TableRef, Value,
};

use crate::implementation::number_domain;
use sim_lib_numbers_core::domains;

use super::domain::{rational_value_class_symbol, value_shape_symbol};
use super::integer::{
    compact_canonical, is_integer_domain, parse_integer_literal, parse_integer_value,
};

/// An exact rational value: a numerator/denominator pair of integer values over
/// bigint, displayed in compact `num/den` form when both parts are compact.
#[derive(Clone)]
pub struct Rational {
    /// The numerator, an integer-domain value carrying the rational's sign.
    pub num: Value,
    /// The denominator, a positive integer-domain value.
    pub den: Value,
}

impl Rational {
    pub(crate) fn new(num: Value, den: Value) -> Self {
        Self { num, den }
    }
}

impl Object for Rational {
    fn display(&self, cx: &mut Cx) -> Result<String> {
        if let Some(canonical) = compact_canonical(cx, &self.num, &self.den)? {
            return Ok(canonical);
        }
        Ok(format!(
            "#<rational {}/{}>",
            self.num.object().display(cx)?,
            self.den.object().display(cx)?
        ))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl sim_kernel::ObjectCompat for Rational {
    fn class(&self, cx: &mut Cx) -> Result<ClassRef> {
        if let Some(value) = cx
            .registry()
            .class_by_symbol(&rational_value_class_symbol())
        {
            return Ok(value.clone());
        }
        DefaultFactory.class_stub(
            sim_kernel::CORE_NUMBER_CLASS_ID,
            Symbol::qualified("core", "Number"),
        )
    }
    fn as_expr(&self, cx: &mut Cx) -> Result<Expr> {
        if let Some(canonical) = compact_canonical(cx, &self.num, &self.den)? {
            return Ok(Expr::Number(NumberLiteral {
                domain: number_domain(),
                canonical,
            }));
        }
        Ok(Expr::Extension {
            tag: rational_value_class_symbol(),
            payload: Box::new(Expr::Vector(vec![
                self.num.object().as_expr(cx)?,
                self.den.object().as_expr(cx)?,
            ])),
        })
    }
    fn as_table(&self, cx: &mut Cx) -> Result<Value> {
        cx.factory().table(vec![
            (
                Symbol::new("kind"),
                cx.factory().string("rational".to_owned())?,
            ),
            (Symbol::new("num"), self.num.clone()),
            (Symbol::new("den"), self.den.clone()),
        ])
    }
    fn as_number_value(&self) -> Option<&dyn NumberValue> {
        Some(self)
    }
    fn as_object_encoder(&self) -> Option<&dyn ObjectEncode> {
        Some(self)
    }
}

impl NumberValue for Rational {
    fn number_domain(&self, _cx: &mut Cx) -> Result<Symbol> {
        Ok(number_domain())
    }

    fn number_literal(&self, cx: &mut Cx) -> Result<Option<NumberLiteral>> {
        Ok(
            compact_canonical(cx, &self.num, &self.den)?.map(|canonical| NumberLiteral {
                domain: number_domain(),
                canonical,
            }),
        )
    }
}

impl ObjectEncode for Rational {
    fn object_encoding(&self, cx: &mut Cx) -> Result<ObjectEncoding> {
        Ok(ObjectEncoding::Constructor {
            class: rational_value_class_symbol(),
            args: vec![
                self.num.object().as_expr(cx)?,
                self.den.object().as_expr(cx)?,
            ],
        })
    }
}

impl sim_citizen::Citizen for Rational {
    fn citizen_symbol() -> Symbol {
        rational_value_class_symbol()
    }

    fn citizen_version() -> u32 {
        0
    }

    fn citizen_arity() -> usize {
        2
    }

    fn citizen_fields() -> &'static [&'static str] {
        &["num", "den"]
    }
}

pub(crate) struct RationalValueClass {
    id: AtomicU32,
}

impl RationalValueClass {
    pub(crate) fn new() -> Self {
        Self {
            id: AtomicU32::new(0),
        }
    }

    pub(crate) fn set_id(&self, id: ClassId) {
        self.id.store(id.0, Ordering::Relaxed);
    }
}

impl Object for RationalValueClass {
    fn display(&self, _cx: &mut Cx) -> Result<String> {
        Ok(format!("#<class {}>", rational_value_class_symbol()))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl sim_kernel::ObjectCompat for RationalValueClass {
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
        Ok(Expr::Symbol(rational_value_class_symbol()))
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

impl Callable for RationalValueClass {
    fn call(&self, cx: &mut Cx, args: Args) -> Result<Value> {
        let values = args.into_vec();
        let [num, den] = values.as_slice() else {
            return Err(Error::Eval(format!(
                "class {} expects exactly two arguments",
                rational_value_class_symbol()
            )));
        };
        make_rational(cx, num.clone(), den.clone())
    }
}

impl Class for RationalValueClass {
    fn id(&self) -> ClassId {
        ClassId(self.id.load(Ordering::Relaxed))
    }

    fn symbol(&self) -> Symbol {
        rational_value_class_symbol()
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
        Ok(cx
            .registry()
            .class_by_symbol(&rational_value_class_symbol())
            .cloned())
    }

    fn members(&self, cx: &mut Cx) -> Result<TableRef> {
        cx.factory().table(Vec::new())
    }
}

impl ReadConstructor for RationalValueClass {
    fn symbol(&self) -> Symbol {
        rational_value_class_symbol()
    }

    fn args_shape(&self, cx: &mut Cx) -> Result<ShapeRef> {
        cx.factory().nil()
    }

    fn construct_read(&self, cx: &mut Cx, args: Vec<Value>) -> Result<Value> {
        self.call(cx, Args::new(args))
    }
}

pub(crate) fn make_rational(cx: &mut Cx, num: Value, den: Value) -> Result<Value> {
    build_rational(cx, num, den, false)
}

pub(crate) fn make_reduced_rational(cx: &mut Cx, num: Value, den: Value) -> Result<Value> {
    build_rational(cx, num, den, true)
}

fn build_rational(cx: &mut Cx, num: Value, den: Value, cross_reduce: bool) -> Result<Value> {
    let (num, den) = normalize_integer_parts(cx, num, den, cross_reduce)?;
    if let Some(canonical) = compact_canonical(cx, &num, &den)? {
        return cx.factory().number_literal(number_domain(), canonical);
    }
    cx.factory().opaque(Arc::new(Rational::new(num, den)))
}

pub(crate) fn expect_rational_parts(cx: &mut Cx, value: Value, side: &str) -> Result<Rational> {
    let Some(number) = cx.number_value_ref(value.clone())? else {
        return Err(Error::Eval(format!(
            "{side} operand expected number domain {}, found non-number",
            number_domain()
        )));
    };
    if number.domain != number_domain() {
        return Err(Error::Eval(format!(
            "{side} operand expected number domain {}, found {}",
            number_domain(),
            number.domain
        )));
    }
    if let Some(rational) = value.object().downcast_ref::<Rational>() {
        return Ok(rational.clone());
    }
    let literal = number.literal.ok_or_else(|| {
        Error::Eval(format!(
            "{side} operand in {} does not have a canonical rational form",
            number_domain()
        ))
    })?;
    let (num_text, den_text) = literal.canonical.split_once('/').ok_or_else(|| {
        Error::Eval(format!(
            "{side} operand was not a valid rational literal: {}",
            literal.canonical
        ))
    })?;
    Ok(Rational::new(
        parse_integer_value(cx, num_text)?,
        parse_integer_value(cx, den_text)?,
    ))
}

fn normalize_integer_parts(
    cx: &mut Cx,
    num: Value,
    den: Value,
    cross_reduce: bool,
) -> Result<(Value, Value)> {
    require_integer_value(cx, &num, "numerator")?;
    require_integer_value(cx, &den, "denominator")?;
    let Some(den_literal) = cx
        .number_value_ref(den.clone())?
        .and_then(|number| number.literal)
    else {
        return Err(Error::Eval(
            "denominator integer value does not have a canonical literal form".to_owned(),
        ));
    };
    let den_big = parse_integer_literal(&den_literal)?;
    if den_big == BigInt::from(0_u8) {
        return Err(Error::Eval(
            "rational denominator must not be zero".to_owned(),
        ));
    }

    let (mut num, mut den) = if den_big.sign() == Sign::Minus {
        (
            negate_integer_value(cx, num.clone())?,
            negate_integer_value(cx, den.clone())?,
        )
    } else {
        (num, den)
    };

    if let Some((common_num, common_den)) =
        coerce_for_reduction(cx, num.clone(), den.clone(), cross_reduce)?
    {
        let gcd = gcd_integer_values(cx, common_num.clone(), common_den.clone())?;
        if let Some(gcd_literal) = cx
            .number_value_ref(gcd.clone())?
            .and_then(|value| value.literal)
            && parse_integer_literal(&gcd_literal)? != BigInt::from(1_u8)
        {
            num = exact_divide_integer_value(cx, common_num, gcd.clone())?;
            den = exact_divide_integer_value(cx, common_den, gcd)?;
            return Ok((num, den));
        }
        num = common_num;
        den = common_den;
    }

    Ok((num, den))
}

fn require_integer_value(cx: &mut Cx, value: &Value, side: &str) -> Result<()> {
    let Some(number) = cx.number_value_ref(value.clone())? else {
        return Err(Error::Eval(format!(
            "{side} expected integer number value, found non-number"
        )));
    };
    if !is_integer_domain(&number.domain) {
        return Err(Error::Eval(format!(
            "{side} expected integer number domain, found {}",
            number.domain
        )));
    }
    Ok(())
}

fn negate_integer_value(cx: &mut Cx, value: Value) -> Result<Value> {
    cx.apply_value_number_unary_op(&Symbol::qualified("math", "neg"), value)
}

fn coerce_for_reduction(
    cx: &mut Cx,
    num: Value,
    den: Value,
    cross_reduce: bool,
) -> Result<Option<(Value, Value)>> {
    let num_ref = cx.number_value_ref(num.clone())?.ok_or_else(|| {
        Error::Eval("rational numerator lost numeric identity during normalization".to_owned())
    })?;
    let den_ref = cx.number_value_ref(den.clone())?.ok_or_else(|| {
        Error::Eval("rational denominator lost numeric identity during normalization".to_owned())
    })?;
    if num_ref.domain == den_ref.domain {
        return Ok(Some((num, den)));
    }
    if !cross_reduce
        || cx
            .registry()
            .number_domain_by_symbol(&domains::bigint())
            .is_none()
    {
        return Ok(None);
    }
    let Some(num_literal) = num_ref.literal else {
        return Ok(None);
    };
    let Some(den_literal) = den_ref.literal else {
        return Ok(None);
    };
    let num_big = parse_integer_literal(&num_literal)?;
    let den_big = parse_integer_literal(&den_literal)?;
    Ok(Some((
        cx.factory()
            .number_literal(domains::bigint(), num_big.to_string())?,
        cx.factory()
            .number_literal(domains::bigint(), den_big.to_string())?,
    )))
}

fn gcd_integer_values(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    let rem = Symbol::qualified("math", "rem");
    let mut left = absolute_integer_value(cx, left)?;
    let mut right = absolute_integer_value(cx, right)?;
    loop {
        let Some(right_literal) = cx
            .number_value_ref(right.clone())?
            .and_then(|value| value.literal)
        else {
            return Ok(left);
        };
        if parse_integer_literal(&right_literal)? == BigInt::from(0_u8) {
            return Ok(left);
        }
        let next = cx.apply_value_number_binary_op(&rem, left, right.clone())?;
        left = right;
        right = next;
    }
}

fn absolute_integer_value(cx: &mut Cx, value: Value) -> Result<Value> {
    let Some(literal) = cx
        .number_value_ref(value.clone())?
        .and_then(|value| value.literal)
    else {
        return Ok(value);
    };
    let big = parse_integer_literal(&literal)?;
    if big.sign() == Sign::Minus {
        negate_integer_value(cx, value)
    } else {
        Ok(value)
    }
}

fn exact_divide_integer_value(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    let left_ref = cx.number_value_ref(left)?.ok_or_else(|| {
        Error::Eval("exact integer division expected a numeric left operand".to_owned())
    })?;
    let right_ref = cx.number_value_ref(right)?.ok_or_else(|| {
        Error::Eval("exact integer division expected a numeric right operand".to_owned())
    })?;
    if left_ref.domain != right_ref.domain {
        return Err(Error::Eval(format!(
            "exact integer division requires a shared domain, found {} and {}",
            left_ref.domain, right_ref.domain
        )));
    }
    let left_literal = left_ref.literal.ok_or_else(|| {
        Error::Eval("exact integer division requires a canonical left integer literal".to_owned())
    })?;
    let right_literal = right_ref.literal.ok_or_else(|| {
        Error::Eval("exact integer division requires a canonical right integer literal".to_owned())
    })?;
    let left_big = parse_integer_literal(&left_literal)?;
    let right_big = parse_integer_literal(&right_literal)?;
    if right_big == BigInt::from(0_u8) {
        return Err(Error::Eval(
            "exact integer division encountered a zero divisor".to_owned(),
        ));
    }
    if (&left_big % &right_big) != BigInt::from(0_u8) {
        return Err(Error::Eval(format!(
            "exact integer division found a non-divisible pair {}/{}",
            left_big, right_big
        )));
    }
    cx.factory()
        .number_literal(left_ref.domain, (left_big / right_big).to_string())
}
