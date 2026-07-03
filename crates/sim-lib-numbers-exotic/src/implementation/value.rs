//! The `ExoticReal` trait and the lazy `ContinuedFraction` value: its
//! coefficient stream, f64 truncation, and rational approximation as a non-
//! citizen handle reconstructed from a `numbers/cf` descriptor.

use std::{
    any::Any,
    cmp::Ordering,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering as AtomicOrdering},
    },
};

use num_bigint::BigInt;
use sim_kernel::{
    ClassRef, Cx, DefaultFactory, Error, Expr, Factory, LengthResult, ListValue, NumberLiteral,
    NumberValue, Object, Result, Symbol, Value,
};
use sim_lib_numbers_core::domains;

use super::domain::number_domain;

/// An exotic real number carried as a lazily evaluated stream, the contract
/// every `numbers/cf` value implements. Implementors expose work-bounded
/// truncations rather than a closed-form value.
pub trait ExoticReal: Send + Sync + 'static {
    /// Truncates the real to an `f64` using at most `max_work` coefficients,
    /// returning the approximation and the number of coefficients consumed.
    fn as_f64(&self, max_work: usize) -> (f64, u32);

    /// Truncates the real to an exact rational approximation using at most
    /// `max_work` coefficients, or `None` when no rational surface is available.
    fn truncate_rational(&self, cx: &mut Cx, max_work: usize) -> Result<Option<Value>>;

    /// The stable name of this exotic real (for example `phi` or `e`).
    fn name(&self) -> &'static str;
}

#[sim_citizen_derive::non_citizen(
    reason = "continued-fraction value can carry a live coefficient generator; reconstruct from numbers/cf descriptor data",
    kind = "handle",
    descriptor = "numbers/cf"
)]
/// A lazy continued-fraction real: a fixed head of coefficients followed by an
/// optional generated tail, the sole `ExoticReal` implementor in this crate.
pub struct ContinuedFraction {
    /// The leading coefficients known up front.
    pub head: Vec<i128>,
    /// The optional lazy tail generating further coefficients on demand; `None`
    /// for a finite continued fraction.
    pub tail: Option<CfTail>,
    symbol: Symbol,
    name: &'static str,
    cache: Arc<Mutex<Vec<i128>>>,
}

impl ContinuedFraction {
    pub(crate) fn builtin(
        symbol: Symbol,
        name: &'static str,
        head: Vec<i128>,
        tail: Option<CfTail>,
    ) -> Self {
        Self {
            cache: Arc::new(Mutex::new(head.clone())),
            head,
            tail,
            symbol,
            name,
        }
    }

    fn ensure_coeff(&self, index: usize) -> Result<Option<i128>> {
        let mut cache = self
            .cache
            .lock()
            .map_err(|_| Error::HostError("continued fraction cache lock poisoned".to_owned()))?;
        while cache.len() <= index {
            let Some(tail) = self.tail.as_ref() else {
                return Ok(None);
            };
            let Some(next) = tail.next()? else {
                return Ok(None);
            };
            cache.push(next);
        }
        Ok(cache.get(index).copied())
    }

    fn coeffs(&self, max_work: usize) -> Result<Vec<i128>> {
        let mut out = Vec::new();
        for index in 0..max_work {
            let Some(coeff) = self.ensure_coeff(index)? else {
                break;
            };
            out.push(coeff);
        }
        Ok(out)
    }

    fn coeff_value(&self, cx: &mut Cx, index: usize) -> Result<Option<Value>> {
        let Some(coeff) = self.ensure_coeff(index)? else {
            return Ok(None);
        };
        cx.factory()
            .number_literal(domains::i64(), coeff.to_string())
            .map(Some)
    }

    fn tail_kind(&self) -> &'static str {
        match self.tail.as_ref() {
            None => "finite",
            Some(tail) if tail.is_endless() => "endless",
            Some(_) => "lazy",
        }
    }
}

impl ExoticReal for ContinuedFraction {
    fn as_f64(&self, max_work: usize) -> (f64, u32) {
        let coeffs = self.coeffs(max_work).unwrap_or_default();
        if coeffs.is_empty() {
            return (f64::NAN, 0);
        }
        let mut value = *coeffs.last().unwrap() as f64;
        for coeff in coeffs[..coeffs.len() - 1].iter().rev() {
            value = *coeff as f64 + (1.0 / value);
        }
        (value, coeffs.len() as u32)
    }

    fn truncate_rational(&self, cx: &mut Cx, max_work: usize) -> Result<Option<Value>> {
        if cx
            .registry()
            .number_domain_by_symbol(&domains::rational())
            .is_none()
        {
            return Ok(None);
        }
        let coeffs = self.coeffs(max_work)?;
        if coeffs.is_empty() {
            return Ok(None);
        }
        let (num, den) = convergent_ratio(&coeffs);
        cx.factory()
            .number_literal(domains::rational(), format!("{num}/{den}"))
            .map(Some)
    }

    fn name(&self) -> &'static str {
        self.name
    }
}

impl Object for ContinuedFraction {
    fn display(&self, _cx: &mut Cx) -> Result<String> {
        Ok(format!("#<continued-fraction {}>", self.symbol))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl sim_kernel::ObjectCompat for ContinuedFraction {
    fn class(&self, cx: &mut Cx) -> Result<ClassRef> {
        if let Some(value) = cx
            .registry()
            .class_by_symbol(&Symbol::qualified("core", "Number"))
        {
            return Ok(value.clone());
        }
        DefaultFactory.class_stub(
            sim_kernel::CORE_NUMBER_CLASS_ID,
            Symbol::qualified("core", "Number"),
        )
    }
    fn as_expr(&self, _cx: &mut Cx) -> Result<Expr> {
        Ok(Expr::Symbol(self.symbol.clone()))
    }
    fn as_table(&self, cx: &mut Cx) -> Result<Value> {
        let head = self
            .head
            .iter()
            .map(|coeff| {
                cx.factory()
                    .number_literal(domains::i64(), coeff.to_string())
            })
            .collect::<Result<Vec<_>>>()?;
        cx.factory().table(vec![
            (
                Symbol::new("kind"),
                cx.factory().string("continued-fraction".to_owned())?,
            ),
            (Symbol::new("domain"), cx.factory().symbol(number_domain())?),
            (
                Symbol::new("name"),
                cx.factory().string(self.name.to_owned())?,
            ),
            (Symbol::new("head"), cx.factory().list(head)?),
            (
                Symbol::new("tail"),
                cx.factory().string(self.tail_kind().to_owned())?,
            ),
        ])
    }
    fn as_number_value(&self) -> Option<&dyn NumberValue> {
        Some(self)
    }
    fn as_list(&self) -> Option<&dyn ListValue> {
        Some(self)
    }
}

impl NumberValue for ContinuedFraction {
    fn number_domain(&self, _cx: &mut Cx) -> Result<Symbol> {
        Ok(number_domain())
    }

    fn number_literal(&self, _cx: &mut Cx) -> Result<Option<NumberLiteral>> {
        Ok(None)
    }
}

impl ListValue for ContinuedFraction {
    fn is_empty(&self, _cx: &mut Cx) -> Result<bool> {
        Ok(self.ensure_coeff(0)?.is_none())
    }

    fn car(&self, cx: &mut Cx) -> Result<Option<Value>> {
        self.coeff_value(cx, 0)
    }

    fn cdr(&self, cx: &mut Cx) -> Result<Option<Value>> {
        if self.ensure_coeff(1)?.is_none() {
            return Ok(Some(cx.factory().list(Vec::new())?));
        }
        cx.factory()
            .opaque(Arc::new(CoefficientList::new(Arc::new(self.clone_cf()), 1)))
            .map(Some)
    }

    fn len(&self, _cx: &mut Cx) -> Result<LengthResult> {
        Ok(match self.tail {
            None => LengthResult::Known(self.head.len()),
            Some(_) => LengthResult::Unknown,
        })
    }

    fn len_cmp(&self, _cx: &mut Cx, n: usize) -> Result<Ordering> {
        if self.tail.is_some() {
            return Ok(if self.head.len() > n {
                Ordering::Greater
            } else {
                Ordering::Equal
            });
        }
        Ok(self.head.len().cmp(&n))
    }

    fn get(&self, cx: &mut Cx, index: usize) -> Result<Option<Value>> {
        self.coeff_value(cx, index)
    }
}

impl ContinuedFraction {
    fn clone_cf(&self) -> Self {
        Self {
            head: self.head.clone(),
            tail: self.tail.clone(),
            symbol: self.symbol.clone(),
            name: self.name,
            cache: self.cache.clone(),
        }
    }
}

#[derive(Clone)]
pub struct CfTail {
    inner: Arc<CfTailInner>,
}

struct CfTailInner {
    generator: Mutex<Box<dyn FnMut() -> Option<i128> + Send + Sync>>,
    exhausted: AtomicBool,
    endless: bool,
}

impl CfTail {
    pub fn lazy(generator: impl FnMut() -> Option<i128> + Send + Sync + 'static) -> Self {
        Self::new(false, generator)
    }

    pub fn endless(generator: impl FnMut() -> Option<i128> + Send + Sync + 'static) -> Self {
        Self::new(true, generator)
    }

    fn new(endless: bool, generator: impl FnMut() -> Option<i128> + Send + Sync + 'static) -> Self {
        Self {
            inner: Arc::new(CfTailInner {
                generator: Mutex::new(Box::new(generator)),
                exhausted: AtomicBool::new(false),
                endless,
            }),
        }
    }

    fn next(&self) -> Result<Option<i128>> {
        if self.inner.exhausted.load(AtomicOrdering::Relaxed) {
            return Ok(None);
        }
        let mut generator =
            self.inner.generator.lock().map_err(|_| {
                Error::HostError("continued fraction tail lock poisoned".to_owned())
            })?;
        let next = generator.as_mut()();
        if next.is_none() {
            self.inner.exhausted.store(true, AtomicOrdering::Relaxed);
        }
        Ok(next)
    }

    fn is_endless(&self) -> bool {
        self.inner.endless
    }
}

#[sim_citizen_derive::non_citizen(
    reason = "continued-fraction coefficient list view; descriptor is the source continued-fraction value",
    kind = "handle",
    descriptor = "numbers/cf"
)]
#[derive(Clone)]
struct CoefficientList {
    source: Arc<ContinuedFraction>,
    index: usize,
}

impl CoefficientList {
    fn new(source: Arc<ContinuedFraction>, index: usize) -> Self {
        Self { source, index }
    }
}

impl Object for CoefficientList {
    fn display(&self, _cx: &mut Cx) -> Result<String> {
        Ok(format!("#<cf-coefficients {}>", self.source.name))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl sim_kernel::ObjectCompat for CoefficientList {
    fn class(&self, cx: &mut Cx) -> Result<ClassRef> {
        if let Some(value) = cx
            .registry()
            .class_by_symbol(&Symbol::qualified("core", "List"))
        {
            return Ok(value.clone());
        }
        DefaultFactory.class_stub(
            sim_kernel::CORE_LIST_CLASS_ID,
            Symbol::qualified("core", "List"),
        )
    }
    fn as_list(&self) -> Option<&dyn ListValue> {
        Some(self)
    }
}

impl ListValue for CoefficientList {
    fn is_empty(&self, _cx: &mut Cx) -> Result<bool> {
        Ok(self.source.ensure_coeff(self.index)?.is_none())
    }

    fn car(&self, cx: &mut Cx) -> Result<Option<Value>> {
        self.source.coeff_value(cx, self.index)
    }

    fn cdr(&self, cx: &mut Cx) -> Result<Option<Value>> {
        if self.source.ensure_coeff(self.index + 1)?.is_none() {
            return Ok(Some(cx.factory().list(Vec::new())?));
        }
        cx.factory()
            .opaque(Arc::new(Self::new(self.source.clone(), self.index + 1)))
            .map(Some)
    }

    fn len(&self, _cx: &mut Cx) -> Result<LengthResult> {
        Ok(match self.source.tail {
            None => LengthResult::Known(self.source.head.len().saturating_sub(self.index)),
            Some(_) => LengthResult::Unknown,
        })
    }

    fn get(&self, cx: &mut Cx, index: usize) -> Result<Option<Value>> {
        self.source.coeff_value(cx, self.index + index)
    }
}

enum IntAcc {
    Small(i128),
    Big(BigInt),
}

impl IntAcc {
    fn zero() -> Self {
        Self::Small(0)
    }

    fn one() -> Self {
        Self::Small(1)
    }

    fn mul_add(&self, factor: i128, addend: &Self) -> Self {
        match (self, addend) {
            (Self::Small(left), Self::Small(right)) => {
                if let Some(product) = left.checked_mul(factor)
                    && let Some(sum) = product.checked_add(*right)
                {
                    return Self::Small(sum);
                }
                Self::Big(BigInt::from(*left) * BigInt::from(factor) + BigInt::from(*right))
            }
            (left, right) => {
                Self::Big(left.clone().into_big() * BigInt::from(factor) + right.clone().into_big())
            }
        }
    }

    fn into_big(self) -> BigInt {
        match self {
            Self::Small(value) => BigInt::from(value),
            Self::Big(value) => value,
        }
    }
}

impl From<BigInt> for IntAcc {
    fn from(value: BigInt) -> Self {
        Self::Big(value)
    }
}

impl Clone for IntAcc {
    fn clone(&self) -> Self {
        match self {
            Self::Small(value) => Self::Small(*value),
            Self::Big(value) => Self::Big(value.clone()),
        }
    }
}

fn convergent_ratio(coeffs: &[i128]) -> (BigInt, BigInt) {
    let mut p_prev2 = IntAcc::zero();
    let mut p_prev1 = IntAcc::one();
    let mut q_prev2 = IntAcc::one();
    let mut q_prev1 = IntAcc::zero();
    for coeff in coeffs {
        let p = p_prev1.mul_add(*coeff, &p_prev2);
        let q = q_prev1.mul_add(*coeff, &q_prev2);
        p_prev2 = p_prev1;
        p_prev1 = p;
        q_prev2 = q_prev1;
        q_prev1 = q;
    }
    (p_prev1.into_big(), q_prev1.into_big())
}
