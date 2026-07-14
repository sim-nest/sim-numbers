//! Backend plugin traits and option types: `Differentiator`, `Quadrature`, and
//! `OdeSolver`, plus the callable adapter, options, and problem records.

use sim_kernel::{Args, Cx, Error, Result, Symbol, Value};
use sim_lib_numbers_cas::CasExpr;
use sim_lib_numbers_func::Func;

/// Common interface for every numeric backend plugin: it reports its method
/// name and the kind of operation it implements.
pub trait NumericPlugin: Send + Sync + 'static {
    /// The method name this plugin registers under (for example, `central` or `rk4`).
    fn name(&self) -> Symbol;
    /// The numeric operation kind this plugin implements.
    fn kind(&self) -> NumericKind;
}

/// The category of numeric backend, used to route a plugin to the right slot in
/// the registry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumericKind {
    /// A differentiator backend (numeric derivative at a point).
    Differentiator,
    /// A fixed-rule quadrature backend (definite integral with a set rule).
    QuadratureFixed,
    /// An adaptive quadrature backend (definite integral to a tolerance).
    QuadratureAdaptive,
    /// A fixed-step ODE solver backend.
    OdeFixed,
    /// An adaptive-step ODE solver backend.
    OdeAdaptive,
    /// An implicit differential-algebraic equation solver backend.
    DaeImplicit,
}

/// Options controlling a numeric-differentiation call.
///
/// `method = auto` first uses a symbolic CAS body when the function has one,
/// then a registered differentiator named by the function's
/// `differentiator_hint`, then central finite difference.
#[derive(Clone, Debug)]
pub struct DiffOpts {
    /// The differentiator method to use, or `auto` to follow the automatic
    /// symbolic, hinted-exact, then finite-difference order.
    pub method: Symbol,
    /// The finite-difference step size.
    pub h: f64,
}

impl DiffOpts {
    /// Returns default options: the `auto` method with a small default step.
    ///
    /// # Examples
    ///
    /// ```
    /// use sim_lib_numbers_numeric::DiffOpts;
    ///
    /// let opts = DiffOpts::auto();
    /// assert_eq!(opts.method.to_string(), "auto");
    /// assert!(opts.h > 0.0);
    /// ```
    pub fn auto() -> Self {
        Self {
            method: Symbol::new("auto"),
            h: 1.0e-6,
        }
    }
}

/// Options controlling an integration (quadrature) call.
#[derive(Clone, Debug)]
pub struct QuadOpts {
    /// The quadrature method to use, or `auto` to let the registry choose.
    pub method: Symbol,
    /// The number of subdivisions, for fixed-rule quadrature.
    pub n: Option<usize>,
    /// The error tolerance, for adaptive quadrature.
    pub tol: Option<f64>,
}

impl QuadOpts {
    /// Returns default options for fixed-rule integration (`auto`, no tolerance).
    pub fn fixed_default() -> Self {
        Self {
            method: Symbol::new("auto"),
            n: None,
            tol: None,
        }
    }

    /// Returns default options for adaptive integration (`auto` with a tolerance).
    ///
    /// # Examples
    ///
    /// ```
    /// use sim_lib_numbers_numeric::QuadOpts;
    ///
    /// let fixed = QuadOpts::fixed_default();
    /// assert!(fixed.tol.is_none());
    ///
    /// let adaptive = QuadOpts::adaptive_default();
    /// assert!(adaptive.tol.is_some());
    /// ```
    pub fn adaptive_default() -> Self {
        Self {
            method: Symbol::new("auto"),
            n: None,
            tol: Some(1.0e-10),
        }
    }
}

/// Options controlling an ODE-solve call.
#[derive(Clone, Debug)]
pub struct OdeOpts {
    /// The ODE solver method to use, or `auto` to let the registry choose.
    pub method: Symbol,
    /// The fixed step size, for fixed-step solvers.
    pub h: Option<f64>,
    /// The error tolerance, for adaptive solvers.
    pub tol: Option<f64>,
    /// An optional cap on the number of integration steps.
    pub max_steps: Option<usize>,
}

impl OdeOpts {
    /// Returns default options for an adaptive ODE solve (`auto` with a tolerance).
    ///
    /// # Examples
    ///
    /// ```
    /// use sim_lib_numbers_numeric::OdeOpts;
    ///
    /// let opts = OdeOpts::default_adaptive();
    /// assert_eq!(opts.method.to_string(), "auto");
    /// assert!(opts.tol.is_some());
    /// assert!(opts.h.is_none());
    /// ```
    pub fn default_adaptive() -> Self {
        Self {
            method: Symbol::new("auto"),
            h: None,
            tol: Some(1.0e-8),
            max_steps: None,
        }
    }
}

/// A numeric-method callable, either a `Func` value or any ordinary callable
/// runtime value with the variable symbols supplied by the numeric surface.
#[derive(Clone)]
pub struct NumericCallable {
    value: Value,
    vars: Vec<Symbol>,
    body_cas: Option<CasExpr>,
    differentiator_hint: Option<Symbol>,
}

enum FuncVarPolicy {
    Exact,
    UseFuncVarsWithArity,
}

impl NumericCallable {
    /// Builds a unary callable whose `Func` parameter, when present, must match
    /// the requested variable.
    pub fn unary(value: Value, var: Symbol) -> Result<Self> {
        Self::from_value(
            value,
            vec![var],
            FuncVarPolicy::Exact,
            "numeric methods require the Func parameter to match the requested variable",
            "numeric methods expect a Func or ordinary callable",
        )
    }

    /// Builds a binary callable whose `Func` parameters, when present, must
    /// match the requested independent and dependent variables.
    pub fn binary(value: Value, var: Symbol, y_var: Symbol) -> Result<Self> {
        Self::from_value(
            value,
            vec![var, y_var],
            FuncVarPolicy::Exact,
            "ode-solve requires the Func parameters to match the requested x and y variables",
            "ode-solve expects a Func or ordinary callable",
        )
    }

    /// Builds a unary sampling callable. `Func` values keep their own variable
    /// name; ordinary callables use `fallback_var`.
    pub fn sampled_unary(value: Value, fallback_var: Symbol) -> Result<Self> {
        Self::from_value(
            value,
            vec![fallback_var],
            FuncVarPolicy::UseFuncVarsWithArity,
            "numeric sampling requires a unary Func",
            "numeric sampling expects a Func or ordinary callable",
        )
    }

    /// Builds a binary sampling callable. `Func` values keep their own variable
    /// names; ordinary callables use the supplied fallback names.
    pub fn sampled_binary(
        value: Value,
        fallback_var: Symbol,
        fallback_y_var: Symbol,
    ) -> Result<Self> {
        Self::from_value(
            value,
            vec![fallback_var, fallback_y_var],
            FuncVarPolicy::UseFuncVarsWithArity,
            "numeric sampling requires a binary Func",
            "numeric sampling expects a Func or ordinary callable",
        )
    }

    fn from_value(
        value: Value,
        requested_vars: Vec<Symbol>,
        policy: FuncVarPolicy,
        func_mismatch: &str,
        callable_mismatch: &str,
    ) -> Result<Self> {
        if let Some(func) = value.object().downcast_ref::<Func>() {
            let vars = match policy {
                FuncVarPolicy::Exact => {
                    if func.vars.as_slice() != requested_vars.as_slice() {
                        return Err(Error::Eval(func_mismatch.to_owned()));
                    }
                    requested_vars
                }
                FuncVarPolicy::UseFuncVarsWithArity => {
                    if func.vars.len() != requested_vars.len() {
                        return Err(Error::Eval(func_mismatch.to_owned()));
                    }
                    func.vars.clone()
                }
            };
            let body_cas = func.body_cas().cloned();
            let differentiator_hint = func.metadata.differentiator_hint.clone();
            return Ok(Self {
                value,
                vars,
                body_cas,
                differentiator_hint,
            });
        }
        value
            .object()
            .as_callable()
            .ok_or_else(|| Error::Eval(callable_mismatch.to_owned()))?;
        Ok(Self {
            value,
            vars: requested_vars,
            body_cas: None,
            differentiator_hint: None,
        })
    }

    /// Calls the wrapped value with numeric sample arguments.
    pub fn call(&self, cx: &mut Cx, args: Vec<Value>) -> Result<Value> {
        cx.call_value(self.value.clone(), Args::new(args))
    }

    /// The wrapped runtime value.
    pub fn value(&self) -> &Value {
        &self.value
    }

    /// The wrapped `Func`, when this callable is backed by one.
    pub fn as_func(&self) -> Option<&Func> {
        self.value.object().downcast_ref::<Func>()
    }

    /// Variable symbols associated with this numeric callable.
    pub fn vars(&self) -> &[Symbol] {
        &self.vars
    }

    /// The symbolic CAS body when the wrapped value is a symbolic `Func`.
    pub fn body_cas(&self) -> Option<&CasExpr> {
        self.body_cas.as_ref()
    }

    /// The exact differentiator hint when the wrapped value is a hinted `Func`.
    pub fn differentiator_hint(&self) -> Option<&Symbol> {
        self.differentiator_hint.as_ref()
    }
}

/// A first-order initial-value ODE problem `dy/dx = f(x, y)` handed to an
/// [`OdeSolver`].
pub struct OdeProblem<'a> {
    /// The right-hand-side function giving the derivative.
    pub dy: &'a NumericCallable,
    /// The independent-variable symbol (typically `x`).
    pub var: &'a Symbol,
    /// The dependent-variable symbol (typically `y`).
    pub y_var: &'a Symbol,
    /// The initial value of the independent variable.
    pub x0: &'a Value,
    /// The initial value of the dependent variable.
    pub y0: &'a Value,
    /// The end value of the independent variable to integrate toward.
    pub x_end: &'a Value,
}

/// A numeric differentiation backend: computes `df/dvar` at a point.
pub trait Differentiator: NumericPlugin {
    /// Evaluates the numeric derivative of `f` with respect to `var` at `point`.
    fn diff_at(
        &self,
        cx: &mut Cx,
        f: &Func,
        var: &Symbol,
        point: &Value,
        opt: DiffOpts,
    ) -> Result<Value>;

    /// Evaluates the numeric derivative for any numeric callable.
    ///
    /// Differentiators that only sample their input should override this hook.
    /// Exact differentiators that require `Func` metadata inherit the fail-closed
    /// `Func` adapter.
    fn diff_callable_at(
        &self,
        cx: &mut Cx,
        f: &NumericCallable,
        var: &Symbol,
        point: &Value,
        opt: DiffOpts,
    ) -> Result<Value> {
        let func = f
            .as_func()
            .ok_or_else(|| Error::Eval("differentiator requires a Func value".to_owned()))?;
        self.diff_at(cx, func, var, point, opt)
    }
}

/// A numeric integration (quadrature) backend: computes a definite integral.
pub trait Quadrature: NumericPlugin {
    /// Integrates `f` over `var` from `lo` to `hi`.
    fn integrate(
        &self,
        cx: &mut Cx,
        f: &NumericCallable,
        var: &Symbol,
        lo: &Value,
        hi: &Value,
        opt: QuadOpts,
    ) -> Result<Value>;
}

/// A numeric ODE-solving backend: integrates an initial-value problem.
pub trait OdeSolver: NumericPlugin {
    /// Solves `problem`, returning the sampled `(x, y)` points of the trajectory.
    fn solve(
        &self,
        cx: &mut Cx,
        problem: OdeProblem<'_>,
        opt: OdeOpts,
    ) -> Result<Vec<(Value, Value)>>;
}
