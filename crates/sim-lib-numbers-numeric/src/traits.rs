//! Backend plugin traits and option types: `Differentiator`, `Quadrature`, and
//! `OdeSolver`, plus the `DiffOpts`/`QuadOpts`/`OdeOpts` and problem records.

use sim_kernel::{Cx, Result, Symbol, Value};
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
#[derive(Clone, Debug)]
pub struct DiffOpts {
    /// The differentiator method to use, or `auto` to let the registry choose.
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

/// A first-order initial-value ODE problem `dy/dx = f(x, y)` handed to an
/// [`OdeSolver`].
pub struct OdeProblem<'a> {
    /// The right-hand-side function giving the derivative.
    pub dy: &'a Func,
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
}

/// A numeric integration (quadrature) backend: computes a definite integral.
pub trait Quadrature: NumericPlugin {
    /// Integrates `f` over `var` from `lo` to `hi`.
    fn integrate(
        &self,
        cx: &mut Cx,
        f: &Func,
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
