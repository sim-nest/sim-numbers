//! Internal layout of the CAS evaluation library: the `eval` engine over
//! `CasExpr` and the `function` `Lib` that exposes `eval-cas` to the runtime.

mod eval;
mod function;

pub use eval::{cas_to_expr, eval_cas, eval_cas_symbol, eval_cas_symbolic, expr_to_cas};
pub use function::CasEvalLib;
