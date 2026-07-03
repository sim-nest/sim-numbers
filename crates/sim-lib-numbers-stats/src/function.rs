//! Runtime library surface for statistics functions.

use std::{any::Any, sync::Arc};

use sim_kernel::{
    AbiVersion, Args, Callable, ClassRef, Cx, DefaultFactory, Dependency, Export, Expr, Factory,
    Lib, LibManifest, LibTarget, Linker, Object, RawArgs, Result, Symbol, Value, Version,
};

use super::runtime;

/// Returns the symbol bound to the `stats/disparate-impact-claim` operation.
pub fn stats_disparate_impact_claim_symbol() -> Symbol {
    Symbol::qualified("stats", "disparate-impact-claim")
}

/// Returns the symbol bound to the `stats/mean-claim` operation.
pub fn stats_mean_claim_symbol() -> Symbol {
    Symbol::qualified("stats", "mean-claim")
}

/// Returns the symbol bound to the `stats/variance-claim` operation.
pub fn stats_variance_claim_symbol() -> Symbol {
    Symbol::qualified("stats", "variance-claim")
}

/// Returns the symbol bound to the `stats/entropy-claim` operation.
pub fn stats_entropy_claim_symbol() -> Symbol {
    Symbol::qualified("stats", "entropy-claim")
}

/// Returns the symbol bound to the `stats/claims` batch operation.
pub fn stats_claims_symbol() -> Symbol {
    Symbol::qualified("stats", "claims")
}

fn function_symbols() -> [Symbol; 5] {
    [
        stats_mean_claim_symbol(),
        stats_variance_claim_symbol(),
        stats_entropy_claim_symbol(),
        stats_disparate_impact_claim_symbol(),
        stats_claims_symbol(),
    ]
}

#[derive(Clone)]
struct StatsFunction {
    symbol: Symbol,
}

impl Object for StatsFunction {
    fn display(&self, _cx: &mut Cx) -> Result<String> {
        Ok(format!("#<function {}>", self.symbol))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl sim_kernel::ObjectCompat for StatsFunction {
    fn class(&self, cx: &mut Cx) -> Result<ClassRef> {
        if let Some(value) = cx
            .registry()
            .class_by_symbol(&Symbol::qualified("core", "Function"))
        {
            return Ok(value.clone());
        }
        DefaultFactory.class_stub(
            sim_kernel::CORE_FUNCTION_CLASS_ID,
            Symbol::qualified("core", "Function"),
        )
    }

    fn as_expr(&self, _cx: &mut Cx) -> Result<Expr> {
        Ok(Expr::Symbol(self.symbol.clone()))
    }

    fn as_callable(&self) -> Option<&dyn Callable> {
        Some(self)
    }
}

impl Callable for StatsFunction {
    fn call(&self, cx: &mut Cx, args: Args) -> Result<Value> {
        if self.symbol == stats_mean_claim_symbol() {
            return runtime::call_stats_mean_claim(cx, args);
        }
        if self.symbol == stats_variance_claim_symbol() {
            return runtime::call_stats_variance_claim(cx, args);
        }
        if self.symbol == stats_entropy_claim_symbol() {
            return runtime::call_stats_entropy_claim(cx, args);
        }
        if self.symbol == stats_disparate_impact_claim_symbol() {
            return runtime::call_stats_disparate_impact_claim(cx, args);
        }
        if self.symbol == stats_claims_symbol() {
            return runtime::call_stats_claims(cx, args);
        }
        unreachable!("unregistered stats function {}", self.symbol)
    }

    fn call_exprs(&self, cx: &mut Cx, args: RawArgs) -> Result<Value> {
        let values = args
            .into_exprs()
            .into_iter()
            .map(|expr| cx.eval_expr(expr))
            .collect::<Result<Vec<_>>>()?;
        self.call(cx, Args::new(values))
    }
}

/// Library that installs the runtime statistics functions.
pub struct StatsNumbersLib;

impl StatsNumbersLib {
    /// Creates a new statistics runtime library.
    pub fn new() -> Self {
        Self
    }
}

impl Default for StatsNumbersLib {
    fn default() -> Self {
        Self::new()
    }
}

impl Lib for StatsNumbersLib {
    fn manifest(&self) -> LibManifest {
        LibManifest {
            id: Symbol::qualified("numbers", "stats"),
            version: Version(env!("CARGO_PKG_VERSION").to_owned()),
            abi: AbiVersion { major: 0, minor: 1 },
            target: LibTarget::HostRegistered,
            requires: Vec::<Dependency>::new(),
            capabilities: Vec::new(),
            exports: function_symbols()
                .into_iter()
                .map(|symbol| Export::Function {
                    symbol,
                    function_id: None,
                })
                .collect(),
        }
    }

    fn load(&self, _cx: &mut sim_kernel::LoadCx, linker: &mut Linker<'_>) -> Result<()> {
        for symbol in function_symbols() {
            linker.function_value(
                symbol.clone(),
                DefaultFactory.opaque(Arc::new(StatsFunction { symbol }))?,
            )?;
        }
        Ok(())
    }
}
