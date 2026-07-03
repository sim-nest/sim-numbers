//! The linear-algebra library and its operation callables, registering the
//! `dot`/`matmul`/`det`/`inv`/... symbols that dispatch into the `ops` module.

use std::any::Any;
use std::sync::Arc;

use sim_kernel::{
    AbiVersion, Args, Callable, ClassRef, DefaultFactory, Dependency, Export, Expr, Factory, Lib,
    LibManifest, LibTarget, Linker, Object, Result, Symbol, Value, Version,
};
use sim_lib_numbers_core::domains;

use super::ops;

fn function_symbols() -> [Symbol; 11] {
    [
        Symbol::new("dot"),
        Symbol::new("matmul"),
        Symbol::new("cross"),
        Symbol::new("transpose"),
        Symbol::new("det"),
        Symbol::new("inv"),
        Symbol::new("trace"),
        Symbol::new("norm"),
        Symbol::new("eye"),
        Symbol::new("zeros"),
        Symbol::new("ones"),
    ]
}

#[derive(Clone)]
struct LinalgFunction {
    symbol: Symbol,
}

impl Object for LinalgFunction {
    fn display(&self, _cx: &mut sim_kernel::Cx) -> Result<String> {
        Ok(format!("#<function {}>", self.symbol))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl sim_kernel::ObjectCompat for LinalgFunction {
    fn class(&self, cx: &mut sim_kernel::Cx) -> Result<ClassRef> {
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
    fn as_expr(&self, _cx: &mut sim_kernel::Cx) -> Result<Expr> {
        Ok(Expr::Symbol(self.symbol.clone()))
    }
    fn as_callable(&self) -> Option<&dyn Callable> {
        Some(self)
    }
}

impl Callable for LinalgFunction {
    fn call(&self, cx: &mut sim_kernel::Cx, args: Args) -> Result<Value> {
        ops::dispatch(cx, &self.symbol, args.into_vec())
    }
}

/// Registered library that installs linear-algebra functions over tensors.
///
/// Loading this [`Lib`] registers eleven callable functions -- `dot`, `matmul`,
/// `cross`, `transpose`, `det`, `inv`, `trace`, `norm`, and the `eye`, `zeros`,
/// and `ones` constructors -- each dispatching into the operation
/// implementations against the base tensor domain.
pub struct TensorLinalgLib;

impl TensorLinalgLib {
    /// Creates the linear-algebra library. The value is stateless; the function
    /// values are installed when it is loaded into a [`Cx`](sim_kernel::Cx).
    pub fn new() -> Self {
        Self
    }
}

impl Default for TensorLinalgLib {
    fn default() -> Self {
        Self::new()
    }
}

impl Lib for TensorLinalgLib {
    fn manifest(&self) -> LibManifest {
        LibManifest {
            id: domains::tensor_linalg(),
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
                DefaultFactory.opaque(Arc::new(LinalgFunction { symbol }))?,
            )?;
        }
        Ok(())
    }
}
