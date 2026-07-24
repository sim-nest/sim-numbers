//! Tensor execution site over the kernel EvalFabric contract.

use std::sync::Arc;

use sim_kernel::{
    CapabilityName, ClassRef, Cx, DefaultFactory, Env, Error, EvalFabric, EvalReply, EvalRequest,
    Factory, Object, Result, ShapeId, Symbol, Value,
};

use super::execution::{
    CpuTensorExecutor, TensorExecutor, TensorExecutorCard, tensor_executor_symbol,
    tensor_executor_value, tensor_site_symbol,
};

/// Eval-fabric wrapper that binds one tensor executor for a realization.
#[derive(Clone)]
pub struct TensorSite {
    symbol: Symbol,
    executor: Arc<dyn TensorExecutor>,
    capabilities: Arc<[CapabilityName]>,
}

impl TensorSite {
    /// Builds a tensor site around an executor and required capability set.
    pub fn new(
        symbol: Symbol,
        executor: Arc<dyn TensorExecutor>,
        capabilities: Vec<CapabilityName>,
    ) -> Self {
        Self {
            symbol,
            executor,
            capabilities: capabilities.into(),
        }
    }

    /// Builds the default local tensor site using the CPU executor.
    pub fn local_cpu() -> Self {
        Self::new(
            tensor_site_symbol(),
            Arc::new(CpuTensorExecutor::new()),
            Vec::new(),
        )
    }

    /// Returns the site symbol.
    pub fn symbol(&self) -> &Symbol {
        &self.symbol
    }

    /// Returns the executor card exposed by this site.
    pub fn card(&self) -> TensorExecutorCard {
        self.executor.card()
    }

    /// Returns the capabilities this site requires for realization.
    pub fn capabilities(&self) -> &[CapabilityName] {
        &self.capabilities
    }
}

impl EvalFabric for TensorSite {
    fn realize(&self, cx: &mut Cx, request: EvalRequest) -> Result<EvalReply> {
        cx.require_all(&self.capabilities)?;
        cx.require_all(&request.required_capabilities)?;

        let executor = tensor_executor_value(self.executor.clone())?;
        let mut child = Env::child(Arc::new(cx.env().clone()));
        child.define(tensor_executor_symbol(), executor);
        let value = cx.with_env(child, |cx| cx.eval_expr(request.expr))?;
        if let Some(shape_value) = request.result_shape.clone() {
            check_result_shape(cx, &shape_value, value.clone())?;
        }
        Ok(EvalReply {
            value,
            diagnostics: Vec::new(),
            trace: request.trace.then(|| {
                DefaultFactory
                    .symbol(Symbol::qualified("tensor", "trace/local"))
                    .expect("trace symbol should be boxable")
            }),
        })
    }
}

impl Object for TensorSite {
    fn display(&self, _cx: &mut Cx) -> Result<String> {
        Ok(format!("#<tensor-site {}>", self.symbol))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl sim_kernel::ObjectCompat for TensorSite {
    fn class(&self, cx: &mut Cx) -> Result<ClassRef> {
        if let Some(value) = cx
            .registry()
            .class_by_symbol(&Symbol::qualified("core", "EvalFabric"))
        {
            return Ok(value.clone());
        }
        DefaultFactory.class_stub(
            sim_kernel::CORE_EVAL_REQUEST_CLASS_ID,
            Symbol::qualified("core", "EvalFabric"),
        )
    }

    fn as_eval_fabric(&self) -> Option<&dyn EvalFabric> {
        Some(self)
    }

    fn as_table(&self, cx: &mut Cx) -> Result<Value> {
        let card = self.card();
        let operations = card
            .operations
            .iter()
            .map(|symbol| cx.factory().symbol(symbol.clone()))
            .collect::<Result<Vec<_>>>()?;
        let device_capability = match card.device_capability {
            Some(capability) => cx.factory().string(capability.as_str().to_owned())?,
            None => cx.factory().nil()?,
        };
        cx.factory().table(vec![
            (
                Symbol::new("site"),
                cx.factory().symbol(self.symbol.clone())?,
            ),
            (Symbol::new("executor"), cx.factory().symbol(card.symbol)?),
            (Symbol::new("provider"), cx.factory().string(card.provider)?),
            (Symbol::new("locality"), cx.factory().symbol(card.locality)?),
            (Symbol::new("operations"), cx.factory().list(operations)?),
            (Symbol::new("device-capability"), device_capability),
        ])
    }
}

fn check_result_shape(cx: &mut Cx, shape_value: &Value, value: Value) -> Result<()> {
    let shape = shape_value.object().as_shape().ok_or(Error::TypeMismatch {
        expected: "shape",
        found: "non-shape",
    })?;
    let matched = shape.check_value(cx, value)?;
    if matched.accepted {
        Ok(())
    } else {
        Err(Error::WrongShape {
            expected: shape.id().unwrap_or(ShapeId(0)),
            diagnostics: matched.diagnostics,
        })
    }
}
