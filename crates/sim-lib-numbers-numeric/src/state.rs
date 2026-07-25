//! State admission checks for composed numeric pipelines.

use sim_kernel::{Cx, Error, Result, Value};
use sim_lib_numbers_tensor::{active_tensor_executor, bounded_element_count, tensor_value_ref};

use super::{
    pipeline::{ComposedPipeline, StateKind},
    traits::NumericCallable,
};

pub(crate) fn validate_ode_state(
    cx: &mut Cx,
    pipeline: &ComposedPipeline,
    dy: &NumericCallable,
    x0: &Value,
    y0: &Value,
) -> Result<()> {
    match pipeline.state {
        StateKind::F64 => {
            if tensor_value_ref(y0).is_some() {
                return Err(Error::Eval(
                    "numeric/run-composed f64 state does not accept tensor y0".to_owned(),
                ));
            }
            Ok(())
        }
        StateKind::Tensor => validate_tensor_ode_state(cx, dy, x0, y0),
    }
}

pub(crate) fn ensure_quadrature_state(pipeline: &ComposedPipeline) -> Result<()> {
    if pipeline.state != StateKind::F64 {
        return Err(Error::Eval(
            "numeric/run-composed quadrature supports only f64 state".to_owned(),
        ));
    }
    Ok(())
}

fn validate_tensor_ode_state(
    cx: &mut Cx,
    dy: &NumericCallable,
    x0: &Value,
    y0: &Value,
) -> Result<()> {
    let y0_tensor = tensor_value_ref(y0).ok_or_else(|| {
        Error::Eval("numeric/run-composed tensor state expects tensor y0".to_owned())
    })?;
    bounded_element_count(y0_tensor.shape())?;

    let Some(executor) = active_tensor_executor(cx) else {
        return Ok(());
    };
    let card = executor.card();
    if card.device_capability.is_none() {
        return Ok(());
    }
    if dy.body_cas().is_none() {
        return Err(Error::Eval(
            "numeric/run-composed tensor hardware requires a pure symbolic Func RHS".to_owned(),
        ));
    }

    let sample = dy.call(cx, vec![x0.clone(), y0.clone()])?;
    let sample_tensor = tensor_value_ref(&sample).ok_or_else(|| {
        Error::Eval("numeric/run-composed tensor hardware RHS must return a tensor".to_owned())
    })?;
    if sample_tensor.shape() != y0_tensor.shape() {
        return Err(Error::Eval(format!(
            "numeric/run-composed tensor hardware RHS shape {:?} did not match state {:?}",
            sample_tensor.shape(),
            y0_tensor.shape()
        )));
    }
    if sample_tensor.dtype() != y0_tensor.dtype() {
        return Err(Error::Eval(format!(
            "numeric/run-composed tensor hardware RHS dtype {} did not match state {}",
            sample_tensor.dtype(),
            y0_tensor.dtype()
        )));
    }
    bounded_element_count(sample_tensor.shape())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use sim_kernel::{DefaultFactory, EagerPolicy, Ref, Symbol};

    use super::*;
    use crate::{ComposedPipeline, PipelineKind};

    fn test_cx() -> Cx {
        Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory))
    }

    #[test]
    fn tensor_quadrature_fails_closed() {
        let pipeline = ComposedPipeline::new(
            Ref::Symbol(Symbol::new("test-func")),
            PipelineKind::Quadrature,
            Symbol::new("simpson"),
            StateKind::Tensor,
        );

        let err = ensure_quadrature_state(&pipeline).expect_err("tensor quadrature fails closed");
        assert!(err.to_string().contains("quadrature"), "{err}");
    }

    #[test]
    fn tensor_ode_requires_tensor_initial_state() {
        let mut cx = test_cx();
        let func = cx
            .factory()
            .opaque(Arc::new(sim_lib_numbers_func::Func::native(
                vec![Symbol::new("x"), Symbol::new("y")],
                Arc::new(|_cx, args| Ok(args[1].clone())),
            )))
            .unwrap();
        let dy = NumericCallable::sampled_binary(func, Symbol::new("x"), Symbol::new("y")).unwrap();
        let x0 = cx
            .factory()
            .number_literal(sim_lib_numbers_core::domains::f64(), "0.0".to_owned())
            .unwrap();
        let y0 = cx
            .factory()
            .number_literal(sim_lib_numbers_core::domains::f64(), "1.0".to_owned())
            .unwrap();
        let pipeline = ComposedPipeline::new(
            Ref::Symbol(Symbol::new("test-func")),
            PipelineKind::OdeSolve,
            Symbol::new("rk4"),
            StateKind::Tensor,
        );

        let err = validate_ode_state(&mut cx, &pipeline, &dy, &x0, &y0)
            .expect_err("tensor ODE requires tensor state");
        assert!(err.to_string().contains("tensor y0"), "{err}");
    }
}
