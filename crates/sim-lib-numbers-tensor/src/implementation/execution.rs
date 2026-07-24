//! Tensor execution contract, default CPU executor, and eval-fabric site.

use std::{fmt, sync::Arc};

use sim_kernel::{
    CapabilityName, ClassRef, Cx, DefaultFactory, Error, Factory, Object, Result, Symbol, Value,
};

use super::{
    cast::cast_tensor,
    value::{Tensor, build_tensor_value, tensor_value_ref},
};

/// Symbol bound in a TensorSite child environment to the active executor.
pub fn tensor_executor_symbol() -> Symbol {
    Symbol::qualified("tensor", "executor")
}

/// Symbol naming the local tensor execution site exported by the tensor lib.
pub fn tensor_site_symbol() -> Symbol {
    Symbol::new("site/tensor")
}

/// Capability required by TensorSite when a request asks for tensor
/// execution authority.
pub fn tensor_execute_capability() -> CapabilityName {
    CapabilityName::new("tensor.execute")
}

/// Open operation symbol for constructing a tensor from shape, dtype, and cells.
pub fn tensor_op_symbol() -> Symbol {
    Symbol::qualified("tensor", "op/tensor")
}

/// Open operation symbol for constructing a scalar tensor.
pub fn scalar_op_symbol() -> Symbol {
    Symbol::qualified("tensor", "op/scalar")
}

/// Open operation symbol for constructing a vector tensor.
pub fn vec_op_symbol() -> Symbol {
    Symbol::qualified("tensor", "op/vec")
}

/// Open operation symbol for constructing a matrix tensor.
pub fn mat_op_symbol() -> Symbol {
    Symbol::qualified("tensor", "op/mat")
}

/// Open operation symbol for indexing a tensor.
pub fn index_op_symbol() -> Symbol {
    Symbol::qualified("tensor", "op/index")
}

/// Open operation symbol for reshaping a tensor.
pub fn reshape_op_symbol() -> Symbol {
    Symbol::qualified("tensor", "op/reshape")
}

/// Open operation symbol for slicing a tensor.
pub fn slice_op_symbol() -> Symbol {
    Symbol::qualified("tensor", "op/slice")
}

/// Open operation symbol for mapping a callable over a tensor.
pub fn map_op_symbol() -> Symbol {
    Symbol::qualified("tensor", "op/map")
}

/// Open operation symbol for explicit tensor casts.
pub fn cast_op_symbol() -> Symbol {
    Symbol::qualified("tensor", "op/cast")
}

/// Tensor shape and dtype expected from an execution request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorMeta {
    shape: Arc<[usize]>,
    dtype: Symbol,
}

impl TensorMeta {
    /// Builds tensor metadata from a shape and scalar dtype.
    pub fn new(shape: Vec<usize>, dtype: Symbol) -> Self {
        Self {
            shape: shape.into(),
            dtype,
        }
    }

    /// Builds tensor metadata from an existing tensor value.
    pub fn from_tensor(tensor: &Tensor) -> Self {
        Self::new(tensor.shape().to_vec(), tensor.dtype().clone())
    }

    /// Returns the tensor shape, outermost axis first.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the scalar dtype every cell must have.
    pub fn dtype(&self) -> &Symbol {
        &self.dtype
    }
}

/// Open operation descriptor carried by a tensor request.
#[derive(Clone, Debug)]
pub struct TensorOp {
    /// Operation symbol, for example [`reshape_op_symbol`].
    pub symbol: Symbol,
    /// Open provider-specific attributes for the operation.
    pub attributes: Value,
}

impl TensorOp {
    /// Builds an operation descriptor with explicit attributes.
    pub fn new(symbol: Symbol, attributes: Value) -> Self {
        Self { symbol, attributes }
    }

    /// Builds an operation descriptor with nil attributes.
    pub fn without_attributes(cx: &mut Cx, symbol: Symbol) -> Result<Self> {
        Ok(Self::new(symbol, cx.factory().nil()?))
    }
}

/// A checked tensor execution request.
#[derive(Clone)]
pub struct TensorRequest {
    /// Operation to run.
    pub operation: TensorOp,
    /// Tensor inputs already validated by the caller.
    pub inputs: Arc<[Tensor]>,
    /// Expected output metadata.
    pub output: TensorMeta,
}

impl TensorRequest {
    /// Builds a tensor execution request.
    pub fn new(operation: TensorOp, inputs: Vec<Tensor>, output: TensorMeta) -> Self {
        Self {
            operation,
            inputs: inputs.into(),
            output,
        }
    }
}

/// Result of submitting a tensor request to an executor.
#[derive(Clone)]
pub enum TensorExecution {
    /// The request finished and produced a tensor.
    Complete(Tensor),
    /// The executor declined before taking ownership of the request.
    Unsupported {
        /// Reason the executor declined the request.
        reason: Arc<str>,
    },
}

/// Description of one tensor executor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorExecutorCard {
    /// Stable executor symbol.
    pub symbol: Symbol,
    /// Human-readable provider label.
    pub provider: String,
    /// Placement locality this executor uses.
    pub locality: Symbol,
    /// Operation symbols the executor accepts.
    pub operations: Arc<[Symbol]>,
    /// Physical-device capability required by this executor, if any.
    pub device_capability: Option<CapabilityName>,
}

impl TensorExecutorCard {
    /// Builds an executor card.
    pub fn new(
        symbol: Symbol,
        provider: impl Into<String>,
        locality: Symbol,
        operations: Vec<Symbol>,
        device_capability: Option<CapabilityName>,
    ) -> Self {
        Self {
            symbol,
            provider: provider.into(),
            locality,
            operations: operations.into(),
            device_capability,
        }
    }
}

/// Evidence returned after an executor has flushed accepted submissions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubmissionEvidence {
    /// Executor that produced the evidence.
    pub executor: Symbol,
    /// Number of accepted submissions represented by this flush.
    pub accepted: usize,
}

impl SubmissionEvidence {
    /// Builds flush evidence for an executor.
    pub fn new(executor: Symbol, accepted: usize) -> Self {
        Self { executor, accepted }
    }
}

/// Error reported by tensor execution contracts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TensorExecError {
    /// A required capability was absent.
    CapabilityDenied {
        /// The denied capability.
        capability: CapabilityName,
    },
    /// The request is not valid for the executor contract.
    InvalidRequest {
        /// Explanation of the invalid request.
        message: Arc<str>,
    },
    /// The executor does not support the requested operation.
    Unsupported {
        /// Operation that was declined.
        operation: Symbol,
        /// Explanation of the unsupported path.
        reason: Arc<str>,
    },
    /// A result did not match the requested tensor metadata.
    Shape {
        /// Explanation of the shape or dtype mismatch.
        message: Arc<str>,
    },
    /// Evaluation failed while realizing a tensor expression.
    Eval {
        /// Explanation of the evaluation failure.
        message: Arc<str>,
    },
}

impl TensorExecError {
    fn invalid(message: impl Into<Arc<str>>) -> Self {
        Self::InvalidRequest {
            message: message.into(),
        }
    }

    fn shape(message: impl Into<Arc<str>>) -> Self {
        Self::Shape {
            message: message.into(),
        }
    }

    fn unsupported(operation: Symbol, reason: impl Into<Arc<str>>) -> Self {
        Self::Unsupported {
            operation,
            reason: reason.into(),
        }
    }
}

impl fmt::Display for TensorExecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CapabilityDenied { capability } => {
                write!(f, "capability denied: {capability}")
            }
            Self::InvalidRequest { message } => f.write_str(message),
            Self::Unsupported { operation, reason } => {
                write!(f, "unsupported tensor operation {operation}: {reason}")
            }
            Self::Shape { message } => f.write_str(message),
            Self::Eval { message } => f.write_str(message),
        }
    }
}

impl std::error::Error for TensorExecError {}

impl From<Error> for TensorExecError {
    fn from(error: Error) -> Self {
        match error {
            Error::CapabilityDenied { capability } => Self::CapabilityDenied { capability },
            Error::WrongShape { diagnostics, .. } => {
                let message = diagnostics
                    .first()
                    .map(|diagnostic| diagnostic.message.clone())
                    .unwrap_or_else(|| "tensor result shape check failed".to_owned());
                Self::Shape {
                    message: Arc::from(message),
                }
            }
            other => Self::Eval {
                message: Arc::from(other.to_string()),
            },
        }
    }
}

impl From<TensorExecError> for Error {
    fn from(error: TensorExecError) -> Self {
        match error {
            TensorExecError::CapabilityDenied { capability } => {
                Error::CapabilityDenied { capability }
            }
            other => Error::Eval(other.to_string()),
        }
    }
}

/// A loadable provider that executes checked tensor requests.
pub trait TensorExecutor: Send + Sync + 'static {
    /// Returns the executor descriptor.
    fn card(&self) -> TensorExecutorCard;

    /// Executes one checked tensor request.
    fn execute(
        &self,
        cx: &mut Cx,
        request: TensorRequest,
    ) -> std::result::Result<TensorExecution, TensorExecError>;

    /// Flushes accepted submissions and returns synchronization evidence.
    fn flush(&self) -> std::result::Result<SubmissionEvidence, TensorExecError>;
}

/// Default host executor that delegates to the registered tensor functions.
#[derive(Clone, Debug, Default)]
pub struct CpuTensorExecutor;

impl CpuTensorExecutor {
    /// Builds the default CPU executor.
    pub fn new() -> Self {
        Self
    }
}

impl TensorExecutor for CpuTensorExecutor {
    fn card(&self) -> TensorExecutorCard {
        TensorExecutorCard::new(
            Symbol::qualified("tensor", "executor/cpu"),
            "cpu",
            Symbol::qualified("core", "local-fabric"),
            vec![
                tensor_op_symbol(),
                scalar_op_symbol(),
                vec_op_symbol(),
                mat_op_symbol(),
                reshape_op_symbol(),
                cast_op_symbol(),
            ],
            None,
        )
    }

    fn execute(
        &self,
        cx: &mut Cx,
        request: TensorRequest,
    ) -> std::result::Result<TensorExecution, TensorExecError> {
        let operation = request.operation.symbol.clone();
        let result = if operation == tensor_op_symbol() || operation == vec_op_symbol() {
            execute_tensor(cx, &request)?
        } else if operation == scalar_op_symbol() {
            execute_scalar(&request)?
        } else if operation == mat_op_symbol() {
            execute_mat(cx, &request)?
        } else if operation == reshape_op_symbol() {
            execute_reshape(cx, &request)?
        } else if operation == cast_op_symbol() {
            execute_cast(&request)?
        } else if operation == index_op_symbol() {
            return Err(TensorExecError::unsupported(
                operation,
                "index returns a scalar value, not a tensor",
            ));
        } else {
            return Ok(TensorExecution::Unsupported {
                reason: Arc::from("unknown tensor operation"),
            });
        };
        check_output(&request.output, &result)?;
        Ok(TensorExecution::Complete(result))
    }

    fn flush(&self) -> std::result::Result<SubmissionEvidence, TensorExecError> {
        Ok(SubmissionEvidence::new(
            Symbol::qualified("tensor", "executor/cpu"),
            0,
        ))
    }
}

impl Object for CpuTensorExecutor {
    fn display(&self, _cx: &mut Cx) -> Result<String> {
        Ok("#<tensor-executor cpu>".to_owned())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl sim_kernel::ObjectCompat for CpuTensorExecutor {
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
}

fn execute_tensor(
    cx: &mut Cx,
    request: &TensorRequest,
) -> std::result::Result<Tensor, TensorExecError> {
    let cells = request
        .inputs
        .iter()
        .map(|tensor| {
            if tensor.rank() == 0 {
                tensor.cell(0)
            } else {
                Err(Error::Eval(
                    "tensor op/tensor expects scalar tensor inputs as cells".to_owned(),
                ))
            }
        })
        .collect::<Result<Vec<_>>>()
        .map_err(TensorExecError::from)?;
    build_tensor_value(
        cx,
        request.output.shape().to_vec(),
        Some(request.output.dtype().clone()),
        cells,
    )
    .map_err(TensorExecError::from)
    .and_then(|value| tensor_from_value(&value))
}

fn execute_scalar(request: &TensorRequest) -> std::result::Result<Tensor, TensorExecError> {
    let [tensor] = request.inputs.as_ref() else {
        return Err(TensorExecError::invalid(
            "scalar operation expects exactly one tensor input",
        ));
    };
    if tensor.rank() != 0 {
        return Err(TensorExecError::invalid(
            "scalar operation expects a rank-0 tensor input",
        ));
    }
    Ok(tensor.clone())
}

fn execute_mat(
    cx: &mut Cx,
    request: &TensorRequest,
) -> std::result::Result<Tensor, TensorExecError> {
    if request.output.shape().len() != 2 {
        return Err(TensorExecError::invalid(
            "matrix operation expects rank-2 output metadata",
        ));
    }
    let row_width = request.output.shape()[1];
    let mut cells = Vec::new();
    for row in request.inputs.iter() {
        if row.shape() != [row_width] {
            return Err(TensorExecError::invalid(
                "matrix operation inputs must be rank-1 rows matching output width",
            ));
        }
        cells.extend(row.cells().map_err(TensorExecError::from)?.iter().cloned());
    }
    build_tensor_value(
        cx,
        request.output.shape().to_vec(),
        Some(request.output.dtype().clone()),
        cells,
    )
    .map_err(TensorExecError::from)
    .and_then(|value| tensor_from_value(&value))
}

fn execute_reshape(
    cx: &mut Cx,
    request: &TensorRequest,
) -> std::result::Result<Tensor, TensorExecError> {
    let [tensor] = request.inputs.as_ref() else {
        return Err(TensorExecError::invalid(
            "reshape operation expects exactly one tensor input",
        ));
    };
    build_tensor_value(
        cx,
        request.output.shape().to_vec(),
        Some(request.output.dtype().clone()),
        tensor
            .cells()
            .map_err(TensorExecError::from)?
            .iter()
            .cloned()
            .collect(),
    )
    .map_err(TensorExecError::from)
    .and_then(|value| tensor_from_value(&value))
}

fn execute_cast(request: &TensorRequest) -> std::result::Result<Tensor, TensorExecError> {
    let [tensor] = request.inputs.as_ref() else {
        return Err(TensorExecError::invalid(
            "cast operation expects exactly one tensor input",
        ));
    };
    cast_tensor(tensor, request.output.dtype().clone()).map_err(TensorExecError::from)
}

fn tensor_from_value(value: &Value) -> std::result::Result<Tensor, TensorExecError> {
    tensor_value_ref(value)
        .cloned()
        .ok_or_else(|| TensorExecError::invalid("tensor executor produced a non-tensor value"))
}

fn check_output(
    expected: &TensorMeta,
    result: &Tensor,
) -> std::result::Result<(), TensorExecError> {
    if expected.shape() != result.shape() {
        return Err(TensorExecError::shape(format!(
            "tensor result shape {:?} did not match {:?}",
            result.shape(),
            expected.shape()
        )));
    }
    if expected.dtype() != result.dtype() {
        return Err(TensorExecError::shape(format!(
            "tensor result dtype {} did not match {}",
            result.dtype(),
            expected.dtype()
        )));
    }
    Ok(())
}
