//! Inspectable resource and provenance records for tensor transforms.

use sim_kernel::{ContentId, Datum, Symbol};

use crate::{
    Direction, Normalization, PaddingPolicy, PlacementPolicy, SignConvention, SignalError,
    SpectrumPacking, TransformKind, TransformPlan, tensor_view::transform_cell_count,
};

/// Cell representation used by a multidimensional transform.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransformPrecision {
    /// IEEE-754 binary64 real cells.
    F64,
    /// Pairs of IEEE-754 binary64 real and imaginary components.
    ComplexF64,
}

impl TransformPrecision {
    pub(crate) const fn cell_bytes(self) -> usize {
        match self {
            Self::F64 => size_of::<f64>(),
            Self::ComplexF64 => 2 * size_of::<f64>(),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::F64 => "f64",
            Self::ComplexF64 => "complex-f64",
        }
    }
}

/// Caller-declared memory budget and external block granularity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransformResources {
    /// Hard ceiling for peak transform scratch storage.
    pub max_scratch_bytes: usize,
    /// Maximum number of tensor cells in one external block.
    pub block_len: usize,
}

impl TransformResources {
    /// Validates that both limits are nonzero.
    pub fn validate(self) -> Result<(), SignalError> {
        if self.max_scratch_bytes == 0 {
            return Err(SignalError::InvalidPolicy {
                policy: "max_scratch_bytes",
                reason: "scratch limit must be nonzero",
            });
        }
        if self.block_len == 0 {
            return Err(SignalError::InvalidPolicy {
                policy: "block_len",
                reason: "external block length must be nonzero",
            });
        }
        Ok(())
    }
}

/// Auditable execution facts for an in-memory or blocked transform.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransformReport {
    /// Peak temporary storage required by the plan, excluding result storage.
    pub scratch_bytes: usize,
    /// Number of complete separable axis passes.
    pub passes: usize,
    /// Total Table/Dir blocks read or written.
    pub io_blocks: usize,
    /// Cell representation used for execution.
    pub precision: TransformPrecision,
    /// Content digest of shape, axes, conventions, and optional resource plan.
    pub plan_digest: ContentId,
}

/// Computes the stable content digest for a multidimensional transform plan.
pub fn transform_plan_digest(
    shape: &[usize],
    axes: &[usize],
    plan: &TransformPlan,
    precision: TransformPrecision,
    resources: Option<TransformResources>,
) -> ContentId {
    let resources = match resources {
        Some(resources) => Datum::Node {
            tag: Symbol::qualified("numbers-signal", "resources-v1"),
            fields: vec![
                datum_field(
                    "max-scratch-bytes",
                    Datum::String(resources.max_scratch_bytes.to_string()),
                ),
                datum_field("block-len", Datum::String(resources.block_len.to_string())),
            ],
        },
        None => Datum::Nil,
    };
    Datum::Node {
        tag: Symbol::qualified("numbers-signal", "transform-plan-v1"),
        fields: vec![
            datum_field("shape", usize_vector(shape)),
            datum_field("axes", usize_vector(axes)),
            datum_field("kind", Datum::String(kind_name(plan.kind))),
            datum_field("template-len", Datum::String(plan.len.to_string())),
            datum_field(
                "direction",
                Datum::String(direction_name(plan.direction).into()),
            ),
            datum_field(
                "normalization",
                Datum::String(normalization_name(plan.normalization).into()),
            ),
            datum_field("sign", Datum::String(sign_name(plan.sign).into())),
            datum_field("packing", Datum::String(packing_name(plan.packing).into())),
            datum_field("length", Datum::String(format!("{:?}", plan.length))),
            datum_field("padding", Datum::String(padding_name(plan.padding).into())),
            datum_field(
                "placement",
                Datum::String(placement_name(plan.placement).into()),
            ),
            datum_field("precision", Datum::String(precision.name().into())),
            datum_field("resources", resources),
        ],
    }
    .content_id()
    .expect("transform plan datum has unique named fields")
}

pub(crate) fn scratch_bytes(
    shape: &[usize],
    axes: &[usize],
    precision: TransformPrecision,
    block_len: usize,
) -> Result<usize, SignalError> {
    let _ = transform_cell_count(shape)?;
    let max_line =
        axes.iter()
            .map(|&axis| shape[axis])
            .max()
            .ok_or(SignalError::InvalidPolicy {
                policy: "axes",
                reason: "at least one transform axis is required",
            })?;
    // The executor holds one caller line plus the 1-D transform's selected
    // input, output, mixed-radix children, or two Bluestein convolution
    // buffers. Sixteen cell widths is a conservative peak for every path.
    let line = max_line
        .checked_mul(precision.cell_bytes())
        .and_then(|bytes| bytes.checked_mul(16))
        .ok_or(SignalError::InvalidTensorView {
            reason: "transform scratch size overflowed",
        })?;
    let block =
        block_len
            .checked_mul(precision.cell_bytes())
            .ok_or(SignalError::InvalidTensorView {
                reason: "block scratch size overflowed",
            })?;
    line.checked_add(block)
        .ok_or(SignalError::InvalidTensorView {
            reason: "transform scratch size overflowed",
        })
}

fn datum_field(name: &str, value: Datum) -> (Symbol, Datum) {
    (Symbol::new(name), value)
}

fn usize_vector(values: &[usize]) -> Datum {
    Datum::Vector(
        values
            .iter()
            .map(|value| Datum::String(value.to_string()))
            .collect(),
    )
}

fn kind_name(kind: TransformKind) -> String {
    match kind {
        TransformKind::Dft => "dft".into(),
        TransformKind::Fft => "fft".into(),
        TransformKind::RealFft => "real-fft".into(),
        TransformKind::Dct(kind) => format!("dct-{kind:?}"),
        TransformKind::Dst(kind) => format!("dst-{kind:?}"),
    }
}

fn direction_name(direction: Direction) -> &'static str {
    match direction {
        Direction::Forward => "forward",
        Direction::Inverse => "inverse",
    }
}

fn normalization_name(normalization: Normalization) -> &'static str {
    match normalization {
        Normalization::None => "none",
        Normalization::Forward => "forward",
        Normalization::Inverse => "inverse",
        Normalization::Orthonormal => "orthonormal",
    }
}

fn sign_name(sign: SignConvention) -> &'static str {
    match sign {
        SignConvention::NegativeForward => "negative-forward",
        SignConvention::PositiveForward => "positive-forward",
    }
}

fn packing_name(packing: SpectrumPacking) -> &'static str {
    match packing {
        SpectrumPacking::Full => "full",
        SpectrumPacking::HermitianHalf => "hermitian-half",
    }
}

fn padding_name(padding: PaddingPolicy) -> &'static str {
    match padding {
        PaddingPolicy::Reject => "reject",
        PaddingPolicy::Zero => "zero",
    }
}

fn placement_name(placement: PlacementPolicy) -> &'static str {
    match placement {
        PlacementPolicy::OutOfPlace => "out-of-place",
        PlacementPolicy::InPlace => "in-place",
    }
}
