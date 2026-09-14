use orbitkv_compiler::dtype::DType;

/// Logical visibility, independent of a kernel library's mask encoding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttentionMask {
    Causal,
    /// Number of previous KV positions visible, in addition to the current one.
    Sliding {
        window_left: usize,
    },
    Unmasked,
}

impl AttentionMask {
    pub(super) fn to_egglog(self) -> String {
        match self {
            Self::Causal => "(CausalAttention)".to_owned(),
            Self::Sliding { window_left } => format!("(SlidingAttention {window_left})"),
            Self::Unmasked => "(UnmaskedAttention)".to_owned(),
        }
    }
}

/// Scaled dot-product attention. Recurrent/linear attention has separate semantics.
/// Query/key and value dimensions are independent; an implementation may support
/// only equal dimensions. Storage encoding and request segmentation belong to views.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttentionSpec {
    pub query_heads: usize,
    pub kv_heads: usize,
    pub query_key_dim: usize,
    pub value_dim: usize,
    pub dtype: DType,
    /// Explicit positive finite scale, resolved by the model frontend.
    pub scale: f64,
    pub mask: AttentionMask,
}

impl AttentionSpec {
    pub(super) fn validate(self) -> Result<(), AttentionError> {
        if self.query_heads == 0
            || self.kv_heads == 0
            || !self.query_heads.is_multiple_of(self.kv_heads)
            || self.query_key_dim == 0
            || self.value_dim == 0
        {
            return Err(AttentionError::Geometry("head dimensions"));
        }
        if !self.scale.is_finite() || self.scale <= 0.0 {
            return Err(AttentionError::Geometry("softmax scale"));
        }
        if let AttentionMask::Sliding { window_left } = self.mask
            && i64::try_from(window_left).is_err()
        {
            return Err(AttentionError::Geometry(
                "window exceeds semantic index range",
            ));
        }
        if !matches!(
            self.dtype,
            DType::F16 | DType::Bf16 | DType::F32 | DType::F64
        ) {
            return Err(AttentionError::DType("attention values"));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttentionError {
    GraphOwnership,
    Geometry(&'static str),
    Shape(&'static str),
    DType(&'static str),
    Layout(&'static str),
}

impl std::fmt::Display for AttentionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GraphOwnership => {
                formatter.write_str("attention inputs must belong to one graph")
            }
            Self::Geometry(field) => write!(formatter, "invalid attention geometry: {field}"),
            Self::Shape(field) => write!(formatter, "invalid attention shape: {field}"),
            Self::DType(field) => write!(formatter, "invalid attention dtype: {field}"),
            Self::Layout(field) => write!(formatter, "invalid attention storage layout: {field}"),
        }
    }
}

impl std::error::Error for AttentionError {}
