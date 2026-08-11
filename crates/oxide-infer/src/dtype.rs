/// Element type stored by an operator tensor.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum DType {
    /// IEEE-754 single precision.
    F32,
    /// IEEE-754 half precision.
    F16,
    /// Brain floating point with an eight-bit exponent.
    Bf16,
}

impl DType {
    /// Returns the storage size of one element.
    pub const fn size_in_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
        }
    }
}
