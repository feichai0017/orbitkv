//! Pool errors, grouped by who has to act on them. A refused lease is not
//! an error but a [`Denied`](crate::Denied): the caller asks again later
//! or frees something.

#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The manifest's tables disagree with the states they index: a line
    /// table naming more lines than the state holds, a table over no
    /// pooled state. Provider-side bug the verifier missed.
    #[error("manifest: {0}")]
    Manifest(String),

    /// The caller named a table the manifest lacks, a line past the
    /// table's rows, or a slot count the chunks cannot hold. Caller-side
    /// bug.
    #[error("caller contract: {0}")]
    Api(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// `bail!(Variant, "...")` for the message-carrying variants above.
macro_rules! bail {
    ($variant:ident, $($t:tt)*) => { return Err($crate::Error::$variant(format!($($t)*))) };
}
pub(crate) use bail;
