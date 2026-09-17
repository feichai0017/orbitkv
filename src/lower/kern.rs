use std::fmt;

use kern_manifest::Verified;

/// A manifest accepted by the local source-integrated kern verifier.
///
/// Keeping the verified type private prevents later compiler stages from
/// manufacturing an unchecked runtime artifact.
#[derive(Clone, Debug)]
pub struct KernArtifact {
    verified: Verified,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LoweringError(pub String);

impl fmt::Display for LoweringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for LoweringError {}

impl KernArtifact {
    /// Admit emitted JSON only after the pinned `kern-manifest` verifier has
    /// checked schema, references, launch bounds, ABI wiring, and dataflow.
    pub fn from_json(json: &str) -> Result<Self, LoweringError> {
        Verified::from_json(json).map(|verified| Self { verified }).map_err(|errors| LoweringError(errors.to_string()))
    }

    pub fn model(&self) -> &str {
        &self.verified.model
    }

    pub fn to_json(&self) -> String {
        self.verified.to_json()
    }

    pub fn verified(&self) -> &Verified {
        &self.verified
    }
}
