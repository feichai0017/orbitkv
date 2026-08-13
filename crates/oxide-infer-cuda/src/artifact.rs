//! Versioned, fail-closed contracts for offline TileLang artifacts.
//!
//! This module deliberately has no CUDA dependency. Artifact selection and
//! validation happen before bytes are passed to the CUDA driver.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Display, Formatter, Write as _};
use std::path::{Component, Path};
use thiserror::Error;

pub const TILE_ARTIFACT_SCHEMA_VERSION: u32 = 1;
pub const TILE_LAUNCH_ABI_VERSION: u32 = 1;
pub const OXIDE_TILE_PROVIDER: &str = "oxide_tile";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TileArtifactManifest {
    pub schema_version: u32,
    pub provider: String,
    pub contract: ContractIdentity,
    pub algorithm: String,
    pub artifact: ArtifactDescriptor,
    pub toolchain: TileToolchain,
    pub target: CudaTarget,
    pub launch_abi_version: u32,
    pub entry_points: Vec<KernelEntryPoint>,
    pub numerics: NumericalContract,
    pub qualification: QualificationRecords,
    #[serde(default)]
    pub properties: BTreeMap<String, String>,
    #[serde(default)]
    pub dimensions: BTreeMap<String, DimensionConstraint>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContractIdentity {
    pub name: String,
    pub version: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactDescriptor {
    pub file_name: String,
    pub format: ArtifactFormat,
    pub sha256: String,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactFormat {
    Cubin,
    Ptx,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TileToolchain {
    pub tilelang_version: String,
    pub source_revision: String,
    pub cuda_toolkit_version: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CudaTarget {
    pub architecture: String,
    pub minimum_driver: CudaVersion,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CudaVersion {
    pub major: u16,
    pub minor: u16,
    #[serde(default)]
    pub patch: u16,
}

impl Display for CudaVersion {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KernelEntryPoint {
    pub symbol: String,
    pub parameters: Vec<KernelParameter>,
    #[serde(default)]
    pub allowed_aliases: Vec<AllowedAlias>,
    pub launch: LaunchRequirements,
    pub workspace: WorkspaceRequirement,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AllowedAlias {
    pub first: String,
    pub second: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LaunchRequirements {
    pub grid_dimensions: [u32; 3],
    pub block_dimensions: [u32; 3],
    #[serde(default)]
    pub cluster_dimensions: Option<[u32; 3]>,
    #[serde(default)]
    pub dynamic_shared_memory_bytes: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KernelParameter {
    pub name: String,
    pub parameter_type: KernelParameterType,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum KernelParameterType {
    DevicePointer {
        element: String,
        access: AccessMode,
        alignment: u32,
    },
    Scalar {
        scalar: ScalarType,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccessMode {
    Read,
    Write,
    ReadWrite,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarType {
    U32,
    U64,
    I32,
    I64,
    F32,
    F64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkspaceRequirement {
    pub bytes: u64,
    pub alignment: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DimensionConstraint {
    pub min: u64,
    pub max: u64,
    pub multiple_of: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericalContract {
    pub accumulation: String,
    pub output: String,
    pub determinism: Determinism,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Determinism {
    Required,
    NotRequired,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualificationRecords {
    pub correctness: String,
    pub sanitizer: String,
    pub graph: String,
    pub benchmark: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceCompatibility {
    pub architecture: String,
    pub driver: CudaVersion,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactRequest {
    pub contract: String,
    pub contract_version: u32,
    pub algorithm: String,
    pub architecture: String,
    pub properties: BTreeMap<String, String>,
    pub dimensions: BTreeMap<String, u64>,
}

#[derive(Debug, Clone)]
pub struct VerifiedTileArtifact {
    manifest: TileArtifactManifest,
    bytes: Vec<u8>,
}

impl VerifiedTileArtifact {
    #[must_use]
    pub fn manifest(&self) -> &TileArtifactManifest {
        &self.manifest
    }

    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

#[derive(Debug, Default)]
pub struct TileArtifactRegistry {
    artifacts: BTreeMap<ArtifactKey, VerifiedTileArtifact>,
}

impl TileArtifactManifest {
    pub fn from_json(bytes: &[u8]) -> Result<Self, TileArtifactError> {
        let manifest: Self = serde_json::from_slice(bytes)
            .map_err(|error| TileArtifactError::InvalidJson(error.to_string()))?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<(), TileArtifactError> {
        if self.schema_version != TILE_ARTIFACT_SCHEMA_VERSION {
            return Err(TileArtifactError::UnsupportedSchema {
                found: self.schema_version,
                supported: TILE_ARTIFACT_SCHEMA_VERSION,
            });
        }
        if self.provider != OXIDE_TILE_PROVIDER {
            return Err(TileArtifactError::UnsupportedProvider {
                found: self.provider.clone(),
            });
        }
        if self.launch_abi_version != TILE_LAUNCH_ABI_VERSION {
            return Err(TileArtifactError::UnsupportedLaunchAbi {
                found: self.launch_abi_version,
                supported: TILE_LAUNCH_ABI_VERSION,
            });
        }

        validate_identifier("contract.name", &self.contract.name)?;
        validate_identifier("algorithm", &self.algorithm)?;
        validate_identifier("target.architecture", &self.target.architecture)?;
        validate_metadata(
            "toolchain.tilelang_version",
            &self.toolchain.tilelang_version,
        )?;
        validate_metadata("toolchain.source_revision", &self.toolchain.source_revision)?;
        validate_metadata(
            "toolchain.cuda_toolkit_version",
            &self.toolchain.cuda_toolkit_version,
        )?;
        validate_file_name(&self.artifact.file_name, self.artifact.format)?;
        validate_sha256(&self.artifact.sha256)?;
        validate_identifier("numerics.accumulation", &self.numerics.accumulation)?;
        validate_identifier("numerics.output", &self.numerics.output)?;
        validate_metadata("qualification.correctness", &self.qualification.correctness)?;
        validate_metadata("qualification.sanitizer", &self.qualification.sanitizer)?;
        validate_metadata("qualification.graph", &self.qualification.graph)?;
        validate_metadata("qualification.benchmark", &self.qualification.benchmark)?;

        if self.entry_points.is_empty() {
            return Err(TileArtifactError::NoEntryPoints);
        }
        let mut symbols = BTreeSet::new();
        for entry_point in &self.entry_points {
            validate_identifier("entry_points[].symbol", &entry_point.symbol)?;
            if !symbols.insert(entry_point.symbol.as_str()) {
                return Err(TileArtifactError::DuplicateEntryPoint {
                    symbol: entry_point.symbol.clone(),
                });
            }
            validate_launch(&entry_point.symbol, entry_point.launch)?;
            validate_alignment(
                "entry_points[].workspace.alignment",
                entry_point.workspace.alignment,
            )?;

            let mut parameter_names = BTreeSet::new();
            for parameter in &entry_point.parameters {
                validate_identifier("entry_points[].parameters[].name", &parameter.name)?;
                if !parameter_names.insert(parameter.name.as_str()) {
                    return Err(TileArtifactError::DuplicateParameter {
                        symbol: entry_point.symbol.clone(),
                        parameter: parameter.name.clone(),
                    });
                }
                if let KernelParameterType::DevicePointer {
                    element, alignment, ..
                } = &parameter.parameter_type
                {
                    validate_identifier("entry_points[].parameters[].element", element)?;
                    validate_alignment("entry_points[].parameters[].alignment", *alignment)?;
                }
            }

            let pointer_names: BTreeSet<&str> = entry_point
                .parameters
                .iter()
                .filter_map(|parameter| {
                    matches!(
                        parameter.parameter_type,
                        KernelParameterType::DevicePointer { .. }
                    )
                    .then_some(parameter.name.as_str())
                })
                .collect();
            let mut aliases = BTreeSet::new();
            for alias in &entry_point.allowed_aliases {
                let valid = alias.first != alias.second
                    && pointer_names.contains(alias.first.as_str())
                    && pointer_names.contains(alias.second.as_str());
                let pair = if alias.first < alias.second {
                    (alias.first.as_str(), alias.second.as_str())
                } else {
                    (alias.second.as_str(), alias.first.as_str())
                };
                if !valid || !aliases.insert(pair) {
                    return Err(TileArtifactError::InvalidAlias {
                        symbol: entry_point.symbol.clone(),
                        first: alias.first.clone(),
                        second: alias.second.clone(),
                    });
                }
            }
        }

        for (name, value) in &self.properties {
            validate_identifier("properties key", name)?;
            validate_metadata("properties value", value)?;
        }
        for (name, constraint) in &self.dimensions {
            validate_identifier("dimensions key", name)?;
            if constraint.min == 0 || constraint.min > constraint.max || constraint.multiple_of == 0
            {
                return Err(TileArtifactError::InvalidDimensionConstraint {
                    name: name.clone(),
                    min: constraint.min,
                    max: constraint.max,
                    multiple_of: constraint.multiple_of,
                });
            }
        }
        Ok(())
    }

    pub fn verify(
        &self,
        bytes: Vec<u8>,
        device: &DeviceCompatibility,
    ) -> Result<VerifiedTileArtifact, TileArtifactError> {
        self.validate()?;
        if device.architecture != self.target.architecture {
            return Err(TileArtifactError::IncompatibleArchitecture {
                required: self.target.architecture.clone(),
                found: device.architecture.clone(),
            });
        }
        if device.driver < self.target.minimum_driver {
            return Err(TileArtifactError::DriverTooOld {
                required: self.target.minimum_driver,
                found: device.driver,
            });
        }

        let actual_size = bytes.len() as u64;
        if actual_size != self.artifact.size_bytes {
            return Err(TileArtifactError::ArtifactSizeMismatch {
                expected: self.artifact.size_bytes,
                found: actual_size,
            });
        }
        let actual_digest = sha256(&bytes);
        if actual_digest != self.artifact.sha256 {
            return Err(TileArtifactError::ArtifactHashMismatch {
                expected: self.artifact.sha256.clone(),
                found: actual_digest,
            });
        }

        Ok(VerifiedTileArtifact {
            manifest: self.clone(),
            bytes,
        })
    }

    fn supports(&self, request: &ArtifactRequest) -> Result<(), TileArtifactError> {
        for (name, expected) in &self.properties {
            match request.properties.get(name) {
                Some(found) if found == expected => {}
                Some(found) => {
                    return Err(TileArtifactError::PropertyMismatch {
                        name: name.clone(),
                        expected: expected.clone(),
                        found: found.clone(),
                    });
                }
                None => return Err(TileArtifactError::MissingProperty { name: name.clone() }),
            }
        }
        if let Some(name) = request
            .properties
            .keys()
            .find(|name| !self.properties.contains_key(*name))
        {
            return Err(TileArtifactError::UnexpectedProperty { name: name.clone() });
        }

        for (name, constraint) in &self.dimensions {
            let Some(value) = request.dimensions.get(name).copied() else {
                return Err(TileArtifactError::MissingDimension { name: name.clone() });
            };
            if value < constraint.min || value > constraint.max {
                return Err(TileArtifactError::DimensionOutOfRange {
                    name: name.clone(),
                    value,
                    min: constraint.min,
                    max: constraint.max,
                });
            }
            if value % constraint.multiple_of != 0 {
                return Err(TileArtifactError::DimensionNotMultiple {
                    name: name.clone(),
                    value,
                    multiple_of: constraint.multiple_of,
                });
            }
        }
        if let Some(name) = request
            .dimensions
            .keys()
            .find(|name| !self.dimensions.contains_key(*name))
        {
            return Err(TileArtifactError::UnexpectedDimension { name: name.clone() });
        }
        Ok(())
    }
}

impl TileArtifactRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, artifact: VerifiedTileArtifact) -> Result<(), TileArtifactError> {
        let key = ArtifactKey::from_manifest(artifact.manifest());
        if self.artifacts.contains_key(&key) {
            return Err(TileArtifactError::DuplicateArtifact {
                contract: key.contract,
                contract_version: key.contract_version,
                algorithm: key.algorithm,
                architecture: key.architecture,
            });
        }
        self.artifacts.insert(key, artifact);
        Ok(())
    }

    pub fn resolve(
        &self,
        request: &ArtifactRequest,
    ) -> Result<&VerifiedTileArtifact, TileArtifactError> {
        let key = ArtifactKey::from_request(request);
        let artifact =
            self.artifacts
                .get(&key)
                .ok_or_else(|| TileArtifactError::ArtifactNotFound {
                    contract: key.contract.clone(),
                    contract_version: key.contract_version,
                    algorithm: key.algorithm.clone(),
                    architecture: key.architecture.clone(),
                })?;
        artifact.manifest.supports(request)?;
        Ok(artifact)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ArtifactKey {
    contract: String,
    contract_version: u32,
    algorithm: String,
    architecture: String,
}

impl ArtifactKey {
    fn from_manifest(manifest: &TileArtifactManifest) -> Self {
        Self {
            contract: manifest.contract.name.clone(),
            contract_version: manifest.contract.version,
            algorithm: manifest.algorithm.clone(),
            architecture: manifest.target.architecture.clone(),
        }
    }

    fn from_request(request: &ArtifactRequest) -> Self {
        Self {
            contract: request.contract.clone(),
            contract_version: request.contract_version,
            algorithm: request.algorithm.clone(),
            architecture: request.architecture.clone(),
        }
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum TileArtifactError {
    #[error("invalid TileLang artifact manifest JSON: {0}")]
    InvalidJson(String),
    #[error("unsupported artifact schema {found}; runtime supports {supported}")]
    UnsupportedSchema { found: u32, supported: u32 },
    #[error("unsupported artifact provider {found:?}")]
    UnsupportedProvider { found: String },
    #[error("unsupported launch ABI {found}; runtime supports {supported}")]
    UnsupportedLaunchAbi { found: u32, supported: u32 },
    #[error("invalid identifier in {field}: {value:?}")]
    InvalidIdentifier { field: &'static str, value: String },
    #[error("empty or whitespace-only metadata in {field}")]
    InvalidMetadata { field: &'static str },
    #[error("artifact file name must be one local file: {file_name:?}")]
    InvalidFileName { file_name: String },
    #[error("artifact format {format:?} does not match file name {file_name:?}")]
    ArtifactFormatMismatch {
        file_name: String,
        format: ArtifactFormat,
    },
    #[error("artifact sha256 must be 64 lowercase hexadecimal characters")]
    InvalidSha256,
    #[error("manifest must contain at least one kernel entry point")]
    NoEntryPoints,
    #[error("duplicate kernel entry point {symbol:?}")]
    DuplicateEntryPoint { symbol: String },
    #[error("duplicate parameter {parameter:?} in entry point {symbol:?}")]
    DuplicateParameter { symbol: String, parameter: String },
    #[error("invalid or duplicate alias ({first:?}, {second:?}) in entry point {symbol:?}")]
    InvalidAlias {
        symbol: String,
        first: String,
        second: String,
    },
    #[error("invalid {field} for entry point {symbol:?}: {dimensions:?}")]
    InvalidLaunchDimensions {
        symbol: String,
        field: &'static str,
        dimensions: [u32; 3],
    },
    #[error("{field} must be a non-zero power of two, found {value}")]
    InvalidAlignment { field: &'static str, value: u32 },
    #[error("invalid dimension {name:?}: min={min}, max={max}, multiple_of={multiple_of}")]
    InvalidDimensionConstraint {
        name: String,
        min: u64,
        max: u64,
        multiple_of: u64,
    },
    #[error("artifact requires architecture {required}, device is {found}")]
    IncompatibleArchitecture { required: String, found: String },
    #[error("artifact requires CUDA driver {required}, device has {found}")]
    DriverTooOld {
        required: CudaVersion,
        found: CudaVersion,
    },
    #[error("artifact size mismatch: expected {expected} bytes, found {found}")]
    ArtifactSizeMismatch { expected: u64, found: u64 },
    #[error("artifact sha256 mismatch: expected {expected}, found {found}")]
    ArtifactHashMismatch { expected: String, found: String },
    #[error(
        "artifact already registered for {contract} v{contract_version}, {algorithm}, {architecture}"
    )]
    DuplicateArtifact {
        contract: String,
        contract_version: u32,
        algorithm: String,
        architecture: String,
    },
    #[error("no artifact for {contract} v{contract_version}, {algorithm}, {architecture}")]
    ArtifactNotFound {
        contract: String,
        contract_version: u32,
        algorithm: String,
        architecture: String,
    },
    #[error("missing required artifact property {name:?}")]
    MissingProperty { name: String },
    #[error("unexpected artifact property {name:?}")]
    UnexpectedProperty { name: String },
    #[error("property {name:?} must be {expected:?}, found {found:?}")]
    PropertyMismatch {
        name: String,
        expected: String,
        found: String,
    },
    #[error("missing required artifact dimension {name:?}")]
    MissingDimension { name: String },
    #[error("unexpected artifact dimension {name:?}")]
    UnexpectedDimension { name: String },
    #[error("dimension {name:?}={value} is outside [{min}, {max}]")]
    DimensionOutOfRange {
        name: String,
        value: u64,
        min: u64,
        max: u64,
    },
    #[error("dimension {name:?}={value} is not a multiple of {multiple_of}")]
    DimensionNotMultiple {
        name: String,
        value: u64,
        multiple_of: u64,
    },
}

fn validate_identifier(field: &'static str, value: &str) -> Result<(), TileArtifactError> {
    let mut characters = value.chars();
    let valid = characters
        .next()
        .is_some_and(|character| character.is_ascii_alphanumeric())
        && characters.all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '_' | '-' | '.')
        });
    if valid {
        Ok(())
    } else {
        Err(TileArtifactError::InvalidIdentifier {
            field,
            value: value.to_owned(),
        })
    }
}

fn validate_metadata(field: &'static str, value: &str) -> Result<(), TileArtifactError> {
    if value.trim().is_empty() {
        Err(TileArtifactError::InvalidMetadata { field })
    } else {
        Ok(())
    }
}

fn validate_file_name(file_name: &str, format: ArtifactFormat) -> Result<(), TileArtifactError> {
    let path = Path::new(file_name);
    let is_single_file = !file_name.contains(['/', '\\'])
        && matches!(path.components().next(), Some(Component::Normal(_)))
        && path.components().count() == 1;
    if !is_single_file {
        return Err(TileArtifactError::InvalidFileName {
            file_name: file_name.to_owned(),
        });
    }

    let expected_extension = match format {
        ArtifactFormat::Cubin => "cubin",
        ArtifactFormat::Ptx => "ptx",
    };
    if path.extension().and_then(|extension| extension.to_str()) != Some(expected_extension) {
        return Err(TileArtifactError::ArtifactFormatMismatch {
            file_name: file_name.to_owned(),
            format,
        });
    }
    Ok(())
}

fn validate_sha256(digest: &str) -> Result<(), TileArtifactError> {
    if digest.len() == 64
        && digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(TileArtifactError::InvalidSha256)
    }
}

fn validate_alignment(field: &'static str, value: u32) -> Result<(), TileArtifactError> {
    if value.is_power_of_two() {
        Ok(())
    } else {
        Err(TileArtifactError::InvalidAlignment { field, value })
    }
}

fn validate_launch(
    symbol: &str,
    requirements: LaunchRequirements,
) -> Result<(), TileArtifactError> {
    if requirements.grid_dimensions.contains(&0) {
        return Err(TileArtifactError::InvalidLaunchDimensions {
            symbol: symbol.to_owned(),
            field: "grid_dimensions",
            dimensions: requirements.grid_dimensions,
        });
    }
    let block_threads = requirements
        .block_dimensions
        .iter()
        .map(|dimension| u64::from(*dimension))
        .product::<u64>();
    if block_threads == 0 || block_threads > 1024 {
        return Err(TileArtifactError::InvalidLaunchDimensions {
            symbol: symbol.to_owned(),
            field: "block_dimensions",
            dimensions: requirements.block_dimensions,
        });
    }
    if let Some(cluster_dimensions) = requirements.cluster_dimensions
        && cluster_dimensions.contains(&0)
    {
        return Err(TileArtifactError::InvalidLaunchDimensions {
            symbol: symbol.to_owned(),
            field: "cluster_dimensions",
            dimensions: cluster_dimensions,
        });
    }
    Ok(())
}

fn sha256(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(64);
    for byte in Sha256::digest(bytes) {
        write!(&mut output, "{byte:02x}").expect("writing to a String cannot fail");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    const ARTIFACT_BYTES: &[u8] = b"oxide tile artifact";

    fn manifest() -> TileArtifactManifest {
        TileArtifactManifest {
            schema_version: TILE_ARTIFACT_SCHEMA_VERSION,
            provider: OXIDE_TILE_PROVIDER.to_owned(),
            contract: ContractIdentity {
                name: "attention.paged_prefill".to_owned(),
                version: 1,
            },
            algorithm: "gqa4_bf16".to_owned(),
            artifact: ArtifactDescriptor {
                file_name: "paged_prefill_sm90a.cubin".to_owned(),
                format: ArtifactFormat::Cubin,
                sha256: sha256(ARTIFACT_BYTES),
                size_bytes: ARTIFACT_BYTES.len() as u64,
            },
            toolchain: TileToolchain {
                tilelang_version: "0.1.13".to_owned(),
                source_revision: "8001cc4ccf6149382d2019654a19f59c1d4d0482".to_owned(),
                cuda_toolkit_version: "13.1".to_owned(),
            },
            target: CudaTarget {
                architecture: "sm_90a".to_owned(),
                minimum_driver: CudaVersion {
                    major: 590,
                    minor: 48,
                    patch: 1,
                },
            },
            launch_abi_version: TILE_LAUNCH_ABI_VERSION,
            entry_points: vec![KernelEntryPoint {
                symbol: "oxide_paged_prefill".to_owned(),
                parameters: vec![
                    KernelParameter {
                        name: "query".to_owned(),
                        parameter_type: KernelParameterType::DevicePointer {
                            element: "bf16".to_owned(),
                            access: AccessMode::Read,
                            alignment: 16,
                        },
                    },
                    KernelParameter {
                        name: "tokens".to_owned(),
                        parameter_type: KernelParameterType::Scalar {
                            scalar: ScalarType::U32,
                        },
                    },
                ],
                allowed_aliases: Vec::new(),
                launch: LaunchRequirements {
                    grid_dimensions: [96, 16, 1],
                    block_dimensions: [128, 1, 1],
                    cluster_dimensions: None,
                    dynamic_shared_memory_bytes: 49_152,
                },
                workspace: WorkspaceRequirement {
                    bytes: 0,
                    alignment: 256,
                },
            }],
            numerics: NumericalContract {
                accumulation: "f32".to_owned(),
                output: "bf16".to_owned(),
                determinism: Determinism::Required,
            },
            qualification: QualificationRecords {
                correctness: "results/paged_prefill_correctness.json".to_owned(),
                sanitizer: "results/paged_prefill_sanitizer.json".to_owned(),
                graph: "results/paged_prefill_graph.json".to_owned(),
                benchmark: "results/paged_prefill_benchmark.json".to_owned(),
            },
            properties: BTreeMap::from([
                ("dtype".to_owned(), "bf16".to_owned()),
                ("layout".to_owned(), "paged".to_owned()),
                ("mask".to_owned(), "causal".to_owned()),
            ]),
            dimensions: BTreeMap::from([
                (
                    "head_dim".to_owned(),
                    DimensionConstraint {
                        min: 128,
                        max: 128,
                        multiple_of: 8,
                    },
                ),
                (
                    "query_tokens".to_owned(),
                    DimensionConstraint {
                        min: 1,
                        max: 4096,
                        multiple_of: 1,
                    },
                ),
            ]),
        }
    }

    fn device() -> DeviceCompatibility {
        DeviceCompatibility {
            architecture: "sm_90a".to_owned(),
            driver: CudaVersion {
                major: 590,
                minor: 48,
                patch: 2,
            },
        }
    }

    fn request() -> ArtifactRequest {
        ArtifactRequest {
            contract: "attention.paged_prefill".to_owned(),
            contract_version: 1,
            algorithm: "gqa4_bf16".to_owned(),
            architecture: "sm_90a".to_owned(),
            properties: BTreeMap::from([
                ("dtype".to_owned(), "bf16".to_owned()),
                ("layout".to_owned(), "paged".to_owned()),
                ("mask".to_owned(), "causal".to_owned()),
            ]),
            dimensions: BTreeMap::from([
                ("head_dim".to_owned(), 128),
                ("query_tokens".to_owned(), 2048),
            ]),
        }
    }

    #[test]
    fn parses_verifies_registers_and_resolves_exact_artifact() {
        let json = serde_json::to_vec(&manifest()).unwrap();
        let parsed = TileArtifactManifest::from_json(&json).unwrap();
        let verified = parsed.verify(ARTIFACT_BYTES.to_vec(), &device()).unwrap();
        assert_eq!(verified.bytes(), ARTIFACT_BYTES);

        let mut registry = TileArtifactRegistry::new();
        registry.register(verified).unwrap();
        let resolved = registry.resolve(&request()).unwrap();
        assert_eq!(resolved.manifest().algorithm, "gqa4_bf16");
    }

    #[test]
    fn rejects_unknown_json_fields() {
        let mut json = serde_json::to_value(manifest()).unwrap();
        json.as_object_mut()
            .unwrap()
            .insert("fallback".to_owned(), serde_json::Value::Bool(true));
        let error =
            TileArtifactManifest::from_json(&serde_json::to_vec(&json).unwrap()).unwrap_err();
        assert!(matches!(error, TileArtifactError::InvalidJson(_)));
    }

    #[test]
    fn rejects_changed_artifact_bytes_and_size() {
        let mut changed_bytes = ARTIFACT_BYTES.to_vec();
        *changed_bytes
            .last_mut()
            .expect("the artifact fixture must not be empty") ^= 1;
        let hash_error = manifest().verify(changed_bytes, &device()).unwrap_err();
        assert!(matches!(
            hash_error,
            TileArtifactError::ArtifactHashMismatch { .. }
        ));

        let size_error = manifest().verify(b"short".to_vec(), &device()).unwrap_err();
        assert!(matches!(
            size_error,
            TileArtifactError::ArtifactSizeMismatch { .. }
        ));
    }

    #[test]
    fn rejects_incompatible_runtime_and_target() {
        let mut wrong_schema = manifest();
        wrong_schema.schema_version += 1;
        assert!(matches!(
            wrong_schema.validate().unwrap_err(),
            TileArtifactError::UnsupportedSchema { .. }
        ));

        let mut wrong_abi = manifest();
        wrong_abi.launch_abi_version += 1;
        assert!(matches!(
            wrong_abi.validate().unwrap_err(),
            TileArtifactError::UnsupportedLaunchAbi { .. }
        ));

        let mut wrong_provider = manifest();
        wrong_provider.provider = "fallback".to_owned();
        assert!(matches!(
            wrong_provider.validate().unwrap_err(),
            TileArtifactError::UnsupportedProvider { .. }
        ));

        let wrong_architecture = DeviceCompatibility {
            architecture: "sm_100".to_owned(),
            ..device()
        };
        assert!(matches!(
            manifest()
                .verify(ARTIFACT_BYTES.to_vec(), &wrong_architecture)
                .unwrap_err(),
            TileArtifactError::IncompatibleArchitecture { .. }
        ));

        let old_driver = DeviceCompatibility {
            driver: CudaVersion {
                major: 590,
                minor: 47,
                patch: 9,
            },
            ..device()
        };
        assert!(matches!(
            manifest()
                .verify(ARTIFACT_BYTES.to_vec(), &old_driver)
                .unwrap_err(),
            TileArtifactError::DriverTooOld { .. }
        ));
    }

    #[test]
    fn rejects_invalid_names_alignments_and_duplicates() {
        let mut invalid_file = manifest();
        invalid_file.artifact.file_name = "../artifact.cubin".to_owned();
        assert!(matches!(
            invalid_file.validate().unwrap_err(),
            TileArtifactError::InvalidFileName { .. }
        ));

        let mut invalid_alignment = manifest();
        invalid_alignment.entry_points[0].workspace.alignment = 48;
        assert!(matches!(
            invalid_alignment.validate().unwrap_err(),
            TileArtifactError::InvalidAlignment { .. }
        ));

        let mut duplicate_entry = manifest();
        duplicate_entry
            .entry_points
            .push(duplicate_entry.entry_points[0].clone());
        assert!(matches!(
            duplicate_entry.validate().unwrap_err(),
            TileArtifactError::DuplicateEntryPoint { .. }
        ));

        let mut duplicate_parameter = manifest();
        let parameter = duplicate_parameter.entry_points[0].parameters[0].clone();
        duplicate_parameter.entry_points[0]
            .parameters
            .push(parameter);
        assert!(matches!(
            duplicate_parameter.validate().unwrap_err(),
            TileArtifactError::DuplicateParameter { .. }
        ));

        let mut invalid_launch = manifest();
        invalid_launch.entry_points[0].launch.block_dimensions = [1025, 1, 1];
        assert!(matches!(
            invalid_launch.validate().unwrap_err(),
            TileArtifactError::InvalidLaunchDimensions { .. }
        ));

        let mut invalid_alias = manifest();
        invalid_alias.entry_points[0]
            .allowed_aliases
            .push(AllowedAlias {
                first: "query".to_owned(),
                second: "tokens".to_owned(),
            });
        assert!(matches!(
            invalid_alias.validate().unwrap_err(),
            TileArtifactError::InvalidAlias { .. }
        ));
    }

    #[test]
    fn registry_rejects_fallbacks_and_contract_mismatches() {
        let verified = manifest()
            .verify(ARTIFACT_BYTES.to_vec(), &device())
            .unwrap();
        let mut registry = TileArtifactRegistry::new();
        registry.register(verified.clone()).unwrap();
        assert!(matches!(
            registry.register(verified).unwrap_err(),
            TileArtifactError::DuplicateArtifact { .. }
        ));

        let mut wrong_property = request();
        wrong_property
            .properties
            .insert("dtype".to_owned(), "f16".to_owned());
        assert!(matches!(
            registry.resolve(&wrong_property).unwrap_err(),
            TileArtifactError::PropertyMismatch { .. }
        ));

        let mut wrong_multiple = request();
        wrong_multiple.dimensions.insert("head_dim".to_owned(), 127);
        assert!(matches!(
            registry.resolve(&wrong_multiple).unwrap_err(),
            TileArtifactError::DimensionOutOfRange { .. }
        ));

        let mut missing = request();
        missing.contract_version = 2;
        assert!(matches!(
            registry.resolve(&missing).unwrap_err(),
            TileArtifactError::ArtifactNotFound { .. }
        ));
    }
}
