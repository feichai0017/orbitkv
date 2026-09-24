use std::{collections::HashMap, sync::Arc};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use orbitkv_state::{AttentionRole, Scalar16, StorageFormat};

use super::{EncodedSegment, ans, turboquant_bytes};

pub(crate) struct GpuCodec {
    fp8_encode: CudaFunction,
    fp8_decode: CudaFunction,
    turbo_encode: CudaFunction,
    turbo_decode: CudaFunction,
    codebooks: HashMap<(u32, u8), CudaSlice<f32>>,
    ans: Option<ans::Library>,
}

impl GpuCodec {
    pub(crate) fn new(ctx: &Arc<CudaContext>) -> Result<Self, String> {
        use cudarc::driver::sys::CUdevice_attribute::{
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        };
        let major = ctx
            .attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(|e| e.to_string())?;
        let minor = ctx
            .attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .map_err(|e| e.to_string())?;
        let arch = if major >= 9 {
            Some("compute_90")
        } else if major == 8 && minor >= 9 {
            Some("compute_89")
        } else {
            None
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            include_str!("kernels.cu"),
            cudarc::nvrtc::CompileOptions {
                arch,
                ..Default::default()
            },
        )
        .map_err(|e| format!("codec NVRTC: {e:?}"))?;
        let module = ctx.load_module(ptx).map_err(|e| e.to_string())?;
        Ok(Self {
            fp8_encode: module
                .load_function("fp8_encode")
                .map_err(|e| e.to_string())?,
            fp8_decode: module
                .load_function("fp8_decode")
                .map_err(|e| e.to_string())?,
            turbo_encode: module
                .load_function("turbo_encode")
                .map_err(|e| e.to_string())?,
            turbo_decode: module
                .load_function("turbo_decode")
                .map_err(|e| e.to_string())?,
            codebooks: HashMap::new(),
            ans: None,
        })
    }

    fn ans(&mut self) -> Result<&ans::Library, String> {
        if self.ans.is_none() {
            self.ans = Some(ans::Library::load()?);
        }
        Ok(self.ans.as_ref().expect("loaded"))
    }

    fn codebook(
        &mut self,
        stream: &Arc<CudaStream>,
        dim: u32,
        bits: u8,
    ) -> Result<&CudaSlice<f32>, String> {
        if let std::collections::hash_map::Entry::Vacant(entry) = self.codebooks.entry((dim, bits))
        {
            entry.insert(
                stream
                    .clone_htod(&centroids(dim, bits))
                    .map_err(|e| e.to_string())?,
            );
        }
        Ok(&self.codebooks[&(dim, bits)])
    }

    /// Input is the engine's live GPU page; only encoded bytes subsequently cross PCIe.
    pub(crate) fn encode(
        &mut self,
        stream: &Arc<CudaStream>,
        source: u64,
        bytes: usize,
        format: StorageFormat,
        budget: usize,
    ) -> Result<Option<(CudaSlice<u8>, usize)>, String> {
        if bytes == 0 || bytes > 16 * 1024 * 1024 {
            return Ok(None);
        }
        if let Some(data_type) = ans_type(format) {
            return self.ans()?.encode(stream, source, bytes, budget, data_type);
        }
        let stored = match format {
            StorageFormat::Fp8FromBf16 | StorageFormat::Fp8FromFp16 if bytes.is_multiple_of(2) => {
                bytes / 2
            }
            StorageFormat::TurboQuant {
                head_dim,
                bits,
                role,
                ..
            } => match turboquant_bytes(bytes, head_dim, bits, role) {
                Some(size) => size,
                None => return Ok(None),
            },
            _ => return Ok(None),
        };
        // Includes status and all cached codebooks (at most 8 geometries, 512 bytes).
        if stored.saturating_add(1024) > budget {
            return Ok(None);
        }
        let output = stream
            .alloc_zeros::<u8>(stored)
            .map_err(|e| e.to_string())?;
        let invalid = stream.alloc_zeros::<u32>(1).map_err(|e| e.to_string())?;
        match format {
            StorageFormat::Fp8FromBf16 | StorageFormat::Fp8FromFp16 => {
                let n = (bytes / 2) as u64;
                let bf = i32::from(format == StorageFormat::Fp8FromBf16);
                let mut launch = stream.launch_builder(&self.fp8_encode);
                launch
                    .arg(&source)
                    .arg(&output)
                    .arg(&n)
                    .arg(&bf)
                    .arg(&invalid);
                // SAFETY: source is a leased engine range; output and status are owned through sync.
                unsafe { launch.launch(LaunchConfig::for_num_elems(n.min(65535 * 256) as u32)) }
                    .map_err(|e| e.to_string())?;
            }
            StorageFormat::TurboQuant {
                scalar,
                role,
                head_dim,
                seed,
                bits,
            } => {
                let func = self.turbo_encode.clone();
                let codebook = self.codebook(stream, head_dim, bits)?;
                let vectors = (bytes / (head_dim as usize * 2)) as i32;
                let dim = head_dim as i32;
                let nbits = bits as i32;
                let bf = i32::from(scalar == Scalar16::Bf16);
                let key = match role {
                    AttentionRole::PackedKeyValue => 2,
                    _ => i32::from(role == AttentionRole::Key),
                };
                let mut launch = stream.launch_builder(&func);
                launch
                    .arg(&source)
                    .arg(&output)
                    .arg(&vectors)
                    .arg(&dim)
                    .arg(&nbits)
                    .arg(&bf)
                    .arg(&key)
                    .arg(&seed)
                    .arg(codebook)
                    .arg(&invalid);
                // SAFETY: exact contiguous vectors validated before registration; one cooperative block/vector.
                unsafe {
                    launch.launch(LaunchConfig {
                        grid_dim: ((vectors as u32).min(65535), 1, 1),
                        block_dim: (head_dim, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(|e| e.to_string())?;
            }
            _ => unreachable!("format checked"),
        }
        stream.synchronize().map_err(|e| e.to_string())?;
        if stream.clone_dtoh(&invalid).map_err(|e| e.to_string())?[0] != 0 {
            return Ok(None);
        }
        Ok(Some((output, stored)))
    }

    /// The caller validates the host checksum before upload and keeps input/target owned.
    pub(crate) fn decode(
        &mut self,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<u8>,
        target: u64,
        meta: &EncodedSegment,
        budget: usize,
    ) -> Result<(), String> {
        if ans_type(meta.format).is_some() {
            return self.ans()?.decode(
                stream,
                input,
                meta.stored_bytes,
                target,
                meta.logical_bytes,
                budget,
            );
        }
        match meta.format {
            StorageFormat::Fp8FromBf16 | StorageFormat::Fp8FromFp16 => {
                let n = (meta.logical_bytes / 2) as u64;
                let bf = i32::from(meta.format == StorageFormat::Fp8FromBf16);
                let mut launch = stream.launch_builder(&self.fp8_decode);
                launch.arg(input).arg(&target).arg(&n).arg(&bf);
                // SAFETY: validated input size is n, leased destination holds 2*n bytes.
                unsafe { launch.launch(LaunchConfig::for_num_elems(n.min(65535 * 256) as u32)) }
                    .map_err(|e| e.to_string())?;
            }
            StorageFormat::TurboQuant {
                scalar,
                role,
                head_dim,
                seed,
                bits,
            } => {
                let func = self.turbo_decode.clone();
                let codebook = self.codebook(stream, head_dim, bits)?;
                let vectors = (meta.logical_bytes / (head_dim as usize * 2)) as i32;
                let dim = head_dim as i32;
                let nbits = bits as i32;
                let bf = i32::from(scalar == Scalar16::Bf16);
                let key = match role {
                    AttentionRole::PackedKeyValue => 2,
                    _ => i32::from(role == AttentionRole::Key),
                };
                let mut launch = stream.launch_builder(&func);
                launch
                    .arg(input)
                    .arg(&target)
                    .arg(&vectors)
                    .arg(&dim)
                    .arg(&nbits)
                    .arg(&bf)
                    .arg(&key)
                    .arg(&seed)
                    .arg(codebook);
                // SAFETY: version, vector stride and destination capacity checked before launch.
                unsafe {
                    launch.launch(LaunchConfig {
                        grid_dim: ((vectors as u32).min(65535), 1, 1),
                        block_dim: (head_dim, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(|e| e.to_string())?;
            }
            _ => return Err("unexpected GPU decode format".into()),
        }
        stream.synchronize().map_err(|e| e.to_string())
    }
}

/// Lloyd-Max Gaussian centroids, computed once per supported geometry.
/// Integration and stopping criterion match LMCache's TurboQuant MSE codebook.
fn centroids(dim: u32, bits: u8) -> Vec<f32> {
    let sigma = 1.0 / (dim as f64).sqrt();
    let limit = 3.5 * sigma;
    let count = 1usize << bits;
    let mut values: Vec<f64> = (0..count)
        .map(|i| -limit + (i as f64 + 0.5) * 2.0 * limit / count as f64)
        .collect();
    for _ in 0..200 {
        let mut edges = vec![-3.0 * limit];
        edges.extend(values.windows(2).map(|w| (w[0] + w[1]) * 0.5));
        edges.push(3.0 * limit);
        let next: Vec<f64> = edges
            .windows(2)
            .map(|w| {
                let mut mass = 0.0;
                let mut moment = 0.0;
                for i in 0..=200 {
                    let x = w[0] + (w[1] - w[0]) * i as f64 / 200.0;
                    let weight = (-0.5 * (x / sigma).powi(2)).exp()
                        * if i == 0 || i == 200 { 0.5 } else { 1.0 };
                    mass += weight;
                    moment += x * weight;
                }
                moment / mass
            })
            .collect();
        let delta = values
            .iter()
            .zip(&next)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        values = next;
        if delta < 1e-10 {
            break;
        }
    }
    values.into_iter().map(|x| x as f32).collect()
}

fn ans_type(format: StorageFormat) -> Option<i32> {
    match format {
        StorageFormat::Ans => Some(1),
        StorageFormat::Ans16 => Some(9),
        StorageFormat::AnsFp8 => Some(10),
        _ => None,
    }
}
