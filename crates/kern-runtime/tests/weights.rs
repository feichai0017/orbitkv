//! A checkpoint as the runtime sees it: safetensors artifacts, in memory
//! or on disk, parsed into a `Tensors`, tensors found by name with their
//! bytes still where they were.

use kern_manifest::types::DType;
use kern_runtime::{Blob, Safetensors, Tensors};

fn blob(tensors: &[(&str, &str, &[u64], &[u8])]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, dtype, shape, bytes) in tensors {
        let at = data.len();
        data.extend_from_slice(bytes);
        header.insert(
            name.to_string(),
            serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [at, data.len()]}),
        );
    }
    let mut header = serde_json::Value::Object(header).to_string().into_bytes();
    header.resize(header.len().div_ceil(8) * 8, b' ');
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend(header);
    out.extend(data);
    out
}

#[test]
fn tensors_are_found_by_name_across_blobs_with_their_bytes_in_place() {
    let a = blob(&[("q", "BF16", &[2, 2], &[1, 2, 3, 4, 5, 6, 7, 8]), ("scale", "F8_E8M0", &[3], &[9, 10, 11])]);
    let b = blob(&[("k", "F32", &[1], &[0, 0, 128, 63])]);
    let st = Safetensors::parse(&[&a, &b]).unwrap();
    let q = st.find("q").unwrap();
    assert_eq!((q.dtype, q.shape.as_slice()), (DType::Bf16, &[2, 2][..]));
    let Blob::Host(bytes) = q.data else { panic!("safetensors bytes are host bytes") };
    assert_eq!(bytes, &[1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(bytes.as_ptr() as usize - a.as_ptr() as usize, a.len() - 11, "a slice of the blob, not a copy");
    assert_eq!(st.find("scale").unwrap().dtype, DType::Fp8E8m0);
    let k = st.find("k").unwrap();
    assert_eq!((k.dtype, k.data.bytes()), (DType::F32, 4));
    assert_eq!(k.data.slice(2, 2), Some(Blob::Host(&[128, 63][..])));
    assert_eq!(k.data.slice(2, 3), None);
}

#[test]
fn what_the_checkpoint_cannot_answer_names_the_tensor() {
    let a = blob(&[("q", "BF16", &[2], &[1, 2, 3, 4]), ("wide", "F64", &[1], &[0; 8])]);
    let st = Safetensors::parse(&[&a]).unwrap();
    let e = st.find("missing").unwrap_err().to_string();
    assert!(e.contains("tensor `missing` is in none of the 1 artifact(s)"), "{e}");
    let e = st.find("wide").unwrap_err().to_string();
    assert!(e.contains("tensor `wide`: dtype F64 has no manifest dtype"), "{e}");
    let e = Safetensors::parse(&[&a, &a]).err().expect("refused").to_string();
    assert!(e.contains("tensor `q` is in more than one of the 2 artifact(s)"), "{e}");
    let e = Safetensors::parse(&[b"not a checkpoint"]).err().expect("refused").to_string();
    assert!(e.contains("unparseable safetensors"), "{e}");
}

#[test]
fn a_shard_on_disk_hands_out_spans_of_the_file() {
    let a = blob(&[("q", "BF16", &[2, 2], &[1, 2, 3, 4, 5, 6, 7, 8]), ("k", "F32", &[1], &[0, 0, 128, 63])]);
    let dir = std::env::temp_dir().join(format!("kern-weights-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("a.safetensors");
    std::fs::write(&path, &a).unwrap();
    let st = Safetensors::open(&[&path]).unwrap();
    let q = st.find("q").unwrap();
    let Blob::File { file, at, bytes } = q.data else { panic!("a shard's bytes are spans of the file") };
    assert_eq!((q.dtype, q.shape.as_slice(), at, bytes), (DType::Bf16, &[2, 2][..], a.len() as u64 - 12, 8));
    let mut out = [0u8; 8];
    std::os::unix::fs::FileExt::read_exact_at(file, &mut out, at).unwrap();
    assert_eq!(out, [1, 2, 3, 4, 5, 6, 7, 8]);
    let k = st.find("k").unwrap();
    assert_eq!(k.data.slice(2, 2), Some(Blob::File { file, at: a.len() as u64 - 2, bytes: 2 }));
    assert_eq!(k.data.slice(2, 3), None);
    let e = Safetensors::open(&[&path, &path]).err().expect("refused").to_string();
    assert!(e.contains("tensor `k` is in more than one of the 2 artifact(s)"), "{e}");
    let e = Safetensors::open(&[dir.join("missing.safetensors")]).err().expect("refused").to_string();
    assert!(e.contains("missing.safetensors"), "{e}");
    std::fs::remove_dir_all(&dir).unwrap();
}
