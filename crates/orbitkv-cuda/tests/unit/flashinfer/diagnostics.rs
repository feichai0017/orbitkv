use super::*;

#[test]
#[ignore = "debug instrument: dump llama swiglu(+quant) chain egglog"]
fn dump_llama_swiglu_chain_egglog() {
    const I: usize = 8;
    let mut cx = Graph::default();
    let xgu = cx.tensor(('s', 2 * I)).as_dtype(DType::Bf16);
    let scale = cx.tensor(()).as_dtype(DType::F32);
    let gate = xgu.slice((.., ..I));
    let up = xgu.slice((.., I..));
    let h = gate.swish() * up;
    // quant tail (the llama fp8 spelling)
    let hf = h.cast(DType::F32);
    let scale_e = scale.expand_dim(0, 's').expand_dim(1, I);
    let q = (hf / scale_e).cast(DType::F8E4M3);
    let _ = q.cast(DType::F32).output();
    let (program, _root) = orbitkv_compiler::egglog_utils::hlir_to_egglog(&cx);
    println!("{program}");
}
