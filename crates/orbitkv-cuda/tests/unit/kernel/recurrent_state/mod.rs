use super::*;

#[test]
fn cuda_template_requires_and_replaces_every_parameter() {
    let source = RecurrentKernelSource {
        dynamic_defines: "#define const_b dyn_dims[0]",
        dynamic_parameter: ", const int* dyn_dims",
        total: "(const_b*8)",
        state_index: "const_z",
        decay_index: "(const_z/8)",
        key_index: "(const_z/2)",
        delta_index: "((const_z/4)*2+(const_z%2))",
    }
    .render();
    assert!(!source.contains('@'));
    assert!(source.contains("__global__ void delta_state_update"));
    assert!(source.contains("#define const_b dyn_dims[0]"));
}
