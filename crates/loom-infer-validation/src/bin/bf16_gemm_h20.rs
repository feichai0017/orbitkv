fn main() -> Result<(), Box<dyn std::error::Error>> {
    loom_infer_validation::gates::bf16_gemm::run()
}
