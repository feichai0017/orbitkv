fn main() -> Result<(), Box<dyn std::error::Error>> {
    oxide_infer_lab::gates::bf16_gemm::run()
}
