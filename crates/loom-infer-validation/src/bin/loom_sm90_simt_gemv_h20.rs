fn main() -> Result<(), Box<dyn std::error::Error>> {
    loom_infer_validation::gates::loom_sm90_simt_gemv::run()
}
