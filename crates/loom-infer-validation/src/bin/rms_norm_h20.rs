fn main() -> Result<(), Box<dyn std::error::Error>> {
    loom_infer_validation::gates::rms_norm::run()
}
