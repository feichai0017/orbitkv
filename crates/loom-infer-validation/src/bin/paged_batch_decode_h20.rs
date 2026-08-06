fn main() -> Result<(), Box<dyn std::error::Error>> {
    loom_infer_validation::gates::paged_batch_decode::run()
}
