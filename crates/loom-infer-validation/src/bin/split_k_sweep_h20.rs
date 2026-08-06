fn main() -> Result<(), Box<dyn std::error::Error>> {
    loom_infer_validation::benchmarks::split_k::run()
}
