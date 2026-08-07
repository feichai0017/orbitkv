fn main() -> Result<(), Box<dyn std::error::Error>> {
    loom_infer_validation::benchmarks::paged_prefill_graph::run()
}
