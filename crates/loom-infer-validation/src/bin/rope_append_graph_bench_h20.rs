fn main() -> Result<(), Box<dyn std::error::Error>> {
    loom_infer_validation::benchmarks::rope_append_graph::run()
}
