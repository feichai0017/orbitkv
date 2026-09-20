// Binary wrapper for OrbitKV Cache Manager
// This delegates to the orbitkv-server crate's run() function

fn main() -> Result<(), Box<dyn std::error::Error>> {
    orbitkv_server::run()
}
