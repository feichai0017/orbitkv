// Binary wrapper for orbitkv-server
// This delegates to the orbitkv-server crate's run() function

fn main() -> Result<(), Box<dyn std::error::Error>> {
    orbitkv_server::run()
}
