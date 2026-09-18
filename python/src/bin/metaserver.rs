// Binary wrapper for orbitkv-metaserver
// This delegates to the orbitkv-metaserver crate's run() function

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    orbitkv_metaserver::run().await
}
