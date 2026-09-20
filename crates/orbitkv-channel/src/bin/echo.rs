use std::io::Write;

use orbitkv_channel::{CommandCode, LocalServer, Response, StatusCode};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let service_name = std::env::args()
        .nth(1)
        .ok_or("usage: orbitkv-channel-echo <service-name> [session-epoch]")?;
    let session_epoch = std::env::args()
        .nth(2)
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(1);
    let server = LocalServer::bind(&service_name)?;
    println!("READY");
    std::io::stdout().flush()?;

    let mut running = true;
    while running {
        if !server.try_serve_for_epoch(session_epoch, |command| {
            running = command.code != CommandCode::Shutdown;
            let mut response = Response::ok(command);
            if command.code == CommandCode::Ping {
                response.value0 = command.arg0.wrapping_add(1);
            } else if command.code != CommandCode::Shutdown {
                response.status = StatusCode::Invalid;
            }
            response
        })? {
            std::thread::yield_now();
        }
    }
    Ok(())
}
