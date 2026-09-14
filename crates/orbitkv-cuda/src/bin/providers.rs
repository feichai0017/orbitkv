use std::process::ExitCode;

use orbitkv_cuda::providers::registry::{ProviderId, ProviderOrigin, inspect, provider_lock};

fn run() -> Result<(), String> {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    match arguments.as_slice() {
        [action] if action == "list" => {
            println!(
                "{}",
                serde_json::to_string_pretty(provider_lock()).map_err(|error| error.to_string())?
            );
        }
        [action, name] if action == "inspect" => {
            println!(
                "{}",
                serde_json::to_string_pretty(&inspect(name.parse()?)?)
                    .map_err(|error| error.to_string())?
            );
        }
        [action, name] if action == "fetch" => {
            let providers = if name == "all" {
                ProviderId::ALL
                    .iter()
                    .copied()
                    .filter(|id| matches!(id.origin(), ProviderOrigin::Git(_)))
                    .collect()
            } else {
                vec![name.parse::<ProviderId>()?]
            };
            for provider in providers {
                println!("{provider}: {}", provider.prefetch()?.display());
            }
        }
        _ => {
            return Err("usage: providers list | inspect <provider> | fetch <provider|all>".into());
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}
