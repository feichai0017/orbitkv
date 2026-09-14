use std::process::ExitCode;

fn main() -> ExitCode {
    let Some(provider) = std::env::args().nth(1) else {
        eprintln!("usage: fetch-provider <deepgemm|flashinfer|flashattention>");
        return ExitCode::FAILURE;
    };
    let result = match provider.as_str() {
        "deepgemm" => orbitkv_cuda::host::deepgemm::jit::prefetch(),
        "flashinfer" => orbitkv_cuda::host::flashinfer::jit::prefetch(),
        "flashattention" => orbitkv_cuda::host::flashattention::jit::prefetch(),
        _ => {
            eprintln!(
                "unknown provider {provider:?}; expected deepgemm, flashinfer or flashattention"
            );
            return ExitCode::FAILURE;
        }
    };
    match result {
        Ok(path) => {
            println!("{}", path.display());
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}
