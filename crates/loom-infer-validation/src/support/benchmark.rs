//! JSON Lines schema shared by Loom performance runners.

use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct BenchmarkRecord<'a> {
    pub schema_version: u32,
    pub provider: &'a str,
    pub provider_version: &'a str,
    pub provider_commit: &'a str,
    pub run_label: &'a str,
    pub measurement: &'a str,
    pub operator: &'a str,
    pub case: &'a str,
    pub dtype: &'a str,
    pub layout: &'a str,
    pub execution: serde_json::Value,
    pub kernels_per_call: usize,
    pub shape: serde_json::Value,
    pub fixture_id: &'a str,
    pub fixture_digests: serde_json::Value,
    pub warmup_launches: usize,
    pub launches_per_sample: usize,
    pub samples_us: Vec<f64>,
}

impl BenchmarkRecord<'_> {
    pub fn write_json_line(&self) -> Result<(), serde_json::Error> {
        println!("{}", serde_json::to_string(self)?);
        Ok(())
    }
}
