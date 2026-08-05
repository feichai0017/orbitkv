//! Stable key-value prefixes for machine-readable validation output.

use std::fmt::{self, Display, Formatter};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GateCase<'a> {
    gate: &'a str,
    case: &'a str,
}

impl<'a> GateCase<'a> {
    pub const fn new(gate: &'a str, case: &'a str) -> Self {
        Self { gate, case }
    }
}

impl Display for GateCase<'_> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "gate={} case={} status=pass",
            self.gate, self.case
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_case_preserves_the_machine_readable_prefix() {
        assert_eq!(
            GateCase::new("single_decode_h20", "mha_l1").to_string(),
            "gate=single_decode_h20 case=mha_l1 status=pass"
        );
    }
}
