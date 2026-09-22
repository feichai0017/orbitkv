use orbitkv_state::{
    BundleComponent, RecoveryContract, RecoveryRule, StateBundle, StateComponent, StateRequirement,
    TokenRange,
};
use pyo3::{exceptions::PyValueError, prelude::*};

#[pyclass(name = "RecoveryContract", frozen)]
pub(crate) struct PyRecoveryContract {
    contract: RecoveryContract,
}

#[pymethods]
impl PyRecoveryContract {
    #[new]
    fn new(namespace: String, page_tokens: u64, groups: Vec<(u32, String, u64)>) -> PyResult<Self> {
        let mut requirements = Vec::with_capacity(groups.len());
        for (group, kind, window) in groups {
            let (components, rule) = match kind.as_str() {
                "attention" if window == 0 => {
                    (vec![StateComponent::AttentionKv], RecoveryRule::Prefix)
                }
                "mla" if window == 0 => (vec![StateComponent::MlaKv], RecoveryRule::Prefix),
                "window" => (
                    vec![StateComponent::SlidingWindowKv],
                    RecoveryRule::Window { tokens: window },
                ),
                "recurrent" if window == 0 => (
                    vec![
                        StateComponent::RecurrentCheckpoint,
                        StateComponent::ConvolutionState,
                    ],
                    RecoveryRule::Checkpoint,
                ),
                _ => {
                    return Err(PyValueError::new_err(
                        "unsupported recovery group or window",
                    ));
                }
            };
            requirements.push(StateRequirement {
                group,
                components: components.into_iter().collect(),
                rule,
            });
        }
        let contract = RecoveryContract::compile(namespace, page_tokens, requirements)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self { contract })
    }

    fn restorable_boundaries(
        &self,
        namespace: String,
        start: u64,
        end: u64,
        groups: Vec<(u32, Vec<u64>)>,
    ) -> PyResult<Vec<u64>> {
        let bundle = StateBundle {
            namespace,
            span: TokenRange { start, end },
            components: groups
                .into_iter()
                .map(|(group, page_ends)| BundleComponent { group, page_ends })
                .collect(),
        };
        self.contract
            .restorable_boundaries(&bundle)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }
}
