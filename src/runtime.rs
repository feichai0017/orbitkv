use std::collections::{BTreeMap, BTreeSet};

use thiserror::Error;

use crate::{
    CellVersion, CompiledKvClass, CompiledKvPlan, LayoutProgram, PlanError, RetentionKind,
    TemporalAddress,
};

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct LogicalBlock {
    pub class_name: String,
    pub ordinal: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Submission {
    pub submission_id: u64,
    pub blocks: BTreeSet<LogicalBlock>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ResidentTemporalBlock {
    pub logical: LogicalBlock,
    pub temporal: TemporalAddress,
}

#[derive(Clone, Debug, Default)]
struct BlockState {
    version: Option<CellVersion>,
    logical: Option<LogicalBlock>,
    gpu_pins: BTreeSet<u64>,
    retiring: bool,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum RuntimeError {
    #[error("boundary cannot move backwards")]
    BoundaryMovedBackwards,
    #[error("unknown KV class {0:?}")]
    UnknownClass(String),
    #[error("unknown submission {0}")]
    UnknownSubmission(u64),
    #[error("logical block is not resident: {0:?}")]
    BlockNotResident(LogicalBlock),
    #[error("unsafe reuse blocked: {0:?} is still pinned")]
    UnsafeReuse(LogicalBlock),
    #[error("slot collision: {0:?} is still semantically live")]
    SlotCollision(LogicalBlock),
    #[error("submission generation exhausted")]
    SubmissionGenerationExhausted,
    #[error("block generation exhausted for {0:?}")]
    BlockGenerationExhausted(LogicalBlock),
    #[error("compiled slot count does not fit the host address space for {0:?}")]
    SlotCountTooLarge(String),
    #[error("logical block ordinal does not fit the host address space: {0}")]
    BlockOrdinalTooLarge(u64),
    #[error("full-retention block became semantically dead: {0:?}")]
    FullBlockRetired(LogicalBlock),
    #[error("simulator stored the wrong temporal version for {0:?}")]
    TemporalVersionMismatch(LogicalBlock),
    #[error(transparent)]
    Plan(#[from] PlanError),
}

pub struct KvRuntimeSimulator {
    plan: CompiledKvPlan,
    layout: LayoutProgram,
    boundary: u64,
    next_submission: u64,
    classes: BTreeMap<String, CompiledKvClass>,
    bounded_slots: BTreeMap<String, Vec<BlockState>>,
    append_slots: BTreeMap<String, Vec<BlockState>>,
    submissions: BTreeMap<u64, Submission>,
}

impl KvRuntimeSimulator {
    /// Creates a reference runtime for a compiled block plan.
    ///
    /// # Errors
    ///
    /// Returns an error if a compiled slot count cannot fit the host address
    /// space.
    pub fn new(plan: CompiledKvPlan) -> Result<Self, RuntimeError> {
        let layout = plan.layout_program()?;
        let mut classes = BTreeMap::new();
        let mut bounded_slots = BTreeMap::new();
        let mut append_slots = BTreeMap::new();
        for class in &plan.classes {
            classes.insert(class.spec.name.clone(), class.clone());
            if let Some(slot_count) = class.slot_count {
                let slots = usize::try_from(slot_count)
                    .map_err(|_| RuntimeError::SlotCountTooLarge(class.spec.name.clone()))?;
                bounded_slots.insert(class.spec.name.clone(), vec![BlockState::default(); slots]);
            } else {
                append_slots.insert(class.spec.name.clone(), Vec::new());
            }
        }
        Ok(Self {
            plan,
            layout,
            boundary: 0,
            next_submission: 1,
            classes,
            bounded_slots,
            append_slots,
            submissions: BTreeMap::new(),
        })
    }

    #[must_use]
    pub const fn boundary(&self) -> u64 {
        self.boundary
    }

    /// Appends tokens until the requested pre-query boundary.
    ///
    /// # Errors
    ///
    /// Returns an error on backward progress, unsafe slot reuse, invalid plan
    /// state, or generation exhaustion.
    pub fn append_to(&mut self, boundary: u64) -> Result<(), RuntimeError> {
        if boundary < self.boundary {
            return Err(RuntimeError::BoundaryMovedBackwards);
        }
        for token in self.boundary..boundary {
            self.mark_semantic_retirement()?;
            self.materialize_block(token / self.plan.page_tokens)?;
            self.boundary = token + 1;
            self.mark_semantic_retirement()?;
        }
        Ok(())
    }

    /// Pins the current immutable live-block view for one simulated submission.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid plan state or submission generation
    /// exhaustion.
    pub fn submit_view(&mut self) -> Result<Submission, RuntimeError> {
        let blocks = self.live_blocks()?.into_iter().collect::<BTreeSet<_>>();
        let submission = Submission {
            submission_id: self.next_submission,
            blocks,
        };
        self.next_submission = self
            .next_submission
            .checked_add(1)
            .ok_or(RuntimeError::SubmissionGenerationExhausted)?;
        for logical in &submission.blocks {
            self.state_for_mut(logical)?
                .gpu_pins
                .insert(submission.submission_id);
        }
        self.submissions
            .insert(submission.submission_id, submission.clone());
        Ok(submission)
    }

    /// Settles a submission and releases its block pins.
    ///
    /// # Errors
    ///
    /// Returns an error if the submission is unknown or a pinned block is no
    /// longer resident.
    pub fn complete(&mut self, submission_id: u64) -> Result<(), RuntimeError> {
        let submission = self
            .submissions
            .remove(&submission_id)
            .ok_or(RuntimeError::UnknownSubmission(submission_id))?;
        for logical in &submission.blocks {
            let state = self.state_for_mut(logical)?;
            state.gpu_pins.remove(&submission_id);
            if state.retiring && state.gpu_pins.is_empty() {
                state.logical = None;
                state.retiring = false;
            }
        }
        Ok(())
    }

    /// Returns the semantically live blocks materialized by the runtime.
    ///
    /// # Errors
    ///
    /// Returns an error if the compiled continuation program is invalid.
    pub fn live_blocks(&self) -> Result<Vec<LogicalBlock>, RuntimeError> {
        let continuation = self.plan.continuation_blocks(self.boundary)?;
        let mut result = Vec::new();
        for class in &self.plan.classes {
            for &ordinal in &continuation[&class.spec.name] {
                let logical = LogicalBlock {
                    class_name: class.spec.name.clone(),
                    ordinal,
                };
                if self
                    .try_state_for(&logical)?
                    .is_some_and(|state| state.logical.as_ref() == Some(&logical))
                {
                    result.push(logical);
                }
            }
        }
        Ok(result)
    }

    #[must_use]
    pub fn resident_blocks(&self) -> Vec<LogicalBlock> {
        self.bounded_slots
            .values()
            .chain(self.append_slots.values())
            .flat_map(|states| states.iter())
            .filter_map(|state| state.logical.clone())
            .collect()
    }

    /// Returns actual resident blocks with their compiler-defined cells and
    /// versions.
    ///
    /// # Errors
    ///
    /// Returns an error if the simulator's stored version diverges from the
    /// address program.
    pub fn resident_temporal_blocks(
        &self,
        request_id: &str,
    ) -> Result<Vec<ResidentTemporalBlock>, RuntimeError> {
        self.bounded_slots
            .values()
            .chain(self.append_slots.values())
            .flat_map(|states| states.iter())
            .filter_map(|state| state.logical.as_ref().map(|logical| (state, logical)))
            .map(|(state, logical)| {
                let temporal = self.layout.temporal_address(
                    request_id,
                    &logical.class_name,
                    logical.ordinal,
                )?;
                if state.version != Some(temporal.version) {
                    return Err(RuntimeError::TemporalVersionMismatch(logical.clone()));
                }
                Ok(ResidentTemporalBlock {
                    logical: logical.clone(),
                    temporal,
                })
            })
            .collect()
    }

    #[must_use]
    pub fn retiring_blocks(&self) -> Vec<LogicalBlock> {
        self.bounded_slots
            .values()
            .chain(self.append_slots.values())
            .flat_map(|states| states.iter())
            .filter(|state| state.retiring)
            .filter_map(|state| state.logical.clone())
            .collect()
    }

    fn materialize_block(&mut self, ordinal: u64) -> Result<(), RuntimeError> {
        let classes = self.plan.classes.clone();
        for class in classes {
            let logical = LogicalBlock {
                class_name: class.spec.name.clone(),
                ordinal,
            };
            let address =
                self.layout
                    .temporal_address("simulator", &logical.class_name, ordinal)?;
            let state = self.slot_for_mut(&class, ordinal)?;
            if state.logical.as_ref() == Some(&logical) {
                continue;
            }
            if let Some(previous) = &state.logical {
                if !state.gpu_pins.is_empty() {
                    return Err(RuntimeError::UnsafeReuse(previous.clone()));
                }
                if !state.retiring {
                    return Err(RuntimeError::SlotCollision(previous.clone()));
                }
            }
            state.version = Some(address.version);
            state.logical = Some(logical);
            state.retiring = false;
        }
        Ok(())
    }

    fn mark_semantic_retirement(&mut self) -> Result<(), RuntimeError> {
        let live = self.live_blocks()?.into_iter().collect::<BTreeSet<_>>();
        let classes = self.plan.classes.clone();
        for class in classes {
            let states = if class.slot_count.is_some() {
                self.bounded_slots
                    .get_mut(&class.spec.name)
                    .expect("compiled bounded class has slots")
            } else {
                self.append_slots
                    .get_mut(&class.spec.name)
                    .expect("compiled full class has append slots")
            };
            for state in states {
                let Some(logical) = &state.logical else {
                    continue;
                };
                if live.contains(logical) {
                    continue;
                }
                if class.spec.retention == RetentionKind::Full {
                    return Err(RuntimeError::FullBlockRetired(logical.clone()));
                }
                if state.gpu_pins.is_empty() {
                    state.logical = None;
                    state.retiring = false;
                } else {
                    state.retiring = true;
                }
            }
        }
        Ok(())
    }

    fn slot_for_mut(
        &mut self,
        class: &CompiledKvClass,
        ordinal: u64,
    ) -> Result<&mut BlockState, RuntimeError> {
        if let Some(slot_count) = class.slot_count {
            let address = self
                .layout
                .temporal_address("simulator", &class.spec.name, ordinal)?;
            debug_assert!(address.cell.cell_index < slot_count);
            let slot = usize::try_from(address.cell.cell_index)
                .map_err(|_| RuntimeError::BlockOrdinalTooLarge(ordinal))?;
            Ok(&mut self
                .bounded_slots
                .get_mut(&class.spec.name)
                .ok_or_else(|| RuntimeError::UnknownClass(class.spec.name.clone()))?[slot])
        } else {
            let ordinal = usize::try_from(ordinal)
                .map_err(|_| RuntimeError::BlockOrdinalTooLarge(ordinal))?;
            let states = self
                .append_slots
                .get_mut(&class.spec.name)
                .ok_or_else(|| RuntimeError::UnknownClass(class.spec.name.clone()))?;
            if states.len() <= ordinal {
                states.resize_with(ordinal + 1, BlockState::default);
            }
            Ok(&mut states[ordinal])
        }
    }

    fn state_for_mut(&mut self, logical: &LogicalBlock) -> Result<&mut BlockState, RuntimeError> {
        let class = self
            .classes
            .get(&logical.class_name)
            .ok_or_else(|| RuntimeError::UnknownClass(logical.class_name.clone()))?
            .clone();
        let state = self.slot_for_mut(&class, logical.ordinal)?;
        if state.logical.as_ref() != Some(logical) {
            return Err(RuntimeError::BlockNotResident(logical.clone()));
        }
        Ok(state)
    }

    fn try_state_for(&self, logical: &LogicalBlock) -> Result<Option<&BlockState>, RuntimeError> {
        let class = self
            .classes
            .get(&logical.class_name)
            .ok_or_else(|| RuntimeError::UnknownClass(logical.class_name.clone()))?;
        if let Some(slot_count) = class.slot_count {
            let address =
                self.layout
                    .temporal_address("simulator", &logical.class_name, logical.ordinal)?;
            debug_assert!(address.cell.cell_index < slot_count);
            let slot = usize::try_from(address.cell.cell_index)
                .map_err(|_| RuntimeError::BlockOrdinalTooLarge(logical.ordinal))?;
            Ok(self.bounded_slots[&logical.class_name].get(slot))
        } else {
            let ordinal = usize::try_from(logical.ordinal)
                .map_err(|_| RuntimeError::BlockOrdinalTooLarge(logical.ordinal))?;
            Ok(self.append_slots[&logical.class_name].get(ordinal))
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{KvClassSpec, KvPlanInput, RetentionKind, compile_plan};

    use super::*;

    fn sliding_plan() -> CompiledKvPlan {
        compile_plan(KvPlanInput {
            page_tokens: 16,
            classes: vec![KvClassSpec {
                name: "swa".into(),
                layers: vec![0],
                retention: RetentionKind::Sliding,
                bytes_per_token_per_layer: 128,
                window_tokens: Some(32),
            }],
        })
        .unwrap()
    }

    #[test]
    fn resident_sliding_blocks_remain_bounded() {
        let mut runtime = KvRuntimeSimulator::new(sliding_plan()).unwrap();
        runtime.append_to(512).unwrap();
        assert!(runtime.resident_blocks().len() <= 3);
        assert_eq!(
            runtime.live_blocks().unwrap(),
            vec![
                LogicalBlock {
                    class_name: "swa".into(),
                    ordinal: 30
                },
                LogicalBlock {
                    class_name: "swa".into(),
                    ordinal: 31
                }
            ]
        );
    }

    #[test]
    fn gpu_pin_blocks_unsafe_modulo_reuse() {
        let mut runtime = KvRuntimeSimulator::new(sliding_plan()).unwrap();
        runtime.append_to(32).unwrap();
        let submission = runtime.submit_view().unwrap();
        runtime.append_to(48).unwrap();
        assert!(runtime.retiring_blocks().contains(&LogicalBlock {
            class_name: "swa".into(),
            ordinal: 0
        }));
        assert!(matches!(
            runtime.append_to(64),
            Err(RuntimeError::UnsafeReuse(_))
        ));
        runtime.complete(submission.submission_id).unwrap();
        runtime.append_to(64).unwrap();
    }

    #[test]
    fn full_blocks_never_retire() {
        let plan = compile_plan(KvPlanInput {
            page_tokens: 16,
            classes: vec![KvClassSpec {
                name: "full".into(),
                layers: vec![0],
                retention: RetentionKind::Full,
                bytes_per_token_per_layer: 128,
                window_tokens: None,
            }],
        })
        .unwrap();
        let mut runtime = KvRuntimeSimulator::new(plan).unwrap();
        runtime.append_to(128).unwrap();
        assert_eq!(runtime.resident_blocks().len(), 8);
        assert!(runtime.retiring_blocks().is_empty());
    }
}
