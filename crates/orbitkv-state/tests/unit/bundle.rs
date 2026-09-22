use super::*;

fn requirement(group: u32, rule: RecoveryRule) -> StateRequirement {
    let components = match rule {
        RecoveryRule::Prefix => vec![StateComponent::AttentionKv],
        RecoveryRule::Window { .. } => vec![StateComponent::SlidingWindowKv],
        RecoveryRule::Checkpoint => vec![
            StateComponent::RecurrentCheckpoint,
            StateComponent::ConvolutionState,
        ],
    };
    StateRequirement {
        group,
        rule,
        components: components.into_iter().collect(),
    }
}

fn contract(aux: Vec<StateRequirement>) -> RecoveryContract {
    let mut requirements = vec![requirement(0, RecoveryRule::Prefix)];
    requirements.extend(aux);
    RecoveryContract::compile("model/layout/rank".into(), 16, requirements).unwrap()
}

fn evidence(origin: u64, end: u64, groups: Vec<Vec<u64>>) -> StateBundle {
    StateBundle {
        namespace: "model/layout/rank".into(),
        span: TokenRange::new(origin, end).unwrap(),
        components: groups
            .into_iter()
            .enumerate()
            .map(|(group, page_ends)| BundleComponent {
                group: group as u32,
                page_ends,
            })
            .collect(),
    }
}

#[test]
fn hybrid_boundaries_require_attention_and_the_exact_checkpoint() {
    let plan = contract(vec![requirement(1, RecoveryRule::Checkpoint)]);
    let bundle = evidence(64, 128, vec![vec![80, 96, 112, 128], vec![96, 128]]);
    assert_eq!(plan.restorable_boundaries(&bundle).unwrap(), vec![96, 128]);
    let missing = evidence(64, 128, vec![vec![80, 96, 112, 128], vec![]]);
    assert!(plan.restorable_boundaries(&missing).unwrap().is_empty());
    let gap = evidence(64, 128, vec![vec![80, 112, 128], vec![128]]);
    assert!(plan.restorable_boundaries(&gap).unwrap().is_empty());
}

#[test]
fn windows_round_outward_and_require_contiguous_coverage() {
    let plan = contract(vec![requirement(1, RecoveryRule::Window { tokens: 17 })]);
    let bundle = evidence(
        64,
        144,
        vec![vec![80, 96, 112, 128, 144], vec![96, 112, 144]],
    );
    assert_eq!(plan.restorable_boundaries(&bundle).unwrap(), vec![112]);
    // A valid engine-held prefix supplies state before the query origin.
    let short = evidence(64, 80, vec![vec![80], vec![80]]);
    assert_eq!(plan.restorable_boundaries(&short).unwrap(), vec![80]);
}

#[test]
fn all_groups_and_ranks_must_share_a_legal_boundary() {
    let plan = contract(vec![
        requirement(1, RecoveryRule::Window { tokens: 32 }),
        requirement(2, RecoveryRule::Checkpoint),
    ]);
    let a = plan
        .restorable_boundaries(&evidence(
            0,
            64,
            vec![vec![16, 32, 48, 64], vec![16, 32, 48, 64], vec![32, 64]],
        ))
        .unwrap();
    let b = plan
        .restorable_boundaries(&evidence(
            0,
            64,
            vec![vec![16, 32, 48, 64], vec![16, 32, 48, 64], vec![16, 48]],
        ))
        .unwrap();
    assert_eq!(a, vec![32, 64]);
    assert_eq!(b, vec![16, 48]);
    assert!(a.iter().all(|boundary| !b.contains(boundary)));
}

#[test]
fn malformed_or_incompatible_evidence_is_rejected() {
    let plan = contract(vec![requirement(1, RecoveryRule::Checkpoint)]);
    let valid = evidence(64, 96, vec![vec![80, 96], vec![96]]);
    let mut wrong = valid.clone();
    wrong.namespace = "different weights, layout, or rank".into();
    assert_eq!(
        plan.restorable_boundaries(&wrong),
        Err(RecoveryError::IncompatibleNamespace)
    );
    for ends in [vec![96, 80], vec![80, 80], vec![64], vec![97], vec![112]] {
        let mut wrong = valid.clone();
        wrong.components[1].page_ends = ends;
        assert_eq!(
            plan.restorable_boundaries(&wrong),
            Err(RecoveryError::InvalidCoverage(1))
        );
    }
    let mut wrong = valid.clone();
    wrong.components.pop();
    assert_eq!(
        plan.restorable_boundaries(&wrong),
        Err(RecoveryError::InvalidGroup(1))
    );
    let mut wrong = valid.clone();
    wrong.components.push(wrong.components[1].clone());
    assert_eq!(
        plan.restorable_boundaries(&wrong),
        Err(RecoveryError::InvalidGroup(1))
    );
    for span in [
        TokenRange { start: 65, end: 96 },
        TokenRange { start: 96, end: 64 },
    ] {
        let mut wrong = valid.clone();
        wrong.span = span;
        assert_eq!(
            plan.restorable_boundaries(&wrong),
            Err(RecoveryError::InvalidSpan)
        );
    }
}

#[test]
fn compilation_rejects_undeclared_or_impossible_requirements() {
    for (namespace, page, requirements) in [
        ("", 16, vec![requirement(0, RecoveryRule::Prefix)]),
        ("model", 0, vec![requirement(0, RecoveryRule::Prefix)]),
        ("model", 16, vec![]),
        ("model", 16, vec![requirement(0, RecoveryRule::Checkpoint)]),
        (
            "model",
            16,
            vec![
                requirement(0, RecoveryRule::Prefix),
                requirement(1, RecoveryRule::Window { tokens: 0 }),
            ],
        ),
        (
            "model",
            16,
            vec![
                requirement(0, RecoveryRule::Prefix),
                requirement(1, RecoveryRule::Window { tokens: u64::MAX }),
            ],
        ),
    ] {
        assert!(RecoveryContract::compile(namespace.into(), page, requirements).is_err());
    }
    let mut invalid = requirement(0, RecoveryRule::Prefix);
    invalid
        .components
        .insert(StateComponent::RecurrentCheckpoint);
    assert!(RecoveryContract::compile("model".into(), 16, vec![invalid]).is_err());
}

#[test]
fn demand_is_minimal_and_uses_the_engine_held_origin() {
    let plan = contract(vec![
        requirement(1, RecoveryRule::Window { tokens: 17 }),
        requirement(2, RecoveryRule::Checkpoint),
    ]);
    for (origin, end, starts) in [
        (0, 64, [0, 32, 48]),
        (64, 144, [64, 112, 128]),
        (64, 80, [64, 64, 64]),
        (64, 64, [64, 64, 64]),
    ] {
        let ranges = plan
            .required_ranges("model/layout/rank", TokenRange { start: origin, end })
            .unwrap();
        assert_eq!(
            ranges,
            starts
                .into_iter()
                .enumerate()
                .map(|(group, start)| (group as u32, TokenRange { start, end }))
                .collect::<Vec<_>>()
        );
        if origin == end {
            continue;
        }
        let bundle = evidence(
            origin,
            end,
            starts
                .iter()
                .map(|start| ((start + 16)..=end).step_by(16).collect())
                .collect(),
        );
        assert!(plan.restorable_boundaries(&bundle).unwrap().contains(&end));
        for group in 0..3 {
            for page in 0..bundle.components[group].page_ends.len() {
                let mut missing = bundle.clone();
                missing.components[group].page_ends.remove(page);
                assert!(!plan.restorable_boundaries(&missing).unwrap().contains(&end));
            }
        }
    }
}

#[test]
fn demand_checks_identity_and_alignment_without_expanding_long_prefixes() {
    let plan = contract(vec![requirement(1, RecoveryRule::Checkpoint)]);
    let end = u64::MAX / 16 * 16;
    assert_eq!(
        plan.required_ranges("model/layout/rank", TokenRange { start: 0, end }),
        Ok(vec![
            (0, TokenRange { start: 0, end }),
            (
                1,
                TokenRange {
                    start: end - 16,
                    end
                }
            ),
        ])
    );
    assert_eq!(
        plan.required_ranges("other-model", TokenRange { start: 0, end: 16 }),
        Err(RecoveryError::IncompatibleNamespace)
    );
    for span in [
        TokenRange { start: 1, end: 16 },
        TokenRange { start: 0, end: 17 },
        TokenRange { start: 32, end: 16 },
    ] {
        assert_eq!(
            plan.required_ranges("model/layout/rank", span),
            Err(RecoveryError::InvalidSpan)
        );
    }
}
