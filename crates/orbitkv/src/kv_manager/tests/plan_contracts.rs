//! Admission tests for compiled manager plans.

use super::*;

#[test]
fn rejects_noncanonical_profiles_and_periods() {
    let wrong_page = sliding_plan(18, 8);
    let result = CanonicalKvManager::new(
        &wrong_page,
        ManagerConfig {
            maximum_requests: 1,
            maximum_operations: 1,
            maximum_prefixes: 1,
            maximum_reclamations: 1,
            maximum_step_tokens: 1,
        },
        &[BackendArenaRegistration {
            pool_id: 1,
            class_id: 0,
            backend_domain: 0,
            page_count: 3,
            reserved: 0,
            backend_base_index: 0,
        }],
    );
    assert!(matches!(result, Err(KvManagerError::UnsupportedProfile(_))));

    let mut malformed = sliding_plan(18, 16);
    malformed.classes[0].slot_count = Some(99);
    let result = CanonicalKvManager::new(
        &malformed,
        ManagerConfig {
            maximum_requests: 1,
            maximum_operations: 1,
            maximum_prefixes: 1,
            maximum_reclamations: 1,
            maximum_step_tokens: 1,
        },
        &[BackendArenaRegistration {
            pool_id: 1,
            class_id: 0,
            backend_domain: 0,
            page_count: 99,
            reserved: 0,
            backend_base_index: 0,
        }],
    );
    assert!(matches!(result, Err(KvManagerError::UnsupportedProfile(_))));
}

#[test]
fn rejects_malicious_retirement_programs_and_pool_aliases() {
    let plan = hybrid_plan(18);
    let mut malicious_layout = plan.layout_program().expect("layout");
    malicious_layout.classes[0].retirement = RetirementProgram::BlockEndPlus { offset_tokens: 17 };
    assert!(matches!(
        validate_class_program(&plan.classes[0], &malicious_layout.classes[0]),
        Err(KvManagerError::UnsupportedProfile(_))
    ));
    malicious_layout.classes[1].retirement = RetirementProgram::Never;
    assert!(matches!(
        validate_class_program(&plan.classes[1], &malicious_layout.classes[1]),
        Err(KvManagerError::UnsupportedProfile(_))
    ));
    malicious_layout.classes[1].retirement = RetirementProgram::BlockEndPlus { offset_tokens: 18 };
    assert!(matches!(
        validate_class_program(&plan.classes[1], &malicious_layout.classes[1]),
        Err(KvManagerError::UnsupportedProfile(_))
    ));

    let result = CanonicalKvManager::new(
        &plan,
        ManagerConfig {
            maximum_requests: 1,
            maximum_operations: 1,
            maximum_prefixes: 1,
            maximum_reclamations: 1,
            maximum_step_tokens: 1,
        },
        &[backend(0, 71, 1, 0), backend(1, 71, 3, 0)],
    );
    assert!(matches!(result, Err(KvManagerError::InvalidConfiguration)));
}

#[test]
fn backend_ranges_are_disjoint_within_each_domain() {
    let plan = hybrid_plan(18);
    let mut backends = [backend(0, 73, 4, 100), backend(1, 74, 3, 103)];
    backends[1].backend_domain = backends[0].backend_domain;
    let settings = ManagerConfig {
        maximum_requests: 1,
        maximum_operations: 1,
        maximum_prefixes: 1,
        maximum_reclamations: 7,
        maximum_step_tokens: 64,
    };
    assert!(matches!(
        CanonicalKvManager::new(&plan, settings, &backends),
        Err(KvManagerError::InvalidConfiguration)
    ));

    backends[1].backend_base_index = 104;
    assert!(CanonicalKvManager::new(&plan, settings, &backends).is_ok());

    backends[1].backend_domain += 1;
    backends[1].backend_base_index = 100;
    assert!(CanonicalKvManager::new(&plan, settings, &backends).is_ok());
}

#[test]
fn accepts_whole_domain_single_class_chunked_profile() {
    let plan = chunked_plan(32);
    let manager = manager_for_plan(&plan, &[backend(0, 81, 2, 400)], 32, 2);
    assert_eq!(manager.classes.len(), 1);
    let class = manager.classes[0];
    assert_eq!(class.retention, RetentionKind::Chunked);
    assert_eq!(class.chunk_tokens, Some(32));
    assert_eq!(class.blocks_per_epoch, Some(2));
}

#[test]
fn rejects_chunked_multi_class_and_non_whole_domain_profiles() {
    let mut mixed = chunked_plan(32);
    let mut full = full_plan(CANONICAL_PAGE_TOKENS).classes.remove(0);
    full.spec.layers = vec![1];
    mixed.classes.push(full);
    let result = CanonicalKvManager::new(
        &mixed,
        ManagerConfig {
            maximum_requests: 1,
            maximum_operations: 1,
            maximum_prefixes: 1,
            maximum_reclamations: 4,
            maximum_step_tokens: 32,
        },
        &[backend(0, 82, 2, 0), backend(1, 83, 2, 0)],
    );
    assert!(matches!(result, Err(KvManagerError::UnsupportedProfile(_))));

    let plan = chunked_plan(32);
    let layout = plan.layout_program().expect("chunked layout");
    let mut region = plan.classes[0].clone();
    region.block_domain = BlockDomain {
        start_block: 1,
        end_block_exclusive: None,
    };
    assert!(matches!(
        validate_class_program(&region, &layout.classes[0]),
        Err(KvManagerError::UnsupportedProfile(_))
    ));

    let mut headed = plan.classes[0].clone();
    headed.kv_head_range = Some(crate::retention::KvHeadRange {
        start: 0,
        end_exclusive: 1,
    });
    assert!(matches!(
        validate_class_program(&headed, &layout.classes[0]),
        Err(KvManagerError::UnsupportedProfile(_))
    ));
}

#[test]
fn rejects_malformed_chunked_geometry_and_programs() {
    let plan = chunked_plan(32);
    let layout = plan.layout_program().expect("chunked layout");

    let mut malformed = plan.classes[0].clone();
    malformed.chunk_tokens = Some(17);
    assert!(matches!(
        validate_class_program(&malformed, &layout.classes[0]),
        Err(KvManagerError::UnsupportedProfile(_))
    ));

    let mut malformed = plan.classes[0].clone();
    malformed.slot_count = Some(3);
    assert!(matches!(
        validate_class_program(&malformed, &layout.classes[0]),
        Err(KvManagerError::UnsupportedProfile(_))
    ));

    let mut malformed_layout = layout.classes[0].clone();
    malformed_layout.minimum_slots_per_request = Some(3);
    assert!(matches!(
        validate_class_program(&plan.classes[0], &malformed_layout),
        Err(KvManagerError::UnsupportedProfile(_))
    ));

    let mut malformed_layout = layout.classes[0].clone();
    malformed_layout.address = AddressProgram::Periodic { period_blocks: 2 };
    assert!(matches!(
        validate_class_program(&plan.classes[0], &malformed_layout),
        Err(KvManagerError::UnsupportedProfile(_))
    ));

    let mut malformed_layout = layout.classes[0].clone();
    malformed_layout.retirement = RetirementProgram::BlockEndPlus { offset_tokens: 31 };
    assert!(matches!(
        validate_class_program(&plan.classes[0], &malformed_layout),
        Err(KvManagerError::UnsupportedProfile(_))
    ));

    let mut malformed_layout = layout.classes[0].clone();
    malformed_layout.address = AddressProgram::ResettableArena {
        blocks_per_epoch: 3,
    };
    assert!(matches!(
        validate_class_program(&plan.classes[0], &malformed_layout),
        Err(KvManagerError::UnsupportedProfile(_))
    ));

    let mut malformed_layout = layout.classes[0].clone();
    malformed_layout.retirement = RetirementProgram::EpochEnd {
        blocks_per_epoch: 3,
    };
    assert!(matches!(
        validate_class_program(&plan.classes[0], &malformed_layout),
        Err(KvManagerError::UnsupportedProfile(_))
    ));
}
