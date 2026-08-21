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
