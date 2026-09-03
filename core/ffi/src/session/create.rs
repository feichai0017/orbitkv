use orbitkv::kv_manager::CanonicalKvManager;
use orbitkv::plan::{CompiledKvPlan, RetentionKind, TokenStorageKind};
use orbitkv::runtime_session::CacheSharingPolicy;
use orbitkv::{KvPlanInput, RetentionProgramInput, compile_plan, compile_retention_program};

use crate::wire::{
    ORBITKV_PLAN_FORMAT_KV_PLAN, ORBITKV_PLAN_FORMAT_RETENTION_IR, OrbitKvArenaIdentity,
    OrbitKvBackendArenaRegistration,
};

use super::{
    ORBITKV_CACHE_SHARING_POLICY_REQUEST_PRIVATE, ORBITKV_CACHE_SHARING_POLICY_SHARED_PREFIX,
    ORBITKV_STATUS_INVALID_ARGUMENT, OrbitKvSessionCreateConfig, core_status, exact_len, invalid,
};

pub(super) struct SessionMetadata {
    pub session_epoch: u64,
    pub total_page_capacity: u32,
    pub maximum_requests: u32,
    pub maximum_operations: u32,
    pub maximum_prefixes: u32,
    pub class_count: u32,
    pub maximum_write_intents_per_item: u32,
    pub maximum_completion_outputs_per_item: u32,
    pub arena_identities: Box<[OrbitKvArenaIdentity]>,
    pub cache_sharing_policy: CacheSharingPolicy,
}

#[allow(clippy::too_many_lines)]
pub(super) fn compile_session_manager(
    plan_json: &[u8],
    config: OrbitKvSessionCreateConfig,
    backends: &[OrbitKvBackendArenaRegistration],
) -> Result<(CanonicalKvManager, SessionMetadata), (i32, String)> {
    if config.reserved != 0 {
        return invalid("session create config reserved field must be zero");
    }
    let cache_sharing_policy = match config.cache_sharing_policy {
        ORBITKV_CACHE_SHARING_POLICY_REQUEST_PRIVATE => CacheSharingPolicy::RequestPrivate,
        ORBITKV_CACHE_SHARING_POLICY_SHARED_PREFIX => CacheSharingPolicy::SharedPrefix,
        _ => return invalid("session cache sharing policy is unsupported"),
    };
    let config = config.manager;
    if config.reserved != 0 {
        return invalid("manager config reserved field must be zero");
    }
    if !matches!(
        config.plan_format,
        ORBITKV_PLAN_FORMAT_KV_PLAN | ORBITKV_PLAN_FORMAT_RETENTION_IR
    ) {
        return invalid("manager config plan format is unsupported");
    }
    if backends.is_empty() {
        return invalid("at least one backend arena registration is required");
    }
    if backends.iter().any(|backend| backend.reserved != 0) {
        return invalid("backend registration reserved field must be zero");
    }
    let plan = match config.plan_format {
        ORBITKV_PLAN_FORMAT_KV_PLAN => {
            let input = serde_json::from_slice::<KvPlanInput>(plan_json).map_err(|error| {
                (
                    ORBITKV_STATUS_INVALID_ARGUMENT,
                    format!("invalid KvPlanInput JSON: {error}"),
                )
            })?;
            compile_plan(input).map_err(|error| {
                (
                    ORBITKV_STATUS_INVALID_ARGUMENT,
                    format!("invalid canonical KV plan: {error}"),
                )
            })?
        }
        ORBITKV_PLAN_FORMAT_RETENTION_IR => {
            let input =
                serde_json::from_slice::<RetentionProgramInput>(plan_json).map_err(|error| {
                    (
                        ORBITKV_STATUS_INVALID_ARGUMENT,
                        format!("invalid RetentionProgramInput JSON: {error}"),
                    )
                })?;
            compile_retention_program(input).map_err(|error| {
                (
                    ORBITKV_STATUS_INVALID_ARGUMENT,
                    format!("invalid retention program: {error}"),
                )
            })?
        }
        _ => unreachable!("plan format was validated"),
    };
    validate_session_profile(&plan, cache_sharing_policy)?;
    let class_count = u32::try_from(plan.classes.len()).map_err(|_| {
        (
            ORBITKV_STATUS_INVALID_ARGUMENT,
            "class count exceeds uint32_t".into(),
        )
    })?;
    if class_count != exact_len(backends.len()) {
        return invalid("backend arena count must match compiled attention classes");
    }
    let page_tokens = u32::try_from(plan.page_tokens).map_err(|_| {
        (
            ORBITKV_STATUS_INVALID_ARGUMENT,
            "page token count exceeds uint32_t".into(),
        )
    })?;
    let pages_per_step = u64::from(config.maximum_step_tokens).div_ceil(plan.page_tokens);
    let maximum_write_intents_per_item = u32::try_from(
        u64::from(class_count)
            .checked_mul(pages_per_step)
            .ok_or_else(|| {
                (
                    ORBITKV_STATUS_INVALID_ARGUMENT,
                    "prepare output bound overflows".into(),
                )
            })?,
    )
    .map_err(|_| {
        (
            ORBITKV_STATUS_INVALID_ARGUMENT,
            "prepare output bound exceeds uint32_t".into(),
        )
    })?;
    let maximum_completion_outputs_per_item = u32::try_from(plan.classes.iter().try_fold(
        0_u64,
        |sum, class| -> Result<u64, (i32, String)> {
            let delta_bound = pages_per_step.checked_add(2).ok_or_else(|| {
                (
                    ORBITKV_STATUS_INVALID_ARGUMENT,
                    "completion output bound overflows".into(),
                )
            })?;
            let class_bound = if class.spec.retention == RetentionKind::Chunked {
                delta_bound.max(class.slot_count.unwrap_or(0))
            } else {
                delta_bound
            };
            sum.checked_add(class_bound).ok_or_else(|| {
                (
                    ORBITKV_STATUS_INVALID_ARGUMENT,
                    "completion output bound overflows".into(),
                )
            })
        },
    )?)
    .map_err(|_| {
        (
            ORBITKV_STATUS_INVALID_ARGUMENT,
            "completion output bound exceeds uint32_t".into(),
        )
    })?;
    let total_page_capacity = backends.iter().try_fold(0_u32, |sum, backend| {
        sum.checked_add(backend.page_count).ok_or_else(|| {
            (
                ORBITKV_STATUS_INVALID_ARGUMENT,
                "total page capacity exceeds uint32_t".into(),
            )
        })
    })?;
    config
        .maximum_requests
        .checked_mul(total_page_capacity)
        .ok_or_else(|| {
            (
                ORBITKV_STATUS_INVALID_ARGUMENT,
                "maximum requests times total page capacity exceeds uint32_t".into(),
            )
        })?;
    let core_backends = backends.iter().copied().map(Into::into).collect::<Vec<_>>();
    let manager = CanonicalKvManager::new(&plan, config.into(), &core_backends)
        .map_err(|error| (core_status(&error), error.to_string()))?;
    let arena_identities = manager
        .arena_stats()
        .iter()
        .map(|arena| {
            let backend = backends
                .iter()
                .find(|backend| backend.class_id == arena.class_id)
                .expect("core class originated from validated registration");
            OrbitKvArenaIdentity {
                engine_epoch: arena.engine_epoch,
                pool_epoch: arena.pool_epoch,
                backend_base_index: backend.backend_base_index,
                pool_id: arena.pool_id,
                page_count: arena.page_count,
                page_tokens,
                class_id: arena.class_id,
                backend_domain: arena.backend_domain,
                first_page_id: arena.first_page_id,
                reserved: 0,
            }
        })
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let session_epoch = arena_identities
        .first()
        .expect("canonical manager has at least one class")
        .engine_epoch;
    Ok((
        manager,
        SessionMetadata {
            session_epoch,
            total_page_capacity,
            maximum_requests: config.maximum_requests,
            maximum_operations: config.maximum_operations,
            maximum_prefixes: config.maximum_prefixes,
            class_count,
            maximum_write_intents_per_item,
            maximum_completion_outputs_per_item,
            arena_identities,
            cache_sharing_policy,
        },
    ))
}

fn validate_session_profile(
    plan: &CompiledKvPlan,
    policy: CacheSharingPolicy,
) -> Result<(), (i32, String)> {
    let classes = plan.classes.as_slice();
    let full_token = |index: usize| {
        classes.get(index).is_some_and(|class| {
            class.spec.retention == RetentionKind::Full
                && class.spec.storage == TokenStorageKind::TokenKv
        })
    };
    let sliding_token = |index: usize| {
        classes.get(index).is_some_and(|class| {
            class.spec.retention == RetentionKind::Sliding
                && class.spec.storage == TokenStorageKind::TokenKv
        })
    };
    let chunked_token = |index: usize| {
        classes.get(index).is_some_and(|class| {
            class.spec.retention == RetentionKind::Chunked
                && class.spec.storage == TokenStorageKind::TokenKv
        })
    };
    let full_latent = |index: usize| {
        classes.get(index).is_some_and(|class| {
            class.spec.retention == RetentionKind::Full
                && class.spec.storage == TokenStorageKind::LatentKv
        })
    };
    let full = classes.len() == 1 && full_token(0);
    let full_sliding = classes.len() == 2 && full_token(0) && sliding_token(1);
    let supported = match policy {
        CacheSharingPolicy::SharedPrefix => full || full_sliding,
        CacheSharingPolicy::RequestPrivate => {
            full || full_sliding
                || (classes.len() == 1 && (sliding_token(0) || chunked_token(0) || full_latent(0)))
        }
    };
    if !supported {
        return invalid(match policy {
            CacheSharingPolicy::SharedPrefix => {
                "shared-prefix sessions require exact Full token_kv or ordered Full+Sliding token_kv"
            }
            CacheSharingPolicy::RequestPrivate => "request-private session profile is unsupported",
        });
    }
    Ok(())
}
