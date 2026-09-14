use crate::kv_manager::TailActionKind;
use std::collections::BTreeSet;

use super::*;
use crate::kv_manager::{BackendArenaRegistration, ManagerConfig, PageLease, PrefixSemanticKey};
use crate::plan::{
    CompiledKvPlan, KvClassSpec, KvPlanInput, RetentionKind, TokenStorageKind, compile_plan,
};

mod fixtures;
use fixtures::*;

mod chunked;
mod control;
mod core;
mod external_tier;
mod fixed_state;
mod full_sliding_prefix;
mod latent;
mod prefix_release;
mod residence;
mod retirement;
mod sliding;
