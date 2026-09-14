use super::*;
use crate::plan::{KvClassSpec, KvPlanInput, compile_plan, compile_retention_program};
use crate::retention::{IntExpr, Predicate, RetentionProgramInput, RetentionStateDecl};
use std::collections::BTreeMap;
use std::time::Instant;

mod fixtures;
pub(super) mod model;

use fixtures::*;
use model::{
    DEVICE_KV_ACCESS_READ, DEVICE_KV_ACCESS_WRITE, DEVICE_KV_NEEDS_BINDING, DeviceKvEntry,
};

mod chunked;
mod fault_atomicity;
mod lifecycle;
mod performance;
mod plan_contracts;
mod prefix_cow;
mod properties;
