use crate::kv_manager::TailActionKind;
use std::collections::BTreeSet;

use super::*;

include!("core.rs");
mod chunked;
include!("control.rs");
mod external_tier;
mod fixed_state;
mod full_sliding_prefix;
mod latent;
include!("prefix_release.rs");
include!("retirement.rs");
mod residence;
mod sliding;
