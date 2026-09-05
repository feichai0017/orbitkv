use crate::kv_manager::TailActionKind;

use super::*;

include!("core.rs");
mod chunked;
include!("control.rs");
mod external_tier;
mod full_sliding_prefix;
mod latent;
include!("prefix_release.rs");
mod relocation;
include!("retirement.rs");
mod sliding;
