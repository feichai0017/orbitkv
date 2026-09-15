//! Pure backend functions callable from egglog rules.
//!
//! Functions consume scalar facts and return values; graph matching and
//! implementation alternatives remain in the rules that call them. A function
//! must be deterministic and must not query devices, perform I/O or mutate
//! backend state. Its implementation identity belongs in the backend's artifact
//! provenance, just like the rewrite text.

use std::{any::TypeId, sync::Arc};

pub use egglog;
use egglog::{ExecutionState, Primitive, Value, ast::Span, constraint::TypeConstraint};

#[derive(Clone)]
pub struct EgglogPrimitive {
    definition: Arc<dyn Primitive + Send + Sync>,
    implementation: TypeId,
}

impl EgglogPrimitive {
    /// Register a stateless definition. All configuration must be supplied as
    /// egglog arguments, so repeated registration of this type is identical.
    pub fn new<T: Primitive + Default + Send + Sync + 'static>() -> Self {
        Self {
            definition: Arc::new(T::default()),
            implementation: TypeId::of::<T>(),
        }
    }
}

impl Primitive for EgglogPrimitive {
    fn name(&self) -> &str {
        self.definition.name()
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        self.definition.get_type_constraints(span)
    }

    fn apply(&self, state: &mut ExecutionState<'_>, args: &[Value]) -> Option<Value> {
        self.definition.apply(state, args)
    }
}

pub(super) fn collect(
    definitions: impl IntoIterator<Item = EgglogPrimitive>,
) -> Vec<EgglogPrimitive> {
    let mut seen = rustc_hash::FxHashMap::default();
    definitions
        .into_iter()
        .filter(
            |definition| match seen.entry(definition.name().to_owned()) {
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(definition.implementation);
                    true
                }
                std::collections::hash_map::Entry::Occupied(entry) => {
                    assert_eq!(
                        *entry.get(),
                        definition.implementation,
                        "conflicting egglog primitive implementations for {}",
                        definition.name()
                    );
                    false
                }
            },
        )
        .collect()
}

#[cfg(test)]
#[path = "../../tests/unit/egglog_utils/primitives.rs"]
mod tests;
