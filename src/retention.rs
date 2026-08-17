use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum IntExpr {
    QueryPosition,
    KeyPosition,
    Constant {
        value: i64,
    },
    Add {
        lhs: Box<IntExpr>,
        rhs: Box<IntExpr>,
    },
    Sub {
        lhs: Box<IntExpr>,
        rhs: Box<IntExpr>,
    },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Predicate {
    True,
    False,
    LessThan { lhs: IntExpr, rhs: IntExpr },
    LessEqual { lhs: IntExpr, rhs: IntExpr },
    Equal { lhs: IntExpr, rhs: IntExpr },
    And { terms: Vec<Predicate> },
    Or { terms: Vec<Predicate> },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RetentionStateDecl {
    pub name: String,
    pub layers: Vec<u32>,
    pub bytes_per_token_per_layer: u64,
    pub may_read: Predicate,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RetentionProgramInput {
    pub schema: String,
    pub page_tokens: u64,
    pub states: Vec<RetentionStateDecl>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum InferredRetention {
    Unbounded,
    FixedWindow { window_tokens: u64 },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RetentionAnalysis {
    pub state_name: String,
    pub inferred: InferredRetention,
    pub proven_query_key_delta_upper_bound: Option<u64>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct AffineForm {
    query: i64,
    key: i64,
    constant: i64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct DeltaBounds {
    lower: Option<i64>,
    upper: Option<i64>,
    satisfiable: bool,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum RetentionError {
    #[error("retention IR schema must be orbitkv.retention-ir.v1, got {0:?}")]
    UnsupportedSchema(String),
    #[error("retention state {0:?} has no legal readers")]
    UnsatisfiableState(String),
    #[error("integer overflow while analyzing retention expression")]
    ArithmeticOverflow,
    #[error("fixed retention window does not fit u64")]
    WindowOutOfRange,
}

impl IntExpr {
    fn affine(&self) -> Result<AffineForm, RetentionError> {
        match self {
            Self::QueryPosition => Ok(AffineForm {
                query: 1,
                ..AffineForm::default()
            }),
            Self::KeyPosition => Ok(AffineForm {
                key: 1,
                ..AffineForm::default()
            }),
            Self::Constant { value } => Ok(AffineForm {
                constant: *value,
                ..AffineForm::default()
            }),
            Self::Add { lhs, rhs } => lhs.affine()?.checked_add(rhs.affine()?),
            Self::Sub { lhs, rhs } => lhs.affine()?.checked_sub(rhs.affine()?),
        }
    }

    #[must_use]
    pub fn evaluate(&self, query_position: i64, key_position: i64) -> Option<i64> {
        match self {
            Self::QueryPosition => Some(query_position),
            Self::KeyPosition => Some(key_position),
            Self::Constant { value } => Some(*value),
            Self::Add { lhs, rhs } => lhs
                .evaluate(query_position, key_position)?
                .checked_add(rhs.evaluate(query_position, key_position)?),
            Self::Sub { lhs, rhs } => lhs
                .evaluate(query_position, key_position)?
                .checked_sub(rhs.evaluate(query_position, key_position)?),
        }
    }
}

impl Predicate {
    /// Evaluates the declared relation inside the implicit autoregressive
    /// domain `0 <= key_position <= query_position`.
    #[must_use]
    pub fn may_read(&self, query_position: i64, key_position: i64) -> bool {
        if key_position < 0 || query_position < key_position {
            return false;
        }
        match self {
            Self::True => true,
            Self::False => false,
            Self::LessThan { lhs, rhs } => matches!(
                (
                    lhs.evaluate(query_position, key_position),
                    rhs.evaluate(query_position, key_position),
                ),
                (Some(lhs), Some(rhs)) if lhs < rhs
            ),
            Self::LessEqual { lhs, rhs } => matches!(
                (
                    lhs.evaluate(query_position, key_position),
                    rhs.evaluate(query_position, key_position),
                ),
                (Some(lhs), Some(rhs)) if lhs <= rhs
            ),
            Self::Equal { lhs, rhs } => matches!(
                (
                    lhs.evaluate(query_position, key_position),
                    rhs.evaluate(query_position, key_position),
                ),
                (Some(lhs), Some(rhs)) if lhs == rhs
            ),
            Self::And { terms } => terms
                .iter()
                .all(|term| term.may_read(query_position, key_position)),
            Self::Or { terms } => terms
                .iter()
                .any(|term| term.may_read(query_position, key_position)),
        }
    }
}

impl AffineForm {
    fn checked_add(self, rhs: Self) -> Result<Self, RetentionError> {
        Ok(Self {
            query: self
                .query
                .checked_add(rhs.query)
                .ok_or(RetentionError::ArithmeticOverflow)?,
            key: self
                .key
                .checked_add(rhs.key)
                .ok_or(RetentionError::ArithmeticOverflow)?,
            constant: self
                .constant
                .checked_add(rhs.constant)
                .ok_or(RetentionError::ArithmeticOverflow)?,
        })
    }

    fn checked_sub(self, rhs: Self) -> Result<Self, RetentionError> {
        Ok(Self {
            query: self
                .query
                .checked_sub(rhs.query)
                .ok_or(RetentionError::ArithmeticOverflow)?,
            key: self
                .key
                .checked_sub(rhs.key)
                .ok_or(RetentionError::ArithmeticOverflow)?,
            constant: self
                .constant
                .checked_sub(rhs.constant)
                .ok_or(RetentionError::ArithmeticOverflow)?,
        })
    }

    fn as_delta(self) -> Option<(i64, i64)> {
        if self.query == -self.key {
            Some((self.query, self.constant))
        } else {
            None
        }
    }
}

impl DeltaBounds {
    const fn autoregressive_domain() -> Self {
        Self {
            lower: Some(0),
            upper: None,
            satisfiable: true,
        }
    }

    const fn unsatisfiable() -> Self {
        Self {
            lower: None,
            upper: None,
            satisfiable: false,
        }
    }

    fn intersect(self, other: Self) -> Self {
        if !self.satisfiable || !other.satisfiable {
            return Self::unsatisfiable();
        }
        let lower = maximum_optional(self.lower, other.lower);
        let upper = minimum_optional(self.upper, other.upper);
        if lower.zip(upper).is_some_and(|(lower, upper)| lower > upper) {
            return Self::unsatisfiable();
        }
        Self {
            lower,
            upper,
            satisfiable: true,
        }
    }

    fn union(self, other: Self) -> Self {
        if !self.satisfiable {
            return other;
        }
        if !other.satisfiable {
            return self;
        }
        Self {
            lower: minimum_optional(self.lower, other.lower),
            upper: maximum_optional_with_infinity(self.upper, other.upper),
            satisfiable: true,
        }
    }
}

/// Analyzes one state relation and derives a safe exact fixed-window lifetime
/// for the supported difference-constraint domain.
///
/// Predicates that do not prove a finite `query_position - key_position`
/// upper bound safely lower to `Unbounded`.
///
/// # Errors
///
/// Returns an error for arithmetic overflow or an unsatisfiable state.
pub fn analyze_state(state: &RetentionStateDecl) -> Result<RetentionAnalysis, RetentionError> {
    let bounds = analyze_predicate(&state.may_read)?;
    if !bounds.satisfiable {
        return Err(RetentionError::UnsatisfiableState(state.name.clone()));
    }
    let Some(upper) = bounds.upper else {
        return Ok(RetentionAnalysis {
            state_name: state.name.clone(),
            inferred: InferredRetention::Unbounded,
            proven_query_key_delta_upper_bound: None,
        });
    };
    if upper < 0 {
        return Err(RetentionError::UnsatisfiableState(state.name.clone()));
    }
    let proven_query_key_delta_upper_bound =
        u64::try_from(upper).map_err(|_| RetentionError::WindowOutOfRange)?;
    let window_tokens = proven_query_key_delta_upper_bound
        .checked_add(1)
        .ok_or(RetentionError::WindowOutOfRange)?;
    Ok(RetentionAnalysis {
        state_name: state.name.clone(),
        inferred: InferredRetention::FixedWindow { window_tokens },
        proven_query_key_delta_upper_bound: Some(proven_query_key_delta_upper_bound),
    })
}

fn analyze_predicate(predicate: &Predicate) -> Result<DeltaBounds, RetentionError> {
    match predicate {
        Predicate::True => Ok(DeltaBounds::autoregressive_domain()),
        Predicate::False => Ok(DeltaBounds::unsatisfiable()),
        Predicate::LessThan { lhs, rhs } => analyze_comparison(lhs, rhs, true),
        Predicate::LessEqual { lhs, rhs } => analyze_comparison(lhs, rhs, false),
        Predicate::Equal { lhs, rhs } => {
            let forward = analyze_comparison(lhs, rhs, false)?;
            let reverse = analyze_comparison(rhs, lhs, false)?;
            Ok(forward.intersect(reverse))
        }
        Predicate::And { terms } => terms
            .iter()
            .try_fold(DeltaBounds::autoregressive_domain(), |bounds, term| {
                Ok(bounds.intersect(analyze_predicate(term)?))
            }),
        Predicate::Or { terms } => terms
            .iter()
            .try_fold(DeltaBounds::unsatisfiable(), |bounds, term| {
                Ok(bounds.union(analyze_predicate(term)?))
            }),
    }
}

fn analyze_comparison(
    lhs: &IntExpr,
    rhs: &IntExpr,
    strict: bool,
) -> Result<DeltaBounds, RetentionError> {
    let difference = lhs.affine()?.checked_sub(rhs.affine()?)?;
    let Some((coefficient, constant)) = difference.as_delta() else {
        return Ok(DeltaBounds::autoregressive_domain());
    };
    match coefficient {
        0 => {
            let valid = if strict { constant < 0 } else { constant <= 0 };
            Ok(if valid {
                DeltaBounds::autoregressive_domain()
            } else {
                DeltaBounds::unsatisfiable()
            })
        }
        1 => {
            let upper = constant
                .checked_neg()
                .and_then(|value| {
                    if strict {
                        value.checked_sub(1)
                    } else {
                        Some(value)
                    }
                })
                .ok_or(RetentionError::ArithmeticOverflow)?;
            Ok(DeltaBounds::autoregressive_domain().intersect(DeltaBounds {
                lower: None,
                upper: Some(upper),
                satisfiable: true,
            }))
        }
        -1 => {
            let lower = if strict {
                constant
                    .checked_add(1)
                    .ok_or(RetentionError::ArithmeticOverflow)?
            } else {
                constant
            };
            Ok(DeltaBounds::autoregressive_domain().intersect(DeltaBounds {
                lower: Some(lower),
                upper: None,
                satisfiable: true,
            }))
        }
        _ => Ok(DeltaBounds::autoregressive_domain()),
    }
}

const fn maximum_optional(lhs: Option<i64>, rhs: Option<i64>) -> Option<i64> {
    match (lhs, rhs) {
        (Some(lhs), Some(rhs)) => Some(if lhs > rhs { lhs } else { rhs }),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

const fn minimum_optional(lhs: Option<i64>, rhs: Option<i64>) -> Option<i64> {
    match (lhs, rhs) {
        (Some(lhs), Some(rhs)) => Some(if lhs < rhs { lhs } else { rhs }),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

const fn maximum_optional_with_infinity(lhs: Option<i64>, rhs: Option<i64>) -> Option<i64> {
    match (lhs, rhs) {
        (Some(lhs), Some(rhs)) => Some(if lhs > rhs { lhs } else { rhs }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn delta() -> IntExpr {
        IntExpr::Sub {
            lhs: Box::new(IntExpr::QueryPosition),
            rhs: Box::new(IntExpr::KeyPosition),
        }
    }

    fn state(predicate: Predicate) -> RetentionStateDecl {
        RetentionStateDecl {
            name: "state".into(),
            layers: vec![0],
            bytes_per_token_per_layer: 128,
            may_read: predicate,
        }
    }

    #[test]
    fn infers_fixed_window_from_difference_bound() {
        let analysis = analyze_state(&state(Predicate::LessThan {
            lhs: delta(),
            rhs: IntExpr::Constant { value: 32 },
        }))
        .unwrap();
        assert_eq!(
            analysis.inferred,
            InferredRetention::FixedWindow { window_tokens: 32 }
        );
        assert_eq!(analysis.proven_query_key_delta_upper_bound, Some(31));
    }

    #[test]
    fn unrecognized_affine_relation_fails_closed_to_unbounded() {
        let analysis = analyze_state(&state(Predicate::LessThan {
            lhs: IntExpr::Add {
                lhs: Box::new(IntExpr::QueryPosition),
                rhs: Box::new(IntExpr::KeyPosition),
            },
            rhs: IntExpr::Constant { value: 128 },
        }))
        .unwrap();
        assert_eq!(analysis.inferred, InferredRetention::Unbounded);
    }

    #[test]
    fn finite_or_takes_the_widest_branch() {
        let analysis = analyze_state(&state(Predicate::Or {
            terms: vec![
                Predicate::LessThan {
                    lhs: delta(),
                    rhs: IntExpr::Constant { value: 16 },
                },
                Predicate::LessEqual {
                    lhs: delta(),
                    rhs: IntExpr::Constant { value: 63 },
                },
            ],
        }))
        .unwrap();
        assert_eq!(
            analysis.inferred,
            InferredRetention::FixedWindow { window_tokens: 64 }
        );
    }

    #[test]
    fn unbounded_or_branch_fails_closed_to_unbounded() {
        let analysis = analyze_state(&state(Predicate::Or {
            terms: vec![
                Predicate::LessThan {
                    lhs: delta(),
                    rhs: IntExpr::Constant { value: 16 },
                },
                Predicate::True,
            ],
        }))
        .unwrap();
        assert_eq!(analysis.inferred, InferredRetention::Unbounded);
    }

    #[test]
    fn inferred_window_matches_exhaustive_relation() {
        for window in 1..=65_i64 {
            let declaration = state(Predicate::LessThan {
                lhs: delta(),
                rhs: IntExpr::Constant { value: window },
            });
            let analysis = analyze_state(&declaration).unwrap();
            let maximum_delta = (0..=window * 3)
                .flat_map(|query| (0..=query).map(move |key| (query, key)))
                .filter(|&(query, key)| declaration.may_read.may_read(query, key))
                .map(|(query, key)| query - key)
                .max()
                .unwrap();
            assert_eq!(
                analysis.proven_query_key_delta_upper_bound,
                Some(u64::try_from(maximum_delta).unwrap())
            );
            assert_eq!(
                analysis.inferred,
                InferredRetention::FixedWindow {
                    window_tokens: u64::try_from(window).unwrap()
                }
            );
        }
    }
}
