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
    FloorDiv {
        value: Box<IntExpr>,
        divisor: i64,
    },
    Mod {
        value: Box<IntExpr>,
        modulus: i64,
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
pub struct KvHeadRange {
    pub start: u32,
    pub end_exclusive: u32,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RetentionStateDecl {
    pub name: String,
    pub layers: Vec<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_head_range: Option<KvHeadRange>,
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
    Chunked { chunk_tokens: u64 },
    Partitioned { regions: Vec<InferredRegion> },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AtomicRetention {
    Unbounded,
    FixedWindow { window_tokens: u64 },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct InferredRegion {
    pub label: String,
    pub start_token: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_token_exclusive: Option<u64>,
    pub retention: AtomicRetention,
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
    non_affine: bool,
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
    #[error("floor division requires a positive divisor")]
    InvalidFloorDivisor,
    #[error("modulo requires a positive modulus")]
    InvalidModulus,
    #[error("chunk size does not fit u64")]
    ChunkOutOfRange,
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
            Self::FloorDiv { value, divisor } => {
                if *divisor <= 0 {
                    return Err(RetentionError::InvalidFloorDivisor);
                }
                value.affine()?;
                Ok(AffineForm {
                    non_affine: true,
                    ..AffineForm::default()
                })
            }
            Self::Mod { value, modulus } => {
                if *modulus <= 0 {
                    return Err(RetentionError::InvalidModulus);
                }
                value.affine()?;
                Ok(AffineForm {
                    non_affine: true,
                    ..AffineForm::default()
                })
            }
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
            Self::FloorDiv { value, divisor } => {
                if *divisor <= 0 {
                    return None;
                }
                Some(
                    value
                        .evaluate(query_position, key_position)?
                        .div_euclid(*divisor),
                )
            }
            Self::Mod { value, modulus } => {
                if *modulus <= 0 {
                    return None;
                }
                Some(
                    value
                        .evaluate(query_position, key_position)?
                        .rem_euclid(*modulus),
                )
            }
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
            non_affine: self.non_affine || rhs.non_affine,
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
            non_affine: self.non_affine || rhs.non_affine,
        })
    }

    fn as_delta(self) -> Option<(i64, i64)> {
        if !self.non_affine && self.query == -self.key {
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
    if let Some(chunk_tokens) = analyze_same_chunk(&state.may_read)? {
        return Ok(RetentionAnalysis {
            state_name: state.name.clone(),
            inferred: InferredRetention::Chunked { chunk_tokens },
            proven_query_key_delta_upper_bound: Some(chunk_tokens - 1),
        });
    }
    if let Some(maximum_delta) = analyze_dilated_window(&state.may_read)? {
        let window_tokens = maximum_delta
            .checked_add(1)
            .ok_or(RetentionError::WindowOutOfRange)?;
        return Ok(RetentionAnalysis {
            state_name: state.name.clone(),
            inferred: InferredRetention::FixedWindow { window_tokens },
            proven_query_key_delta_upper_bound: Some(maximum_delta),
        });
    }
    if let Some((sink_tokens, window_tokens)) = analyze_sink_and_window(&state.may_read)? {
        return Ok(RetentionAnalysis {
            state_name: state.name.clone(),
            inferred: InferredRetention::Partitioned {
                regions: vec![
                    InferredRegion {
                        label: "sink".into(),
                        start_token: 0,
                        end_token_exclusive: Some(sink_tokens),
                        retention: AtomicRetention::Unbounded,
                    },
                    InferredRegion {
                        label: "local".into(),
                        start_token: sink_tokens,
                        end_token_exclusive: None,
                        retention: AtomicRetention::FixedWindow { window_tokens },
                    },
                ],
            },
            proven_query_key_delta_upper_bound: None,
        });
    }
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

fn analyze_dilated_window(predicate: &Predicate) -> Result<Option<u64>, RetentionError> {
    let Predicate::And { terms } = predicate else {
        return Ok(None);
    };
    let mut modulus = None;
    for term in terms {
        if let Some(term_modulus) = zero_delta_modulus(term)?
            && modulus.replace(term_modulus).is_some()
        {
            return Ok(None);
        }
    }
    let Some(modulus) = modulus else {
        return Ok(None);
    };
    let bounds = analyze_predicate(predicate)?;
    if !bounds.satisfiable {
        return Ok(None);
    }
    let Some(upper) = bounds.upper else {
        return Ok(None);
    };
    if upper < 0 {
        return Ok(None);
    }
    let maximum_delta = upper - upper.rem_euclid(modulus);
    if bounds.lower.is_some_and(|lower| maximum_delta < lower) {
        return Ok(None);
    }
    u64::try_from(maximum_delta)
        .map(Some)
        .map_err(|_| RetentionError::WindowOutOfRange)
}

fn zero_delta_modulus(predicate: &Predicate) -> Result<Option<i64>, RetentionError> {
    let Predicate::Equal { lhs, rhs } = predicate else {
        return Ok(None);
    };
    if is_zero_constant(rhs) {
        return delta_modulus(lhs);
    }
    if is_zero_constant(lhs) {
        return delta_modulus(rhs);
    }
    Ok(None)
}

fn delta_modulus(expression: &IntExpr) -> Result<Option<i64>, RetentionError> {
    let IntExpr::Mod { value, modulus } = expression else {
        return Ok(None);
    };
    if *modulus <= 0 {
        return Err(RetentionError::InvalidModulus);
    }
    let Some((coefficient, constant)) = value.affine()?.as_delta() else {
        return Ok(None);
    };
    Ok((coefficient == 1 && constant == 0).then_some(*modulus))
}

const fn is_zero_constant(expression: &IntExpr) -> bool {
    matches!(expression, IntExpr::Constant { value: 0 })
}

fn analyze_same_chunk(predicate: &Predicate) -> Result<Option<u64>, RetentionError> {
    let Predicate::Equal { lhs, rhs } = predicate else {
        return Ok(None);
    };
    let lhs_divisor = chunk_divisor(lhs, &IntExpr::QueryPosition)?;
    let rhs_divisor = chunk_divisor(rhs, &IntExpr::KeyPosition)?;
    match (lhs_divisor, rhs_divisor) {
        (Some(lhs), Some(rhs)) if lhs == rhs => u64::try_from(lhs)
            .map(Some)
            .map_err(|_| RetentionError::ChunkOutOfRange),
        (Some(_), Some(_)) => Ok(None),
        _ => {
            let lhs_divisor = chunk_divisor(lhs, &IntExpr::KeyPosition)?;
            let rhs_divisor = chunk_divisor(rhs, &IntExpr::QueryPosition)?;
            match (lhs_divisor, rhs_divisor) {
                (Some(lhs), Some(rhs)) if lhs == rhs => u64::try_from(lhs)
                    .map(Some)
                    .map_err(|_| RetentionError::ChunkOutOfRange),
                _ => Ok(None),
            }
        }
    }
}

fn chunk_divisor(expression: &IntExpr, expected: &IntExpr) -> Result<Option<i64>, RetentionError> {
    let IntExpr::FloorDiv { value, divisor } = expression else {
        return Ok(None);
    };
    if *divisor <= 0 {
        return Err(RetentionError::InvalidFloorDivisor);
    }
    Ok((value.as_ref() == expected).then_some(*divisor))
}

fn analyze_sink_and_window(predicate: &Predicate) -> Result<Option<(u64, u64)>, RetentionError> {
    let Predicate::Or { terms } = predicate else {
        return Ok(None);
    };
    let mut sink_tokens = None;
    let mut window_tokens = None;
    for term in terms {
        if let Some(tokens) = key_prefix_tokens(term)? {
            sink_tokens = Some(sink_tokens.map_or(tokens, |current: u64| current.max(tokens)));
            continue;
        }
        let bounds = analyze_predicate(term)?;
        let Some(upper) = bounds.upper else {
            return Ok(None);
        };
        if upper < 0 {
            continue;
        }
        let delta = u64::try_from(upper).map_err(|_| RetentionError::WindowOutOfRange)?;
        let window = delta
            .checked_add(1)
            .ok_or(RetentionError::WindowOutOfRange)?;
        window_tokens = Some(window_tokens.map_or(window, |current: u64| current.max(window)));
    }
    Ok(sink_tokens.zip(window_tokens))
}

fn key_prefix_tokens(predicate: &Predicate) -> Result<Option<u64>, RetentionError> {
    let (lhs, rhs, strict) = match predicate {
        Predicate::LessThan { lhs, rhs } => (lhs, rhs, true),
        Predicate::LessEqual { lhs, rhs } => (lhs, rhs, false),
        _ => return Ok(None),
    };
    let difference = lhs.affine()?.checked_sub(rhs.affine()?)?;
    if difference.non_affine || difference.query != 0 || difference.key != 1 {
        return Ok(None);
    }
    let upper = difference
        .constant
        .checked_neg()
        .and_then(|value| {
            if strict {
                value.checked_sub(1)
            } else {
                Some(value)
            }
        })
        .ok_or(RetentionError::ArithmeticOverflow)?;
    if upper < 0 {
        return Ok(None);
    }
    let upper = u64::try_from(upper).map_err(|_| RetentionError::WindowOutOfRange)?;
    Ok(Some(
        upper
            .checked_add(1)
            .ok_or(RetentionError::WindowOutOfRange)?,
    ))
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
#[path = "../tests/unit/retention/mod.rs"]
mod tests;
