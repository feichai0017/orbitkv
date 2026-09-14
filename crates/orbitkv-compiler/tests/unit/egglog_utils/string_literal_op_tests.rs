use super::decode_string_literal_op;
use crate::shape::symbol_from_egglog_name;

#[test]
fn decodes_both_spellings_whole() {
    assert_eq!(
        decode_string_literal_op(r#"Boxed("s")"#).as_deref(),
        Some("s")
    );
    assert_eq!(decode_string_literal_op(r#""s""#).as_deref(), Some("s"));
    // Whole name, not a prefix.
    assert_eq!(
        decode_string_literal_op(r#"Boxed("s77")"#).as_deref(),
        Some("s77")
    );
    assert_eq!(decode_string_literal_op(r#""s77""#).as_deref(), Some("s77"));
}

#[test]
fn stripping_is_anchored_not_substring() {
    // A name whose content embeds the delimiters must survive.
    assert_eq!(
        decode_string_literal_op(r#"Boxed("a")b")"#).as_deref(),
        Some(r#"a")b"#)
    );
}

#[test]
fn non_string_ops_decode_to_none() {
    assert_eq!(decode_string_literal_op("5"), None);
    assert_eq!(decode_string_literal_op("MAdd"), None);
}

/// The reserved index round-trips: it appears in serialized strides.
#[test]
fn reserved_index_survives_the_boundary() {
    assert_eq!(symbol_from_egglog_name("z").to_string(), "z");
}

/// The bug this whole change exists to remove. Extraction used to take
/// `name.chars().next()`, so a dim named `s77` came back as `s` — a
/// *different, possibly live* variable, silently. The bad name then missed
/// in `dyn_map`, and the CUDA backend turned that miss into a dimension of 0.
#[test]
fn multichar_dim_names_survive_extraction() {
    assert_eq!(symbol_from_egglog_name("s77").to_string(), "s77");
    assert_ne!(symbol_from_egglog_name("s77"), symbol_from_egglog_name("s"));
}

/// Names that cannot be a C identifier or an egglog literal are still
/// rejected loudly rather than mangled into something that collides.
#[test]
#[should_panic(expected = "bad dim name out of egglog")]
fn malformed_dim_name_from_egglog_is_loud() {
    symbol_from_egglog_name("s-77");
}
