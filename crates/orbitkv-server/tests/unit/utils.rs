use super::*;

#[test]
fn test_parse_memory_size_basic() {
    assert_eq!(parse_memory_size("1024").unwrap(), 1024);
    assert_eq!(parse_memory_size("1gb").unwrap(), 1024 * 1024 * 1024);
    assert_eq!(
        parse_memory_size("1.5gb").unwrap(),
        (1.5 * 1024.0 * 1024.0 * 1024.0) as usize
    );
}

#[test]
fn test_parse_memory_size_invalid() {
    assert!(parse_memory_size("").is_err());
    assert!(parse_memory_size("abc").is_err());
    assert!(parse_memory_size("-10gb").is_err());
}
