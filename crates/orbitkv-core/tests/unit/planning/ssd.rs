use super::*;

#[test]
fn automatic_cufile_allows_fallback_but_explicit_routes_do_not() {
    assert_eq!(
        deferred_route(None, true),
        Some((SsdReadPath::Cufile, true))
    );
    assert_eq!(deferred_route(None, false), None);
    for path in [SsdReadPath::Uring, SsdReadPath::Cufile] {
        assert_eq!(deferred_route(Some(path), false), Some((path, false)));
        assert_eq!(deferred_route(Some(path), true), Some((path, false)));
    }
}
