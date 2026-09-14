use super::*;

#[test]
fn source_limit_is_scoped_and_nestable() {
    let initial = kernel_source_limit();
    with_kernel_source_limit(Some(64), || {
        assert_eq!(kernel_source_limit(), Some(64));
        with_kernel_source_limit(None, || assert_eq!(kernel_source_limit(), None));
        assert_eq!(kernel_source_limit(), Some(64));
    });
    assert_eq!(kernel_source_limit(), initial);
}
