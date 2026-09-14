use super::*;

#[test]
fn public_events_contain_no_physical_page_identity() {
    let event = EngineEvent::Token(TokenOutput {
        request_id: RequestId(4),
        token_id: 99,
    });
    assert_eq!(
        event,
        EngineEvent::Token(TokenOutput {
            request_id: RequestId(4),
            token_id: 99,
        })
    );
}
