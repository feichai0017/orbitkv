use super::*;

#[test]
fn scopes_discard_unowned_notifications_and_fence_reused_names() {
    let mut mailbox = NotificationMailbox::default();
    mailbox.record(vec![Notification {
        name: "request".to_string(),
        message: "done".to_string(),
    }]);
    let first = mailbox.open("request");
    assert!(matches!(
        mailbox.status("request", first, &[("done".to_string(), 1)]),
        NotificationMatch::Pending
    ));

    mailbox.record(vec![Notification {
        name: "request".to_string(),
        message: "done".to_string(),
    }]);
    assert!(matches!(
        mailbox.status("request", first, &[("done".to_string(), 1)]),
        NotificationMatch::Matched(message) if message == "done"
    ));

    let second = mailbox.open("request");
    assert_ne!(first, second);
    assert!(matches!(
        mailbox.status("request", first, &[("done".to_string(), 1)]),
        NotificationMatch::Closed
    ));
    assert!(matches!(
        mailbox.status("request", second, &[("done".to_string(), 1)]),
        NotificationMatch::Pending
    ));
    mailbox.close("request", first);
    assert!(matches!(
        mailbox.status("request", second, &[("done".to_string(), 1)]),
        NotificationMatch::Pending
    ));
    mailbox.close("request", second);
    assert!(matches!(
        mailbox.status("request", second, &[("done".to_string(), 1)]),
        NotificationMatch::Closed
    ));
}

#[test]
fn status_requires_the_requested_count_and_preserves_priority() {
    let mut mailbox = NotificationMailbox::default();
    let generation = mailbox.open("request");
    mailbox.record(vec![
        Notification {
            name: "request".to_string(),
            message: "done".to_string(),
        },
        Notification {
            name: "other".to_string(),
            message: "failed".to_string(),
        },
    ]);
    let expectations = &[("failed".to_string(), 1), ("done".to_string(), 2)];
    assert!(matches!(
        mailbox.status("request", generation, expectations),
        NotificationMatch::Pending
    ));

    mailbox.record(vec![
        Notification {
            name: "request".to_string(),
            message: "done".to_string(),
        },
        Notification {
            name: "request".to_string(),
            message: "failed".to_string(),
        },
    ]);
    assert!(matches!(
        mailbox.status("request", generation, expectations),
        NotificationMatch::Matched(message) if message == "failed"
    ));
}
