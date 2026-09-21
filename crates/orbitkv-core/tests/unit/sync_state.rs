use super::*;
use std::sync::atomic::Ordering;

#[test]
fn create_and_attach_round_trip() {
    let load_state = LoadState::new().expect("create LoadState");
    assert_eq!(load_state.get(), LOAD_STATE_PENDING);

    let shm_name = load_state.shm_name().to_string();
    load_state.set_completed();

    let attached = LoadState::attach(&shm_name).expect("attach LoadState");
    assert_eq!(attached.get(), LOAD_STATE_SUCCESS);
}

#[test]
fn attach_rejects_too_small_mapping() {
    let shm_name = format!("orbitkv_test_small_{}", Uuid::new_v4().as_simple());
    let _mapping = ShmemConf::new().os_id(&shm_name).size(1).create().unwrap();

    let err = LoadState::attach(&shm_name).expect_err("should fail to attach too small mapping");
    assert!(matches!(
        err,
        LoadStateError::MappingTooSmall {
            actual: _,
            required: _
        }
    ));
}

#[test]
fn attach_rejects_invalid_header() {
    let load_state = LoadState::new().expect("create LoadState");
    let shm_name = load_state.shm_name().to_string();

    // Corrupt the header magic to force a validation failure.
    unsafe {
        let mem = load_state.ptr.as_ref();
        mem.header.magic.store(0xDEADBEEF, Ordering::Release);
    }

    let err = LoadState::attach(&shm_name).unwrap_err();
    assert!(matches!(err, LoadStateError::InvalidHeader { .. }));
}
