use super::*;

fn row(device: u64, host: usize, size: usize) -> CopyDesc {
    CopyDesc {
        device,
        host: host as *mut u8,
        host_device: host as u64,
        size,
        device_allocation: 1,
        host_registration: 2,
    }
}

fn expanded(copies: &[CopyDesc], max_pitch: usize) -> Vec<(u64, usize)> {
    dma_copies(copies, max_pitch)
        .flat_map(|copy| {
            (0..copy.rows).flat_map(move |row| {
                (0..copy.width).map(move |byte| {
                    (
                        copy.device + (row * copy.device_pitch + byte) as u64,
                        copy.host as usize + row * copy.host_pitch + byte,
                    )
                })
            })
        })
        .collect()
}

#[test]
fn explicit_strided_rows_preserve_gaps_and_submission_order() {
    let copies = [row(100, 200, 4), row(108, 212, 4), row(116, 224, 4)];
    let compiled: Vec<_> = dma_copies(&copies, 12).collect();
    assert_eq!(compiled.len(), 1);
    let copy = &compiled[0];
    assert_eq!(
        (copy.width, copy.rows, copy.device_pitch, copy.host_pitch),
        (4, 3, 8, 12)
    );
    assert_eq!(dma_copies(&copies, 11).count(), 3);
    let mut missing = copies.to_vec();
    missing.remove(1);
    let compiled: Vec<_> = dma_copies(&missing, 24).collect();
    assert_eq!((compiled[0].rows, compiled[0].device_pitch), (2, 16));
    assert_eq!(expanded(&missing, 24).len(), 8);
}

#[test]
fn incompatible_rows_cannot_share_a_submission() {
    let first = row(100, 200, 4);
    for second in [
        row(108, 212, 3),
        row(102, 212, 4),
        row(108, 202, 4),
        row(96, 212, 4),
        row(108, 196, 4),
        CopyDesc {
            device_allocation: 9,
            ..row(108, 212, 4)
        },
        CopyDesc {
            host_registration: 9,
            ..row(108, 212, 4)
        },
    ] {
        assert_eq!(dma_copies(&[first, second], 100).count(), 2);
    }
    assert_eq!(dma_copies(&[], 100).count(), 0);
    let uneven = [first, row(108, 212, 4), row(124, 224, 4)];
    assert_eq!(
        dma_copies(&uneven, 100).map(|c| c.rows).collect::<Vec<_>>(),
        vec![2, 1]
    );
}

#[test]
fn contiguous_rows_coalesce_without_a_pitch_limit() {
    let copies = [row(100, 200, 4), row(104, 204, 6), row(110, 210, 3)];
    let compiled: Vec<_> = dma_copies(&copies, 0).collect();
    assert_eq!(compiled.len(), 1);
    assert_eq!((compiled[0].width, compiled[0].rows), (13, 1));
}

#[test]
fn overflowing_addresses_are_never_merged() {
    for copies in [
        [row(u64::MAX - 3, 200, 4), row(0, 204, 4)],
        [row(100, usize::MAX - 3, 4), row(104, 0, 4)],
        [row(u64::MAX - 11, 200, 4), row(u64::MAX - 3, 212, 4)],
        [row(100, usize::MAX - 15, 4), row(108, usize::MAX - 3, 4)],
    ] {
        assert_eq!(dma_copies(&copies, usize::MAX).count(), 2);
    }
}

#[test]
fn compilation_preserves_every_requested_byte_for_irregular_layouts() {
    let mut seed = 0x71e9_9aad_u64;
    for _ in 0..2000 {
        let mut random = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut copies = Vec::new();
        let mut device = 1000;
        let mut host = 2000;
        for _ in 0..64 {
            let size = 1 + (random() % 16) as usize;
            let mut copy = row(device, host, size);
            copy.device_allocation = (random() % 2) as usize;
            copy.host_registration = (random() % 2) as usize;
            copies.push(copy);
            device += random() % 24;
            host += (random() % 24) as usize;
        }
        let expected: Vec<_> = copies
            .iter()
            .flat_map(|copy| {
                (0..copy.size)
                    .map(move |byte| (copy.device + byte as u64, copy.host as usize + byte))
            })
            .collect();
        assert_eq!(expanded(&copies, 16), expected);
    }
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn strided_dma_preserves_unrequested_bytes_both_directions() {
    use crate::memory::numa::NumaNode;
    use crate::memory::pinned::{PagePolicy, PinnedMemory};
    use cudarc::driver::DevicePtr;

    const ROWS: usize = 33;
    const WIDTH: usize = 129;
    const HOST_PITCH: usize = 193;
    const DEVICE_PITCH: usize = 257;
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let backend = MemcpyBackend::new(&context).unwrap();
    let memory =
        PinnedMemory::allocate(ROWS * HOST_PITCH, PagePolicy::Regular, NumaNode::UNKNOWN).unwrap();
    let target = stream.alloc_zeros::<u8>(ROWS * DEVICE_PITCH).unwrap();
    let device = target.device_ptr(&stream).0;
    stream.synchronize().unwrap();
    // SAFETY: this test exclusively owns the pinned region and synchronizes
    // every transfer before accessing it or freeing either allocation.
    let host =
        unsafe { std::slice::from_raw_parts_mut(memory.as_ptr().cast_mut(), ROWS * HOST_PITCH) };
    let copies: Vec<_> = (0..ROWS)
        .map(|index| CopyDesc {
            device: device + (index * DEVICE_PITCH) as u64,
            host: host.as_mut_ptr().wrapping_add(index * HOST_PITCH),
            host_device: memory.device_ptr().as_ptr() as u64 + (index * HOST_PITCH) as u64,
            size: WIDTH,
            device_allocation: 1,
            host_registration: memory.as_ptr() as usize,
        })
        .collect();
    assert_eq!(dma_copies(&copies, backend.max_pitch).count(), 1);
    for missing_row in [None, Some(13)] {
        let selected: Vec<_> = copies
            .iter()
            .enumerate()
            .filter(|(index, _)| Some(*index) != missing_row)
            .map(|(_, copy)| *copy)
            .collect();
        host.fill(0x7b);
        let mut expected_device = vec![0xa6; ROWS * DEVICE_PITCH];
        // SAFETY: the target is owned above and no transfer is in flight.
        unsafe { sys::cuMemsetD8_v2(device, 0xa6, expected_device.len()).result() }.unwrap();
        for index in 0..ROWS {
            if Some(index) == missing_row {
                continue;
            }
            for byte in 0..WIDTH {
                let value = ((index * 71 + byte * 13) % 251) as u8;
                host[index * HOST_PITCH + byte] = value;
                expected_device[index * DEVICE_PITCH + byte] = value;
            }
        }
        backend.h2d(&selected, &stream).unwrap();
        stream.synchronize().unwrap();
        let actual = stream.clone_dtoh(&target).unwrap();
        assert_eq!(
            actual, expected_device,
            "H2D touched a gap or copied stale rows"
        );
        let mut expected_host = vec![0x3d; host.len()];
        for index in 0..ROWS {
            if Some(index) != missing_row {
                expected_host[index * HOST_PITCH..index * HOST_PITCH + WIDTH]
                    .copy_from_slice(&host[index * HOST_PITCH..index * HOST_PITCH + WIDTH]);
            }
        }
        host.fill(0x3d);
        backend.d2h(&selected, &stream).unwrap();
        stream.synchronize().unwrap();
        assert_eq!(
            host, expected_host,
            "D2H touched a gap or copied stale rows"
        );
    }
}
