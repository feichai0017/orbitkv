use super::*;
use crate::cost::{CostPath, ExecutionResource, Representation};

fn key(path: CostPath) -> CostKey {
    CostKey::new(
        path,
        ExecutionResource::Gpu(1),
        Representation::Raw,
        4096,
        4,
    )
    .with_dma_ranges(2)
}

fn estimate(seconds: f64, error: f64) -> Estimate {
    Estimate {
        count: 10,
        seconds,
        absolute_error: error,
        updated: Instant::now(),
    }
}

#[test]
fn shadow_requires_matching_demand_resources_and_completion_target() {
    let current = key(CostPath::GpuLoadDirect);
    let alternative = key(CostPath::GpuLoadKernel);
    let predictions = [Some(estimate(0.02, 0.0)), Some(estimate(0.01, 0.0))];
    assert_eq!(
        recommendation(&[current, alternative], &predictions, 0),
        "different"
    );

    for (name, incompatible) in [
        ("direction", alternative.with_path(CostPath::GpuSaveKernel)),
        (
            "completion",
            alternative.with_path(CostPath::SsdUringRestore),
        ),
        ("composite", alternative.with_path(CostPath::GpuDecode)),
        (
            "device",
            alternative.with_path_resource(alternative.path, ExecutionResource::Gpu(2)),
        ),
        (
            "owner domain",
            alternative.with_path_resource(alternative.path, ExecutionResource::SsdStore(1)),
        ),
        (
            "representation",
            CostKey {
                representation: Representation::Ans,
                ..alternative
            },
        ),
        (
            "bytes",
            CostKey {
                size: alternative.size + 1,
                ..alternative
            },
        ),
        (
            "fragments",
            CostKey {
                fragments: alternative.fragments + 1,
                ..alternative
            },
        ),
        ("DMA ranges", alternative.with_dma_ranges(8)),
        ("SSD source", alternative.with_ssd_shape(8192, 4, 4096, 4)),
    ] {
        assert_eq!(
            recommendation(&[current, incompatible], &predictions, 0),
            "incomparable",
            "{name}",
        );
    }

    let unknown = CostKey {
        representation: Representation::Unknown,
        ..current
    };
    assert_eq!(
        recommendation(
            &[unknown, unknown.with_path(CostPath::GpuLoadKernel)],
            &predictions,
            0
        ),
        "incomparable",
    );

    let ssd = current
        .with_path_resource(
            CostPath::SsdUringRestore,
            ExecutionResource::SsdRestore {
                device: 1,
                copy_backend: 0,
                stores: 123,
                has_memory: true,
            },
        )
        .with_ssd_shape(8192, 8, 4096, 4);
    assert_eq!(
        recommendation(
            &[ssd, ssd.with_path(CostPath::SsdCufileRestore)],
            &predictions,
            0
        ),
        "different",
    );

    let other_device = ssd.with_path_resource(
        CostPath::SsdCufileRestore,
        ExecutionResource::SsdRestore {
            device: 2,
            copy_backend: 0,
            stores: 123,
            has_memory: true,
        },
    );
    assert_eq!(
        recommendation(&[ssd, other_device], &predictions, 0),
        "incomparable",
        "routes for different destination devices cannot share EngineRestore evidence",
    );
}

#[test]
fn shadow_requires_a_gain_beyond_both_errors_and_the_switching_margin() {
    let candidates = [key(CostPath::GpuLoadDirect), key(CostPath::GpuLoadKernel)];
    for (name, current, alternative, expected) in [
        (
            "clear gain",
            Some(estimate(0.02, 0.001)),
            Some(estimate(0.01, 0.001)),
            "different",
        ),
        (
            "noise overlaps",
            Some(estimate(0.02, 0.006)),
            Some(estimate(0.01, 0.005)),
            "within_margin",
        ),
        (
            "gain below margin",
            Some(estimate(0.02, 0.0)),
            Some(estimate(0.0195, 0.0)),
            "within_margin",
        ),
        (
            "current is faster",
            Some(estimate(0.01, 0.0)),
            Some(estimate(0.02, 0.0)),
            "agree",
        ),
        (
            "equal cost",
            Some(estimate(0.01, 0.0)),
            Some(estimate(0.01, 0.0)),
            "agree",
        ),
        (
            "unknown alternative",
            Some(estimate(0.01, 0.0)),
            None,
            "unknown",
        ),
        (
            "unknown current",
            None,
            Some(estimate(0.01, 0.0)),
            "unknown",
        ),
    ] {
        assert_eq!(
            recommendation(&candidates, &[current, alternative], 0),
            expected,
            "{name}"
        );
    }
    assert_eq!(
        recommendation(&candidates[..1], &[Some(estimate(0.01, 0.0))], 0),
        "unknown"
    );
    assert_eq!(recommendation(&candidates, &[], 0), "unknown");
    assert_eq!(recommendation(&[], &[], 0), "unknown");
    assert_eq!(recommendation(&candidates, &[None, None], 2), "unknown");
}
