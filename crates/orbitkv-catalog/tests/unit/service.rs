use super::*;
use std::time::Duration;

fn service() -> CatalogService {
    let owner = CacheOwner {
        endpoint: "127.0.0.1:12345".into(),
        incarnation: Uuid::new_v4(),
    };
    let view = Arc::new(MembershipView::new(
        owner.clone(),
        crate::Placement::new(vec!["catalog".into()]).unwrap(),
    ));
    assert!(view.renew(Instant::now(), Duration::from_secs(60)));
    view.replace_members([("catalog".into(), owner)]);
    CatalogService::new(
        std::array::from_fn(|_| Arc::new(BlockHashStore::new())),
        view,
    )
}

#[tokio::test]
async fn batched_lookup_validates_every_route_and_the_aggregate_budget() {
    let service = service();
    let hashes: Vec<_> = (0_u32..128).map(|n| n.to_be_bytes().to_vec()).collect();
    let request = LocateBlocksRequest {
        routes: (0..CATALOG_SHARDS)
            .map(|shard| CatalogRoute {
                shard: shard as u32,
                placement_id: service.membership.placement_id().into(),
                incarnation: service.membership.owner().incarnation.to_string(),
            })
            .collect(),
        namespace: "ns".into(),
        block_hashes: hashes.clone(),
        exclude_node: String::new(),
    };
    let rows = service
        .locate_blocks(Request::new(request.clone()))
        .await
        .unwrap()
        .into_inner()
        .blocks;
    assert_eq!(rows.len(), hashes.len());
    assert!(
        rows.iter()
            .zip(&hashes)
            .all(|(r, hash)| r.block_hash == *hash && r.replicas.is_empty())
    );
    for case in [
        "missing",
        "duplicate",
        "stale",
        "wrong-placement",
        "key-count",
        "bytes",
    ] {
        let mut bad = request.clone();
        let expected = match case {
            "missing" => {
                bad.routes.clear();
                tonic::Code::InvalidArgument
            }
            "duplicate" => {
                bad.routes[1] = bad.routes[0].clone();
                tonic::Code::InvalidArgument
            }
            "stale" => {
                bad.routes[15].incarnation = Uuid::new_v4().to_string();
                tonic::Code::FailedPrecondition
            }
            "wrong-placement" => {
                bad.routes[15].placement_id = "old".into();
                tonic::Code::FailedPrecondition
            }
            "key-count" => {
                bad.block_hashes.push(vec![255]);
                tonic::Code::InvalidArgument
            }
            "bytes" => {
                bad.block_hashes = vec![vec![1; 40000], vec![2; 40000]];
                tonic::Code::InvalidArgument
            }
            _ => unreachable!(),
        };
        assert_eq!(
            service
                .locate_blocks(Request::new(bad))
                .await
                .unwrap_err()
                .code(),
            expected,
            "{case}"
        );
    }
    let mut missing = request;
    missing.routes.retain(|r| {
        r.shard as usize != catalog_shard(&StateKey::new("ns".into(), hashes[0].clone()))
    });
    assert_eq!(
        service
            .locate_blocks(Request::new(missing))
            .await
            .unwrap_err()
            .code(),
        tonic::Code::InvalidArgument
    );
}
