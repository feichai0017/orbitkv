use super::*;
use axum::body::Body;
use axum::http::Request;
use tower::ServiceExt;

#[tokio::test]
async fn cleanup_route_is_available_on_public_listener() {
    let response = public_app(AppState {
        prometheus_registry: Registry::new(),
        store: Arc::new(BlockHashStore::new()),
    })
    .oneshot(
        Request::builder()
            .method("POST")
            .uri("/admin/sweep-expired-nodes")
            .body(Body::empty())
            .unwrap(),
    )
    .await
    .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}
