use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::{
    Json, Router,
    routing::{get, post},
};
use log::{info, warn};
use prometheus::{Registry, TextEncoder};
use serde::Serialize;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::Notify;

use crate::store::BlockHashStore;

#[derive(Clone)]
struct AppState {
    prometheus_registry: Registry,
    store: Arc<BlockHashStore>,
}

#[derive(Debug, Serialize)]
struct CleanupResponse {
    removed_owners: usize,
    removed_keys: usize,
}

async fn health_handler() -> &'static str {
    "ok"
}

async fn metrics_handler(State(state): State<AppState>) -> impl IntoResponse {
    let encoder = TextEncoder::new();
    let metric_families = state.prometheus_registry.gather();
    (
        StatusCode::OK,
        encoder
            .encode_to_string(&metric_families)
            .unwrap_or_else(|e| format!("# Error encoding metrics: {e}")),
    )
}

async fn sweep_expired_nodes_handler(
    State(state): State<AppState>,
) -> Result<Json<CleanupResponse>, StatusCode> {
    let store = Arc::clone(&state.store);
    let stats = match tokio::task::spawn_blocking(move || store.sweep_expired()).await {
        Ok(stats) => stats,
        Err(err) => {
            warn!("manual cleanup worker failed: {err}");
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };
    Ok(Json(CleanupResponse {
        removed_owners: stats.removed_owners,
        removed_keys: stats.removed_keys,
    }))
}

fn public_app(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .route(
            "/admin/sweep-expired-nodes",
            post(sweep_expired_nodes_handler),
        )
        .with_state(state)
}

pub async fn start_http_server(
    addr: std::net::SocketAddr,
    prometheus_registry: Registry,
    store: Arc<BlockHashStore>,
    shutdown: Arc<Notify>,
) -> Result<tokio::task::JoinHandle<()>, std::io::Error> {
    let listener = TcpListener::bind(addr).await?;

    let state = AppState {
        prometheus_registry,
        store,
    };

    info!(
        "Starting HTTP server on {} (/health, /metrics, /admin/sweep-expired-nodes)",
        addr
    );

    let handle = tokio::spawn(async move {
        let result = axum::serve(listener, public_app(state))
            .with_graceful_shutdown(async move {
                shutdown.notified().await;
            })
            .await;
        if let Err(err) = result {
            warn!("HTTP server stopped with error: {err}");
        }
    });

    Ok(handle)
}

#[cfg(test)]
mod tests {
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
}
