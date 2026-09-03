use std::future::Future;
use std::pin::Pin;

use futures_core::Stream;

use crate::{BatchIntent, RequestId};

/// Why generation stopped for one request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FinishReason {
    Stop,
    Length,
    Cancelled,
}

/// One sampled token emitted by the local engine.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenOutput {
    pub request_id: RequestId,
    pub token_id: u32,
}

/// Ordered event stream returned directly by the in-process engine.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EngineEvent {
    BatchStarted {
        request_ids: Box<[RequestId]>,
    },
    Token(TokenOutput),
    Finished {
        request_id: RequestId,
        reason: FinishReason,
    },
}

/// Type-erased local event stream. The server can translate this stream to
/// SSE or WebSocket frames without introducing an HTTP hop to the executor.
pub type EngineEventStream<E> =
    Pin<Box<dyn Stream<Item = Result<EngineEvent, E>> + Send + 'static>>;

/// Future that starts one validated local engine batch.
pub type EngineFuture<'a, E> =
    Pin<Box<dyn Future<Output = Result<EngineEventStream<E>, E>> + Send + 'a>>;

/// The only server-to-runtime execution boundary.
///
/// Implementations own the complete `OrbitKV` transaction and Luminal execution.
/// The server sees request intent and ordered output events, never physical KV
/// identities or device page tables.
pub trait Engine: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    fn execute(&self, batch: BatchIntent) -> EngineFuture<'_, Self::Error>;
}

#[cfg(test)]
mod tests {
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
}
