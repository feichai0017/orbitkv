//! Buffered CPU wall-time spans and reported measurements for attribution.
//!
//! The layer does not synchronize a device. Records stay in memory until the
//! caller finishes the trace, keeping filesystem writes outside measurements.

use std::{
    collections::BTreeMap,
    fs::{File, OpenOptions},
    io::{self, BufWriter, Write},
    path::Path,
    sync::{Arc, Mutex},
    time::Instant,
};

use serde_json::{Value, json};
use tracing::{Event, Id, Subscriber, field::Visit, span};
use tracing_subscriber::{
    Layer,
    filter::{LevelFilter, Targets},
    layer::{Context, SubscriberExt},
    registry::LookupSpan,
    util::SubscriberInitExt,
};

const STAGE_TARGET: &str = "orbitkv::stage";
const TRACE_ENV: &str = "ORBITKV_STAGE_TRACE";
const TRACE_SCHEMA_VERSION: u32 = 2;

#[derive(Clone, Default)]
struct Fields(BTreeMap<String, Value>);

impl Visit for Fields {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.0
            .insert(field.name().into(), json!(format!("{value:?}")));
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        self.0.insert(field.name().into(), json!(value));
    }

    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        self.0.insert(field.name().into(), json!(value));
    }

    fn record_i64(&mut self, field: &tracing::field::Field, value: i64) {
        self.0.insert(field.name().into(), json!(value));
    }

    fn record_u128(&mut self, field: &tracing::field::Field, value: u128) {
        self.0.insert(field.name().into(), json!(value));
    }

    fn record_i128(&mut self, field: &tracing::field::Field, value: i128) {
        self.0.insert(field.name().into(), json!(value));
    }

    fn record_f64(&mut self, field: &tracing::field::Field, value: f64) {
        self.0.insert(field.name().into(), json!(value));
    }

    fn record_bool(&mut self, field: &tracing::field::Field, value: bool) {
        self.0.insert(field.name().into(), json!(value));
    }
}

struct StageSpan {
    id: u64,
    parent: Option<u64>,
    start_ns: u128,
    thread: String,
    name: &'static str,
    fields: Fields,
}

struct State {
    origin: Instant,
    next_id: u64,
    open: usize,
    records: Vec<Value>,
}

/// Records spans and measurement events with target `orbitkv::stage`.
pub struct StageLayer(Arc<Mutex<State>>);

/// Owns the newly created trace file. Call `finish` after all stage spans close.
pub struct StageTraceGuard {
    state: Arc<Mutex<State>>,
    writer: Option<BufWriter<File>>,
}

/// Creates a buffered stage layer without installing a subscriber or replacing a file.
pub fn stage_trace_layer(path: impl AsRef<Path>) -> io::Result<(StageLayer, StageTraceGuard)> {
    let file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(
        &mut writer,
        &json!({
            "event": "trace_started", "schema": TRACE_SCHEMA_VERSION,
            "clock": "process-relative monotonic CPU wall time",
            "duration_scope": "synchronous span lifetime; inclusive of nested spans; not device time",
            "metric_scope": "reported measurements with explicit units; never span durations",
        }),
    )?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    let state = Arc::new(Mutex::new(State {
        origin: Instant::now(),
        next_id: 0,
        open: 0,
        records: Vec::new(),
    }));
    Ok((
        StageLayer(Arc::clone(&state)),
        StageTraceGuard {
            state,
            writer: Some(writer),
        },
    ))
}

/// Installs the optional diagnostic subscriber for a standalone process.
/// Embedders with an existing subscriber should compose `stage_trace_layer` instead.
pub fn install_stage_trace_from_env() -> io::Result<Option<StageTraceGuard>> {
    let Some(path) = std::env::var_os(TRACE_ENV) else {
        return Ok(None);
    };
    let (layer, guard) = stage_trace_layer(path)?;
    tracing_subscriber::registry()
        .with(
            layer.with_filter(
                Targets::new()
                    .with_default(LevelFilter::OFF)
                    .with_target(STAGE_TARGET, LevelFilter::INFO),
            ),
        )
        .try_init()
        .map_err(io::Error::other)?;
    Ok(Some(guard))
}

impl<S: Subscriber + for<'a> LookupSpan<'a>> Layer<S> for StageLayer {
    fn on_event(&self, event: &Event<'_>, ctx: Context<'_, S>) {
        if event.metadata().target() != STAGE_TARGET {
            return;
        }
        let parent = ctx.event_scope(event).and_then(|mut scope| {
            scope.find_map(|ancestor| ancestor.extensions().get::<StageSpan>().map(|data| data.id))
        });
        let mut fields = Fields::default();
        event.record(&mut fields);
        let mut state = self.0.lock().expect("stage trace lock poisoned");
        let at_ns = state.origin.elapsed().as_nanos();
        state.records.push(json!({
            "event": "metric", "parent": parent, "at_ns": at_ns,
            "name": event.metadata().name(), "fields": fields.0,
            "thread": format!("{:?}", std::thread::current().id()),
            "panicking": std::thread::panicking(),
        }));
    }

    fn on_new_span(&self, attrs: &span::Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        if attrs.metadata().target() != STAGE_TARGET {
            return;
        }
        let parent = ctx.span(id).and_then(|span| {
            span.parent().and_then(|parent| {
                parent.scope().find_map(|ancestor| {
                    ancestor.extensions().get::<StageSpan>().map(|data| data.id)
                })
            })
        });
        let mut fields = Fields::default();
        attrs.record(&mut fields);
        let mut state = self.0.lock().expect("stage trace lock poisoned");
        let data = StageSpan {
            id: state.next_id,
            parent,
            start_ns: state.origin.elapsed().as_nanos(),
            thread: format!("{:?}", std::thread::current().id()),
            name: attrs.metadata().name(),
            fields,
        };
        state.next_id += 1;
        state.open += 1;
        drop(state);
        ctx.span(id)
            .expect("new stage span missing")
            .extensions_mut()
            .insert(data);
    }

    fn on_record(&self, id: &Id, values: &span::Record<'_>, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id)
            && let Some(data) = span.extensions_mut().get_mut::<StageSpan>()
        {
            values.record(&mut data.fields);
        }
    }

    fn on_close(&self, id: Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(&id) else { return };
        let extensions = span.extensions();
        let Some(data) = extensions.get::<StageSpan>() else {
            return;
        };
        let mut state = self.0.lock().expect("stage trace lock poisoned");
        let end_ns = state.origin.elapsed().as_nanos();
        state.records.push(json!({
            "event": "stage", "id": data.id, "parent": data.parent,
            "name": data.name, "thread": data.thread, "start_ns": data.start_ns,
            "wall_duration_ns": end_ns - data.start_ns, "fields": data.fields.0,
            "panicking": std::thread::panicking(),
        }));
        state.open -= 1;
    }
}

impl StageTraceGuard {
    /// Flushes all records. A successful finish requires all spans to be closed.
    pub fn finish(mut self) -> io::Result<()> {
        self.flush(true)
    }

    fn flush(&mut self, completed: bool) -> io::Result<()> {
        let Some(mut writer) = self.writer.take() else {
            return Ok(());
        };
        let mut state = self
            .state
            .lock()
            .map_err(|_| io::Error::other("stage trace lock poisoned"))?;
        for record in state.records.drain(..) {
            serde_json::to_writer(&mut writer, &record)?;
            writer.write_all(b"\n")?;
        }
        let complete = completed && state.open == 0;
        serde_json::to_writer(
            &mut writer,
            &json!({
                "event": if complete { "trace_completed" } else { "trace_incomplete" },
                "open_spans": state.open,
            }),
        )?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        if completed && !complete {
            return Err(io::Error::other("stage spans remain open at trace finish"));
        }
        Ok(())
    }
}

impl Drop for StageTraceGuard {
    fn drop(&mut self) {
        if let Err(error) = self.flush(false) {
            eprintln!("failed to finish requested stage trace: {error}");
        }
    }
}

#[cfg(test)]
#[path = "../tests/unit/stages/mod.rs"]
mod tests;
