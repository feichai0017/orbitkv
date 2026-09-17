//! The log line: `HH:MM:SS.mmm LEVEL message key=value ...` — the event's
//! own fields, then the enclosing spans' (the frontend's `request_id` and
//! friends). One format; colour only when stderr is a terminal. The clock
//! is UTC. `RUST_LOG` replaces [`DEFAULT_FILTER`] wholesale.

use std::fmt;
use std::io::IsTerminal;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tracing::field::{Field, Visit};
use tracing::{Event, Level, Subscriber};
use tracing_subscriber::field::RecordFields;
use tracing_subscriber::fmt::format::Writer;
use tracing_subscriber::fmt::{FmtContext, FormatEvent, FormatFields, FormattedFields};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

/// The upstream vLLM router logs every finished request at info, one line
/// per request; its warnings and errors still come through.
pub const DEFAULT_FILTER: &str = "info,vllm_server::routes=warn";

pub fn init() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(DEFAULT_FILTER));
    let layer = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stderr)
        .with_ansi(std::io::stderr().is_terminal())
        .event_format(Line)
        .fmt_fields(Fields);
    tracing_subscriber::registry().with(filter).with(layer).init();
}

/// Milliseconds with one decimal, for a `_ms` field.
pub fn ms(d: Duration) -> f64 {
    (d.as_secs_f64() * 1e4).round() / 10.0
}

/// Seconds with one decimal, for a `_s` field.
pub fn secs(d: Duration) -> f64 {
    (d.as_secs_f64() * 10.0).round() / 10.0
}

struct Line;

impl<S, N> FormatEvent<S, N> for Line
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    fn format_event(&self, ctx: &FmtContext<'_, S, N>, mut w: Writer<'_>, event: &Event<'_>) -> fmt::Result {
        let ansi = w.has_ansi_escapes();
        let (color, name) = match *event.metadata().level() {
            Level::ERROR => ("31", "ERROR"),
            Level::WARN => ("33", "WARN "),
            Level::INFO => ("32", "INFO "),
            Level::DEBUG => ("34", "DEBUG"),
            Level::TRACE => ("35", "TRACE"),
        };
        if ansi {
            write!(w, "\x1b[2m{}\x1b[0m \x1b[{color}m{name}\x1b[0m ", clock())?;
        } else {
            write!(w, "{} {name} ", clock())?;
        }
        let mut msg = Message(None);
        event.record(&mut msg);
        if let Some(m) = msg.0 {
            write!(w, "{m}")?;
        }
        ctx.field_format().format_fields(w.by_ref(), event)?;
        if let Some(scope) = ctx.event_scope() {
            for span in scope.from_root() {
                let ext = span.extensions();
                if let Some(f) = ext.get::<FormattedFields<N>>() {
                    write!(w, "{}", f.fields)?;
                }
            }
        }
        writeln!(w)
    }
}

/// UTC wall clock, `HH:MM:SS.mmm`.
fn clock() -> String {
    let d = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    let s = d.as_secs() % 86_400;
    format!("{:02}:{:02}:{:02}.{:03}", s / 3600, s % 3600 / 60, s % 60, d.subsec_millis())
}

/// Just the `message` field.
struct Message(Option<String>);

impl Visit for Message {
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        if field.name() == "message" {
            self.0 = Some(format!("{value:?}"));
        }
    }
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.0 = Some(value.to_owned());
        }
    }
}

/// Every field but `message` (and the `log.*` provenance the `log`
/// bridge adds), each as ` key=value` (leading space, so event and span
/// fields concatenate). Strings are bare unless they hold whitespace or
/// quotes; a bracketed list or map is its own delimiter.
struct Fields;

impl<'w> FormatFields<'w> for Fields {
    fn format_fields<R: RecordFields>(&self, writer: Writer<'w>, fields: R) -> fmt::Result {
        let mut v = Kv { w: writer, err: Ok(()) };
        fields.record(&mut v);
        v.err
    }
}

struct Kv<'w> {
    w: Writer<'w>,
    err: fmt::Result,
}

impl Kv<'_> {
    fn put(&mut self, field: &Field, value: fmt::Arguments<'_>) {
        if field.name() != "message" && !field.name().starts_with("log.") && self.err.is_ok() {
            self.err = write!(self.w, " {}={value}", field.name());
        }
    }
    fn put_str(&mut self, field: &Field, s: &str) {
        let delimited = s.starts_with('[') && s.ends_with(']') || s.starts_with('{') && s.ends_with('}');
        let bare =
            !s.is_empty() && !s.starts_with('"') && !s.chars().any(|c| c.is_whitespace() || c == '"' || c == '=');
        let bare = bare || delimited && !s.contains('"');
        if bare {
            self.put(field, format_args!("{s}"));
        } else {
            self.put(field, format_args!("{s:?}"));
        }
    }
}

impl Visit for Kv<'_> {
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        let s = format!("{value:?}");
        // `?x` on a string arrives already quoted; `%x` and everything
        // else arrives bare and gets quoted only when it needs to be.
        if s.starts_with('"') && s.ends_with('"') {
            self.put(field, format_args!("{s}"));
        } else {
            self.put_str(field, &s);
        }
    }
    fn record_str(&mut self, field: &Field, value: &str) {
        self.put_str(field, value);
    }
    fn record_f64(&mut self, field: &Field, value: f64) {
        // Six decimals, trailing zeros dropped: an f32 arrives widened
        // (0.7 as 0.699999988079071) and a rate needs no more than that.
        let s = format!("{value:.6}");
        let s = s.trim_end_matches('0').trim_end_matches('.');
        self.put(field, format_args!("{s}"));
    }
    fn record_i64(&mut self, field: &Field, value: i64) {
        self.put(field, format_args!("{value}"));
    }
    fn record_u64(&mut self, field: &Field, value: u64) {
        self.put(field, format_args!("{value}"));
    }
    fn record_bool(&mut self, field: &Field, value: bool) {
        self.put(field, format_args!("{value}"));
    }
    fn record_error(&mut self, field: &Field, value: &(dyn std::error::Error + 'static)) {
        self.put_str(field, &value.to_string());
    }
}
