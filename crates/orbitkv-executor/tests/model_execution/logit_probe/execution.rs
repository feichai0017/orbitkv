//! Execute a validated schedule and retain both logical logits and physical batch geometry.

use super::*;
use batches::PlannedQuery;
use std::io::{BufWriter, Write};

fn request_id(case: usize) -> EngineRequestId {
    EngineRequestId(u64::try_from(case).unwrap().checked_add(1).unwrap())
}

pub(super) fn run(
    probe: &Probe,
    batches: &[Vec<PlannedQuery>],
    decoder: &mut CompiledDecoder,
    harness: &mut ModelBatchHarness,
    vocabulary: usize,
    directory: &std::path::Path,
) -> (Vec<CaseTrace>, Vec<serde_json::Value>) {
    let mut traces = probe
        .cases
        .iter()
        .map(|case| CaseTrace {
            id: case.id.clone(),
            prompt_token_ids: case.prompt_token_ids.clone(),
            steps: Vec::new(),
            drain_passed: false,
        })
        .collect::<Vec<_>>();
    let mut submissions = Vec::new();
    for (batch_index, queries) in batches.iter().enumerate() {
        let acquired = queries
            .iter()
            .filter(|q| q.start == 0)
            .map(|q| request_id(q.case))
            .collect::<Vec<_>>();
        if !acquired.is_empty() {
            harness.session.acquire_requests(&acquired).unwrap();
        }
        let intents = queries
            .iter()
            .map(|q| EngineAppendIntent {
                request_id: request_id(q.case),
                target_boundary: u64::try_from(q.end).unwrap(),
            })
            .collect::<Vec<_>>();
        let source = harness.session.prepare_append_batch(&intents).unwrap();
        let view = harness
            .session
            .prepared_execution_view(source.batch_id)
            .unwrap();
        let attention = harness.executor_plan.attention_batches(&view).unwrap();
        let prepared = harness
            .executor_plan
            .lower_prepared(source, &harness.arenas)
            .unwrap();
        let write_slots = batch_write_slots(&prepared, attention.len());
        let classes = batch_decoder_class_steps(&write_slots, &attention);
        let states = prepared
            .fixed_state_requests()
            .map(|(request_id, states)| DecoderFixedStateStep { request_id, states })
            .collect::<Vec<_>>();
        let tokens = queries
            .iter()
            .flat_map(|q| probe.cases[q.case].inputs(q.start, q.end))
            .collect::<Vec<_>>();
        let positions = queries
            .iter()
            .flat_map(|q| q.start..q.end)
            .map(|p| u32::try_from(p).unwrap())
            .collect::<Vec<_>>();
        let step = DecoderStep {
            tokens: &tokens,
            positions: &positions,
            classes: &classes,
        };
        let output = if harness.executor_plan.fixed_states.is_empty() {
            let output = decoder
                .execute_with_logits(step)
                .unwrap_or_else(|error| panic!("submission {batch_index}: {error:?}"));
            StatefulDecoderDiagnosticOutput {
                token_ids: output.token_ids,
                logits: output.logits,
                fixed_states: Box::default(),
            }
        } else {
            decoder
                .execute_with_fixed_states_and_logits(step, &states)
                .unwrap_or_else(|error| panic!("submission {batch_index}: {error:?}"))
        };
        let output_rows = probe.output_row_count(tokens.len(), queries.len());
        assert_eq!(output.logits.len(), output_rows * vocabulary);
        assert_eq!(output.token_ids.len(), output_rows);
        write_logits(probe, queries, &output, &mut traces, vocabulary, directory);
        complete_with_fixed_states(
            &mut harness.session,
            &prepared,
            &harness.arenas,
            &output.fixed_states,
            u64::try_from(batch_index + 1).unwrap(),
        );
        let finished = queries
            .iter()
            .filter(|q| q.finished)
            .map(|q| request_id(q.case))
            .collect::<Vec<_>>();
        if !finished.is_empty() {
            release_requests(&mut harness.session, &finished);
        }
        submissions.push(record_batch(probe, batch_index, queries, tokens.len()));
    }
    assert_session_drained(&harness.session);
    assert_fixed_states_drained(&harness.session);
    for (trace, case) in traces.iter_mut().zip(&probe.cases) {
        assert_eq!(trace.steps.len(), case.continuation_token_ids.len());
        trace.drain_passed = true;
    }
    (traces, submissions)
}

fn assert_fixed_states_drained(session: &RuntimeSession) {
    assert!(session.fixed_state_stats().iter().all(|(_, state)| {
        state.active_owners == 0
            && state.pending_transitions == 0
            && state.pending_retirements == 0
            && state.free_slots == u64::from(state.identity.slot_count)
    }));
}

fn write_logits(
    probe: &Probe,
    queries: &[PlannedQuery],
    output: &StatefulDecoderDiagnosticOutput,
    traces: &mut [CaseTrace],
    vocabulary: usize,
    directory: &std::path::Path,
) {
    let mut row_end = 0;
    for query in queries {
        row_end += probe.output_row_count(query.end - query.start, 1);
        let Some(step) = query.output_step else {
            continue;
        };
        let case = &probe.cases[query.case];
        let row = row_logits(&output.logits, row_end - 1, vocabulary);
        assert!(row.iter().all(|value| value.is_finite()));
        let logits_file = format!("case-{}-step-{step}.f32", query.case);
        let mut file =
            BufWriter::new(std::fs::File::create_new(directory.join(&logits_file)).unwrap());
        for value in row {
            file.write_all(&value.to_le_bytes()).unwrap();
        }
        file.flush().unwrap();
        let steps = &mut traces[query.case].steps;
        assert_eq!(steps.len(), step);
        steps.push(StepTrace {
            step,
            // Logical teacher-forced input; physical prompt chunks and
            // submitted row order are separately retained in batches.
            input_token_ids: if step == 0 {
                case.prompt_token_ids.clone()
            } else {
                vec![case.continuation_token_ids[step - 1]]
            },
            selected_token_id: output.token_ids[row_end - 1],
            logits_file,
        });
    }
}

fn record_batch(
    probe: &Probe,
    batch_index: usize,
    queries: &[PlannedQuery],
    query_tokens: usize,
) -> serde_json::Value {
    serde_json::json!({
            "batch": batch_index, "query_tokens": query_tokens,
            "requests": queries.iter().map(|q| serde_json::json!({
                "case_id": probe.cases[q.case].id, "request_id": request_id(q.case).0,
                "input_token_ids": probe.cases[q.case].inputs(q.start, q.end),
                "positions": (q.start..q.end).collect::<Vec<_>>(),
                "output_step": q.output_step, "released": q.finished,
            })).collect::<Vec<_>>(),
    })
}
