//! Native rendering is CPU-only and does not require a checkpoint chat template.
use vllm_chat::{ChatMessage, ChatRenderer, ChatRequest, DeepSeekV4ChatRenderer};

#[test]
fn native_nonthinking_user_prompt_matches_supplied_v41_encoding() {
    let mut request = ChatRequest { messages: vec![ChatMessage::user("Hello.")], ..ChatRequest::for_test() };
    request.chat_options.template_kwargs.insert("thinking".into(), serde_json::Value::Bool(false));
    let actual = DeepSeekV4ChatRenderer::new().render(&request).unwrap().prompt.into_text().unwrap();
    // Generated with the supplied encoding.py encode_messages(..., thinking_mode="chat").
    // This deliberately covers only the common nonthinking text format; V4.1
    // changes system messages, reasoning effort and tool tags.
    assert_eq!(actual, "<｜begin▁of▁sentence｜><｜User｜>Hello.<｜Assistant｜></think>");
}

#[test]
fn v41_matches_released_reference_text_cases() {
    use serde_json::Value;
    use vllm_chat::{AssistantContentBlock, AssistantToolCall, ChatTool, DeepSeekV41ChatRenderer, ResolvedToolContext};
    let cases: Vec<Value> = serde_json::from_str(include_str!("fixtures/deepseek_v41.json")).unwrap();
    for case in cases {
        let mut messages = Vec::new();
        let mut tools: Vec<ChatTool> = Vec::new();
        for message in case["messages"].as_array().unwrap() {
            if let Some(list) = message["tools"].as_array() {
                tools.extend(
                    list.iter().map(|tool| serde_json::from_value::<ChatTool>(tool["function"].clone()).unwrap()),
                );
            }
            let text = message["content"].as_str().unwrap_or("");
            messages.push(match message["role"].as_str().unwrap() {
                "system" => ChatMessage::system(text),
                "user" => ChatMessage::user(text),
                "assistant" => {
                    let mut blocks = Vec::new();
                    if let Some(reasoning) = message["reasoning_content"].as_str() {
                        blocks.push(AssistantContentBlock::Reasoning { text: reasoning.into() });
                    }
                    if !text.is_empty() {
                        blocks.push(AssistantContentBlock::Text { text: text.into() });
                    }
                    if let Some(calls) = message["tool_calls"].as_array() {
                        for call in calls {
                            let f = &call["function"];
                            blocks.push(AssistantContentBlock::ToolCall(AssistantToolCall {
                                id: call["id"].as_str().or(f["id"].as_str()).unwrap_or("").into(),
                                name: f["name"].as_str().unwrap().into(),
                                arguments: f["arguments"]
                                    .as_str()
                                    .map(str::to_owned)
                                    .unwrap_or_else(|| f["arguments"].to_string()),
                            }));
                        }
                    }
                    ChatMessage::assistant_blocks(blocks)
                }
                "tool" => ChatMessage::tool_response(text, message["tool_call_id"].as_str().unwrap_or("")),
                other => panic!("unsupported fixture role {other}"),
            });
        }
        let tool_context = ResolvedToolContext::new(&messages, tools, None, true).unwrap();
        let mut request = ChatRequest { messages, tool_context, ..ChatRequest::for_test() };
        for (key, source) in
            [("thinking", "thinking"), ("reasoning_effort", "effort"), ("drop_thinking", "drop_thinking")]
        {
            request.chat_options.template_kwargs.insert(key.into(), case[source].clone());
        }
        let rendered = DeepSeekV41ChatRenderer::new().render(&request);
        if case["error"] == Value::Bool(true) {
            assert!(rendered.is_err(), "{} should fail", case["name"]);
        } else {
            // The checkpoint oracle decodes double-encoded objects twice.
            // deepseek-recipe (and vLLM #56260) instead preserves that raw
            // string as `arguments`; keep the original fixture intact and
            // require the rest of the rendered conversation to match it.
            let expected = case["expected"].as_str().unwrap().replace(
                r#"<｜DSML｜ parameter name="query" string="true">double</｜DSML｜ parameter>"#,
                r#"<｜DSML｜ parameter name="arguments" string="true">"{\"query\":\"double\"}"</｜DSML｜ parameter>"#,
            );
            assert_eq!(rendered.unwrap().prompt.into_text().unwrap(), expected, "{}", case["name"]);
        }
    }
}

#[test]
fn v41_typed_reasoning_effort_controls_numeric_budget() {
    use vllm_chat::{DeepSeekV41ChatRenderer, ReasoningEffort};
    for (effort, budget) in [
        (ReasoningEffort::Low, 25),
        (ReasoningEffort::High, 50),
        (ReasoningEffort::XHigh, 75),
        (ReasoningEffort::Max, 100),
    ] {
        let mut request = ChatRequest::for_test();
        request.chat_options.reasoning_effort = Some(effort);
        let prompt = DeepSeekV41ChatRenderer::new().render(&request).unwrap().prompt.into_text().unwrap();
        assert!(
            prompt.starts_with(&format!("<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: {budget} (range 1-100,"))
        );
    }
    let mut request = ChatRequest::for_test();
    request.chat_options.reasoning_effort = Some(ReasoningEffort::Medium);
    assert!(DeepSeekV41ChatRenderer::new().render(&request).is_err());
}
