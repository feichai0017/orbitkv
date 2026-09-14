use anyhow::Result;
use egglog::EGraph;
use std::fmt::{Debug, Display};
use std::string::String;

use crate::prelude::petgraph::dot::{Config, Dot};
use crate::prelude::*;
use petgraph::Directed;
use petgraph::prelude::StableGraph;
use std::io::Write;

pub trait ToHtml {
    fn to_html(&self) -> Result<String>;
}

pub trait ToDot {
    fn to_dot(&self) -> Result<String>;
}

const EGRAPH_VIS_TEMPLATE: &str = r#"
<div id="egraph-visualizer"></div>
<link rel="stylesheet" href="https://esm.sh/egraph-visualizer/dist/style.css" />
<script type="module">
    import { mount } from "https://esm.sh/egraph-visualizer";
    const egraph = {{JSON_TEMPLATE}};
    const mounted = mount(document.getElementById("egraph-visualizer"));
    mounted.render([JSON.stringify(egraph)]);
</script>"#;

impl ToHtml for EGraph {
    fn to_html(&self) -> Result<String> {
        let egraph_as_json = serde_json::to_string_pretty(
            &self.serialize(egglog::SerializeConfig::default()).egraph,
        )?;
        Ok(EGRAPH_VIS_TEMPLATE.replace("{{JSON_TEMPLATE}}", &egraph_as_json))
    }
}

impl ToDot for EGraph {
    fn to_dot(&self) -> Result<String> {
        Ok(self
            .serialize(egglog::SerializeConfig::default())
            .egraph
            .to_dot())
    }
}

/// Implements `ToDot` for [`LLIRGraph`] and [`HLIRGraph`]
/// TODO: This simple extraction can be improved in the future with support for edge labels
impl<N, E> ToDot for petgraph::stable_graph::StableGraph<N, E>
where
    N: std::fmt::Debug,
    E: std::fmt::Debug,
{
    fn to_dot(&self) -> Result<String> {
        Ok(format!(
            "{:?}",
            Dot::with_config(self, &[Config::EdgeNoLabel])
        ))
    }
}

/// View a debug graph in the browser using OrbitKV Visualizer
pub fn display_graph<E>(graph: &StableGraph<impl Display + Debug, E, Directed, u32>) {
    let dot_source = graph_to_dot(graph, None);

    // Open in OrbitKV Visualizer (served from graph_viewer dev server)
    let url = format!(
        "http://viz.orbitkv.compiler.com/?dot={}",
        urlencoding::encode(&dot_source)
    );
    let _ = open::that(&url);
}

/// View a debug graph in the browser using OrbitKV Visualizer
pub fn display_graph_marked_nodes<E>(
    graph: &StableGraph<impl Display + Debug, E, Directed, u32>,
    mark_nodes: Option<Vec<NodeIndex>>,
) {
    let dot_source = graph_to_dot(graph, mark_nodes);

    // Open in OrbitKV Visualizer (served from graph_viewer dev server)
    let url = format!(
        "http://localhost:5173/?dot={}",
        urlencoding::encode(&dot_source)
    );
    let _ = open::that(&url);
}

/// View a debug graph in the browser using OrbitKV Visualizer
pub fn display_graph_to_file<E>(
    graph: &StableGraph<impl Display + Debug, E, Directed, u32>,
    mark_nodes: Option<Vec<NodeIndex>>,
    file_name: &str,
) {
    let dot_source = graph_to_dot(graph, mark_nodes);

    // Write the DOT source to the file
    let mut file = std::fs::File::create(file_name).unwrap();
    file.write_all(dot_source.as_bytes()).unwrap();
}

fn escape_dot_string(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn graph_to_dot<E>(
    graph: &StableGraph<impl Display + Debug, E, Directed, u32>,
    mark_nodes: Option<Vec<NodeIndex>>,
) -> String {
    let mut dot = String::from("digraph {\n");
    let mut map = std::collections::HashMap::new();

    for (next_id, node) in graph.node_indices().enumerate() {
        let weight = graph.node_weight(node).unwrap();
        let label = escape_dot_string(&format!("{}", weight));
        let tooltip = escape_dot_string(&format!("{:?}", weight));
        let id = next_id;
        map.insert(node, id);

        let is_marked = mark_nodes
            .as_ref()
            .map(|m| m.contains(&node))
            .unwrap_or(false);

        if is_marked {
            dot.push_str(&format!(
                "    n{id} [label=\"{label}\", tooltip=\"{tooltip}\", style=filled, fillcolor=\"#ffeb3b\", fontcolor=\"black\", color=\"#ffd600\"];\n"
            ));
        } else {
            dot.push_str(&format!(
                "    n{id} [label=\"{label}\", tooltip=\"{tooltip}\"];\n"
            ));
        }
    }

    for edge in graph.edge_indices() {
        let (src, dest) = graph.edge_endpoints(edge).unwrap();
        if let (Some(src_id), Some(dest_id)) = (map.get(&src), map.get(&dest)) {
            dot.push_str(&format!("    n{src_id} -> n{dest_id};\n"));
        }
    }

    dot.push_str("}\n");
    dot
}
