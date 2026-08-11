# Rename provenance

The project changed its current name from Loom Infer to Oxide Infer. The
rename applies to active source, packages, command-line programs, environment
variables, documentation, and future evidence.

## Current identifiers

| Role | Current identifier |
| --- | --- |
| Project and repository | `oxide-infer` |
| Contract crate | `oxide-infer` |
| CUDA crate | `oxide-infer-cuda` |
| Hardware and benchmark crate | `oxide-infer-lab` |
| Native provider | `Oxide` |
| Rust module prefix | `oxide_infer` |
| Environment prefix | `OXIDE_` |
| Native benchmark label | `oxide-infer` |

Current source does not provide legacy aliases for crates, modules, providers,
commands, or environment variables.

## Historical identity

Commits, tags, evidence records, engine-adapter records, and external links
created before the rename keep their original identifiers. Their names bind
them to the source and schemas that produced them.

The project does not rewrite:

- reviewed files in `docs/results`
- Git commit or tag names
- hashes embedded in evidence
- historical repository URLs in pinned source links
- historical record bytes and embedded identifiers

An old record that names the `Loom` provider qualifies only its recorded
source. It does not qualify the current `Oxide` provider identity when both
records share the same numerical contract.

## New evidence

New records use the current crate, provider, command, and environment names.
Each record references the exact renamed source commit and artifact hash.

A new record can cite an old record as historical context. It cannot replace
the old record or inherit its qualification. Re-running the same fixture on
renamed source produces a new evidence file.

## Links and integration records

Current documentation links to `oxide-infer` paths. A link to a pinned old
commit can retain the old repository or directory name when that name is part
of the historical source identity.

The Mistral.rs proof-of-concept records keep their original file names and
bytes. The example-directory rename moves those files under current Oxide
Infer paths without changing their recorded source identity. A new Mistral.rs
or vLLM adapter must use current identifiers and publish a new source pair.
