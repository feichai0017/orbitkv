# ADR 0001: Greenfield core with source-managed providers

Status: accepted.

The former implementation and its model-specific assumptions were removed from
the working tree. Git history is the only migration and recovery mechanism.
The new core is greenfield, but external runtimes and kernel libraries are used
from pinned source revisions.

This separates two goals that otherwise conflict: learning the complete code
path and avoiding low-value reimplementation. We read, debug, profile, and may
patch upstream source while keeping model semantics, qualification, selection,
and rollback contracts independent of any one provider.

The consequence is that no previous crate or manifest format receives implicit
compatibility. Any reused idea must re-enter through a new contract and test.
