# Pre-Next archive record

This branch preserves the complete non-ignored working tree immediately before
the OrbitKV Next reset on 2026-09-17.

- Archive branch: `archive-orbitkv-pre-next-20260917`
- Original base: `1f6c144d440ff0208db454c95669aecbc6623ea9`
- Working-tree snapshot commit: `0e8bdd62db324fbc5168047db26dc141202c3665`
- Snapshot tree: `f77ee2f903f2206fb62c0dfe7b20bae96bca9a00`
- Snapshot scope: 75 previously tracked changes/deletions plus all 30
  non-ignored untracked source, test, fixture, and design files.

The following ignored local data was deliberately not committed because it is
generated, cache-like, or too large for Git. It remained on the workstation at
reset time:

- `.qualification/`: approximately 123 GiB of binaries, traces, and experiment
  output;
- `target/`: approximately 198 GiB of Cargo build output;
- `website/node_modules/`, `website/dist/`, and `website/.astro/`;
- Python and pytest caches.

The tracked `results/` directory remains in the archive branch. No ignored file
was deleted as part of creating this archive.

To inspect or restore the source snapshot, use the archive branch or the exact
commit above. Do not base new implementation work on this branch; the active
Next line has a deliberately independent build graph.
