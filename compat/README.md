# Compatibility paths

`compat/` contains integrations that are no longer the primary product
architecture. They remain executable for regression comparison, source
provenance, and staged migration.

The SGLang path preserves the complete pinned upstream source, reviewed
overlay, Python bridge, qualification tools, and historical wire contract. It
must not become a second KV ownership authority: OrbitKV remains responsible
for page selection, generations, retirement, and safe reuse.
