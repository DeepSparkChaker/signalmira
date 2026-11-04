# SignalMira Topics v5 changes from v4

- Per-tenant persisted `Policy` with configurable 7-day / 200-review / 5000-review / 30% unassigned triggers, 24-hour cooldown.
- `--check` is read-only and avoids model loading; `--auto` is idempotent and tenant locked.
- Exact per-review source-hash change tracking with a frozen input snapshot, rather than `review_date` or timestamps alone.
- Hybrid 70/30 new/changed vs historical sampling. History stratified by prior topic and ISO language, with product diversity and deterministic hash selection. `--sample-mode new-only` supported.
- Independent candidate and approval workflow preserved; pending candidates block duplicate discovery; `--reject` added.
- Existing `--assign`, `--activate`, `--rename`, model profiles, versioned embeddings and taxonomies retained.
- RAG remains a separate embedding system.

No migration deletes original customer reviews or old topic taxonomies.
