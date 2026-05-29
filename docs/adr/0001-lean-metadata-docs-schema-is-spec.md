# Metadata docs are a lean decision/policy record; the Pydantic schema is the spec

The `docs/metadata/**` tree records judgments and rules — field-decision standards, subspecialty
policy, review/writeback gates, release strategy — not structural facts. Field types, regex, the
`IndexCode` shape, and value-normalization behavior live in the Pydantic models and are read from
code; enum *value* tables may appear in `fields.md` but should be generated or checked against the
enums rather than hand-authored. We chose this because hand-maintained structural docs drift from
the models and produce stale, misleading content (a prior consolidation lost and duplicated exactly
this kind of detail); the trade-off is that a reader sometimes needs the code open for exact types.

## Consequences

- "Lost" structural content from older docs is intentionally not restored into the docs tree.
- A future reader who finds no field types in `fields.md` should look at the models, not re-add them.
