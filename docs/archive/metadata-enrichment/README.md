# Metadata Enrichment Archive

This archive contains historical metadata-enrichment plans, review reports, and superseded active
docs. They are retained as evidence, not active direction.

Current direction lives in:

- `docs/metadata/README.md`
- `docs/metadata/enrichment/README.md`
- `docs/plans/metadata-enrichment-current-plan.md`

## Reading Rule

Do not start here. Use this archive only to inspect why a decision was made or to recover
historical evidence that has already been summarized in active docs.

## Mined Content Register

| Source | Durable content mined | Active destination | Historical/stale content left here |
| --- | --- | --- | --- |
| `plans/gold-standards-and-enrichment-prompt-followup-2026-04-09.md` | Index codes must be exact or clinically substitutable; not every finding has good LOINC/RadLex; early gold-review lessons | `docs/metadata/fields.md`, `docs/metadata/enrichment/evaluation.md` | Old AB/VI-era subspecialty assumptions and outdated fixture values |
| `plans/metadata-assignment-full-gold-suite-expansion-2026-04-10.md` | Gold fixtures are regression evidence; eval expansion should expose field-specific misses | `docs/metadata/enrichment/evaluation.md` | Early prompt/eval mechanics superseded by split-agent evals |
| `plans/metadata-assignment-next-iteration-2026-04-11.md` | Prompt misses should become general rules; broad fields need field-specific guidance | `docs/metadata/enrichment/prompt-guidance.md`, `docs/metadata/fields.md` | Dated next-step checklist |
| `plans/metadata-assignment-targeted-prompt-honing-2026-04-11.md` | Use reviewed field definitions as source material; avoid embedding long exception lists | `docs/metadata/enrichment/prompt-guidance.md` | Old targeted-prompt details |
| `plans/metadata-full-suite-miss-triage-2026-04-12.md` | Miss triage should classify failure modes, not only chase total score | `docs/metadata/enrichment/evaluation.md` | Early case-level triage notes |
| `plans/metadata-targeted-example-pack-2026-04-12.md` | Examples must illustrate general rules and must not be copied from eval cases | `docs/metadata/enrichment/prompt-guidance.md` | Prompt example pack itself, which is not active prompt text |
| `plans/rsna-subspecialty-alignment-2026-04-12.md` | RSNA keep/drop rationale, AB removal, VI to VA, NM/SQ additions | `docs/metadata/subspecialties.md` | Dated implementation plan |
| `plans/coordinated-metadata-enrichment-and-dual-db-release-2026-04-26.md` | Two-repo separation, review app expectations, ontology cache/auditor as evidence tools, no hidden prompt examples, no unapproved source writes, dual DB artifact strategy, local wheelhouse bridge, package release gate | `docs/metadata/enrichment/README.md`, `docs/metadata/enrichment/human-review-and-writeback.md`, `docs/metadata/enrichment/database-artifacts-and-package-pinning.md`, `docs/plans/metadata-enrichment-current-plan.md` | Old broad release sequencing |
| `plans/coordinated-metadata-enrichment-implementation-log-2026-04-26.md` | Pilot facts: 150 reviewed, 46 approved/104 feedback, review UI/export/ingest behavior, dominant feedback themes, wheelhouse paths, script pinning decisions, DB/publish smoke status | `docs/metadata/enrichment/human-review-and-writeback.md`, `docs/metadata/enrichment/evaluation.md`, `docs/metadata/enrichment/database-artifacts-and-package-pinning.md` | Dated implementation log |
| `plans/metadata-enrichment-anatomy-scope-hardening-2026-05-05.md` | Anatomy scope/generalization rules; avoid hard-coded finding-name/code-pair rules | `docs/metadata/fields.md` | Old hardening checklist |
| `plans/pilot-feedback-tooling-hardening-subplan-2026-05-05.md` | Grading-aware comparisons; fix at prompt/schema/validator/candidate/audit layer as appropriate; do not mark source covered merely because corrected source validates | `docs/metadata/enrichment/evaluation.md`, `docs/metadata/enrichment/human-review-and-writeback.md` | Earlier required-field and `clear_fields` content |
| `plans/metadata-enrichment-right-sized-tool-2026-05-06.md` | Split-agent architecture, no deterministic keyword assignment, concise prompts, supervised promotion contract, candidate caps, source support standards | `docs/metadata/enrichment/prompt-guidance.md`, `docs/metadata/enrichment/human-review-and-writeback.md`, `docs/plans/metadata-enrichment-current-plan.md` | Old "reassess" and `clear_fields` details |
| `plans/metadata-enrichment-supervised-review-prompt-2026-05-10.md` | Sub-agent review can triage accept/skip/needs-attention/tool-problem, but human actions remain authority | `docs/metadata/enrichment/human-review-and-writeback.md`, `docs/metadata/enrichment/prompt-guidance.md` | Prompt text not used as an eval grader |
| `plans/metadata-prompt-repair-2026-05-11.md` | Remove worklist/workflow framing; prompt pattern should be concise Markdown with a focused decision model | `docs/metadata/enrichment/prompt-guidance.md` | LLM grader hook details |
| `plans/metadata-eval-scoring-and-prompt-cleanup-2026-05-13.md` | Gates versus scores; commission-sensitive scoring; only `entity_type` required; no `clear_fields`; component evals are diagnostic | `docs/metadata/enrichment/evaluation.md`, `docs/plans/metadata-enrichment-current-plan.md` | Boolean LLM grader assumptions |
| `plans/etiology-tempo-verifiable-tuning-2026-05-18.md` | Etiology/time-course adjudications, commission-over-omission policy, no schema dump in prompts, remove hemangioma overfit, reviewed clean-input cases as eval evidence | `docs/metadata/fields.md`, `docs/metadata/enrichment/evaluation.md`, `docs/metadata/enrichment/prompt-guidance.md` | Case-walkthrough notes not suitable as prompt text |
| `plans/metadata-enrichment-current-readiness-2026-05-24.md` | Gate A/Gate B direction, human review only authoritative, 180 events/150 unique/67 approved/83 feedback, 160-to-78 reconciliation, optional future LLM review only | `docs/metadata/enrichment/human-review-and-writeback.md`, `docs/plans/metadata-enrichment-current-plan.md` | Older slice-status narrative |
| `plans/metadata-enrichment-plan-history-2026-05-24.md` | Plan lineage and file inventory for cleanup | `docs/plans/metadata-enrichment-current-plan.md` | Historical plan archaeology |
| `plans/metadata-enrichment-working-tree-inventory-2026-05-24.md` | Distinction between review evidence, tool/eval code, historical docs, and untrusted generated source output | `docs/plans/metadata-enrichment-current-plan.md`, `docs/metadata/enrichment/human-review-and-writeback.md` | Working-tree snapshot |
| `plans/metadata-prompt-improvements-from-head-ct-traces.md` | Source context, anatomy null over whole-body fallback, weak candidate rejection, negative finding handling, and technique/device/postoperative state cautions | `docs/metadata/fields.md`, `docs/metadata/enrichment/prompt-guidance.md` | Old prompt forms and stale `clear_fields` references |
| `reviews/subspecialty-domain-broad-eval-2026-05-12.md` | Horizontal domains overlay body-region domains; cardiac plus chest; cervical lymphoid tissue HN; aorta VA; OI overlay; IR/device guidance | `docs/metadata/subspecialties.md` | Individual sampled misses |
| `reviews/modality-applicability-broad-eval-2026-05-12.md` | Direct modality language; avoid XR overcalls for radiolucent calculus/torsion; avoid every-modality treatment response/artifact; hydronephrosis US+CT | `docs/metadata/fields.md` | Individual sampled misses |
| `reviews/etiology-tempo-broad-eval-2026-05-12.md` | Etiology means common process type, not all possible causes; time course uses long end of common observable persistence | `docs/metadata/fields.md`, `docs/metadata/enrichment/prompt-guidance.md` | Individual sampled misses |
| `reviews/etiology-tempo-extra-etiology-review-2026-05-18.md` | Extra etiology additions are risky; descriptive findings need null unless process is implied | `docs/metadata/fields.md`, `docs/metadata/enrichment/evaluation.md` | Case-specific review text |
| `reviews/metadata-review-feedback-summary-2026-05-24.md` | Latest feedback is valuable future work but not source-writeback authority | `docs/metadata/enrichment/human-review-and-writeback.md`, `docs/metadata/enrichment/evaluation.md` | Feedback row details |
| `reviews/metadata-review-expected-candidates-2026-05-24.md` | 57 feedback-derived candidates require human promotion before they become gold/writeback authority | `docs/metadata/enrichment/human-review-and-writeback.md`, `docs/metadata/enrichment/evaluation.md` | Candidate extraction details |
| `active-docs/gold-standard-review.md` | Older gold worksheet retained for provenance only | Later adjudications in `docs/metadata/fields.md` and checked-in fixtures | Conflicting old expected values |
| `active-docs/proposed-prompt-examples.md` | Warning example for prompt-spam anti-pattern | `docs/metadata/enrichment/prompt-guidance.md` | Old examples with stale values and legacy labels |

## Archived Plan Groups

- Early gold and prompt work: early April 2026 plans and targeted example packs.
- Coordinated enrichment and data-repo work: late April 2026 pilot/release plans.
- May 2026 tool hardening and prompt repair: split-agent, prompt-rationalization, and scoring plans.
- Cleanup/readiness snapshots: late May 2026 branch-readiness and working-tree inventories.

## Archived Review Reports

The review reports under `reviews/` remain historical evidence for prompt and eval tuning. Current
policy has been mined into `docs/metadata/fields.md`, `docs/metadata/subspecialties.md`,
`docs/metadata/enrichment/evaluation.md`, and `docs/metadata/enrichment/prompt-guidance.md`.
