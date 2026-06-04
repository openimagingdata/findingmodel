# Metadata Enrichment Readiness Plan Review

Status: Draft review
Date: 2026-06-02

## Reviewed Plan

Source plan: `/Users/talkasab/.claude/plans/kind-frolicking-beaver.md`

Review goal: improve the readiness plan before execution so the next metadata-enrichment push has
clear gates, low ambiguity, and a controlled path from current cleanup work to supervised corpus
scale.

## Bottom Line

The plan is directionally strong. Its best move is shifting the project from "the tool seems pretty
good" to a measurable readiness process: pinned model, held-out tripwire, fresh unseen
certification, human authority, and continuous supervised sampling during scale runs.

Before executing it, tighten five areas:

- make it an extension of the current active cleanup/Gate A/Gate B plan, not a replacement;
- define dataset authority and split rules so feedback records do not silently become gold;
- make scoring denominators and grade semantics precise enough to automate;
- calibrate the flag-review cap before making flag rate itself a hard readiness gate;
- pin the actual effective model/fallback behavior in a way that is externally verifiable.

## What To Keep

1. **Two-stage readiness verdict.** Requiring both a held-out reviewed split and a fresh unseen
   certification batch is the right structure. The held-out split catches overfitting to existing
   review artifacts; the fresh batch tests corpus behavior.
2. **Gates versus scored quality.** This matches `docs/metadata/enrichment/evaluation.md` and the
   current Pydantic Evals pattern, where assertions/gates and numeric scores are separate report
   surfaces.
3. **Commission-sensitive quality.** The plan correctly treats unsupported additions as more costly
   than omissions, especially for etiologies, time course, modalities, and subspecialties.
4. **Human authority and supervised writeback.** The plan preserves the existing rule that the agent
   proposes, the reviewer approves, and source writes go through manifest-backed apply tooling.
5. **No prompt eval-case dumping.** The plan's "fix at the lowest reliable layer" rule is important
   and should stay explicit in the execution plan.

## Required Improvements

### 1. Add A Phase Minus One: Close Or Supersede The Current Active Plan

The reviewed plan says the split-agent system is built, committed, and running. The repository's
active plan and enrichment README still frame the immediate milestone as cleanup, consolidated
docs, validated review evidence, Gate A writeback controls, Gate B cleanup controls, and database
publication gates.

Recommended change:

- Add a first phase that reconciles `docs/plans/metadata-enrichment-current-plan.md` with the new
  readiness process.
- Mark which current-plan slices are complete, which remain prerequisites, and which are replaced by
  the readiness plan.
- Keep Gate A and Gate B named. The readiness plan should build on them:
  - Gate A: source application authority.
  - Gate B: configured split-agent path works before destructive tool cleanup.
  - Readiness gates: quality and scale certification after those authority/mechanical gates.

This prevents agents from skipping the source-authority work because a later readiness plan looks
more exciting.

### 2. Split "Reviewed Records" Into Authority Classes Before Making Dev/Test Sets

The plan says to split about 40 of the 150 reviewed records into TEST and use about 110 as DEV.
That is too blunt because the 150 include 67 approved and 83 feedback records with different
authority.

Recommended change:

- Create a split manifest that records, for every row:
  - finding model ID and path;
  - latest review status: approved, feedback, skipped if present;
  - source artifact and payload hash;
  - allowed use: writeback authority, quality gold, candidate guidance, prompt/eval guidance, or
    excluded;
  - split label: dev, heldout_test, certification_seed_excluded, or deferred;
  - stratification fields used to assign the split.
- Do not let feedback-derived expected candidates become gold unless they are human-promoted or
  converted into general guidance plus an explicit eval case.
- Keep the 67 approved-output records as source-writeback authority regardless of whether they land
  in dev or held-out quality splits. Writeback authority and quality-eval membership are separate
  axes.

This aligns the new split with `docs/metadata/enrichment/human-review-and-writeback.md`.

### 3. Make Thresholds Machine-Computable Before Treating Them As Gates

The threshold table is useful, but several metrics are not yet defined tightly enough for a report
to fail or pass them consistently.

Recommended change:

- Define "graded pass" per field:
  - exact;
  - adjacent;
  - defensible;
  - omission;
  - unsupported addition;
  - forbidden value;
  - needs human adjudication.
- For set fields, define the denominator separately for recall, precision, and commission:
  - per expected value;
  - per proposed value;
  - per field per record;
  - or per finding.
- For code fields, define whether display-only differences are ignored, warnings, or scored.
- For `expected_time_course`, define which duration/modifier distances count as adjacent versus
  wrong.
- For "fully acceptable finding", define whether null optional fields can pass when abstention is
  defensible.
- For "no single systematic class > 10% of records", define whether the denominator is all records,
  records with any miss, or field-level misses.

Without these details, the readiness report will become another judgment spreadsheet.

### 4. Calibrate The 10 Percent Flag Cap Before Making It A Hard Readiness Gate

The plan says certification fails if auto-flag rate is greater than 10%. That is operationally
attractive but risky before the auditor is calibrated. A sensitive auditor may correctly flag more
than 10% early, while a weak auditor can pass by missing problems.

Recommended change:

- Separate "quality readiness" from "review workload readiness".
- During certification, report:
  - deterministic flags by severity and field;
  - LLM-auditor flags separately, if used;
  - true-positive rate after human review;
  - false-positive categories;
  - unflagged sampled defects.
- Make the hard scale gate: "review queue after calibrated severity filtering fits the batch review
  cap and sampled unflagged quality clears the bar."
- Keep raw flag rate visible, but do not let it alone decide readiness until severity filters and
  false-positive behavior are known.

This preserves the review cap without rewarding under-flagging.

### 5. Pin Effective Model Behavior, Not Just The Agent Tag

The plan correctly identifies floating model aliases as an attribution problem. The current config
uses `agents.metadata_assign` as a fallback chain, and `MetadataAssignmentReview.model_used` records
the primary effective model string. That is not enough for readiness attribution if fallback models
can run or if provider aliases move.

Recommended change:

- Decide whether certification permits fallbacks. If not, add a certification/run mode that fails
  closed unless the primary model is available.
- Record provider, model ID, reasoning level, fallback policy, package versions, prompt bundle hash,
  and source commit in run metadata.
- Verify the model name and snapshot availability against provider docs before locking it. OpenAI's
  model docs describe snapshots as the mechanism for locking behavior where snapshots are available:
  <https://platform.openai.com/docs/models>.
- Add a small config test that proves `metadata_assign` resolves to the intended certification
  model and reasoning level.

The readiness plan should make a score move attributable to a prompt/tool change, not provider or
fallback drift.

### 6. Prefer A Tool-Repo Readiness Layer Over Expanding Data-Repo Comparison Scripts Forever

The plan points to `../findingmodels-metadata/scripts/metadata_compare_clean_rerun.py` and
`metadata_compare_regression_floor.py`. Those scripts already contain useful grading logic, but the
tool repo owns `pydantic-evals`, field scoring, and package tests.

Recommended change:

- Move reusable grade taxonomy and threshold application into the tool repo, probably under
  `packages/findingmodel-ai/evals/`.
- Let data-repo scripts stay thin: select corpus files, run assignment, and adapt data-repo run
  artifacts into the shared report input shape.
- Use Pydantic Evals report concepts directly where practical. Current Pydantic Evals docs support
  case evaluators for assertions/scores and report evaluators over the full `EvaluationReport`.
- Keep JSON output stable enough for docs and CI gates; render Markdown as a secondary artifact.

This avoids two parallel scoring systems: one in checked-in package evals and one in sibling-repo
scripts.

### 7. Add Selection Rules For Fresh Certification And Scale Batches

The plan says "stratified to corpus distribution", but the corpus distribution needs an explicit
selector so future runs are reproducible.

Recommended change:

- Define the candidate pool:
  - exclude the original reviewed 150;
  - exclude any finding with source-model issues that block metadata review;
  - exclude already enriched/approved source records unless intentionally measuring carry-forward.
- Define stratification columns:
  - entity type if known;
  - existing metadata completeness;
  - body region or anatomy locality;
  - subspecialty;
  - modality signals;
  - source complexity proxies such as attribute count or description length.
- Store the random seed and selected IDs in a manifest.
- Require a new fresh batch after a failed certification batch is folded into dev data.

This makes certification repeatable without letting it become tuneable.

### 8. Define Human Review Forms For Certification Versus Scale

Certification asks for full human review; scale asks for all flagged plus a sample. Those are
different review jobs and should not use an ambiguous single "approved/feedback" vocabulary.

Recommended change:

- Certification review should capture:
  - accept;
  - defensible alternative;
  - field-level correction;
  - high-cost unsupported addition;
  - source-model issue;
  - reviewer uncertainty/adjudication needed.
- Scale review should capture:
  - approve for writeback;
  - reject/defer;
  - source issue;
  - tooling issue class;
  - whether an auto-flag was useful.
- Writeback still requires source-apply manifest membership; certification acceptance alone should
  not bypass the approved-output path.

This keeps readiness measurement from accidentally becoming source authority.

## Proposed Revised Sequence

1. **Write and reconcile the active readiness plan.**
   Add `docs/metadata/enrichment/readiness-gates.md`, update
   `docs/plans/metadata-enrichment-current-plan.md`, and mark this review document as incorporated
   or superseded once decisions land.
2. **Close prerequisite authority/mechanical gates.**
   Re-run or document Gate A and Gate B status, including two-repo commit/source pins and current
   approved-output authority.
3. **Freeze evaluation data authority.**
   Build the reviewed-record manifest with authority class, hashes, split labels, and allowed uses.
4. **Implement the readiness report.**
   Apply hard gates, per-field quality floors, commission metrics, failure-class rollups, and stable
   JSON/Markdown output.
5. **Port and expand the regression floor.**
   Replace strict exact matching with graded floor expectations and add missing PET/molecular,
   index-code, and anatomic-location coverage.
6. **Run dev iteration under a pinned certification configuration.**
   Tune only from dev failure classes; check held-out only at candidate-ready milestones.
7. **Run fresh certification.**
   Select a reproducible unseen batch, fully review it, and either pass to scale or fold failure
   classes into dev and draw a new certification batch.
8. **Run supervised scale batches.**
   Use manifest-backed dry runs, calibrated flag queues plus random samples, approved-output
   application, halt criteria, and batch-level readiness reporting.
9. **Review final documentation.**
   Update `docs/metadata/enrichment/README.md`, `evaluation.md`,
   `human-review-and-writeback.md`, `readiness-gates.md`, the active plan, and user-facing
   changelog entries only where external behavior changes.

## Concrete Edits To The Reviewed Plan

- Rename "Phase 0" to "Phase 1" and add a prerequisite "Phase 0: reconcile current plan, Gate A,
  Gate B, and docs."
- Replace "freeze about 40 of the 150 reviewed records" with "build an authority-aware reviewed
  split manifest from latest effective review records."
- Replace "auto-flag rate <= 10%" with a calibrated workload gate that distinguishes deterministic
  high-severity flags, advisory LLM flags, and sampled unflagged defects.
- Add a "run provenance" section covering model snapshot, fallback policy, package versions, prompt
  hash, ontology/anatomic cache identity, source repo commit, data repo commit, and split manifest
  hash.
- Add "review form/schema changes" to files or process, because certification and scale require
  different review outputs from the original approve/feedback pilot.
- Move the readiness-report implementation target to the tool repo unless there is a specific reason
  the data repo must own it.
- Add acceptance criteria for the readiness report itself:
  - fails on injected forbidden values;
  - passes on defensible regression-floor diffs;
  - separates gates from quality scores;
  - emits stable JSON;
  - renders a concise Markdown summary;
  - proves dev/test disjointness.

## Open Decisions For The User

- Which review statuses are allowed into dev gold, held-out gold, and certification analysis?
- Should certification fail closed if the primary model is unavailable, or allow provider fallback
  with separate reporting?
- Which grade labels count as "defensible" for each field?
- Is the 10% scale-review cap a hard cap per batch, or can high-severity spikes trigger a special
  review batch instead of immediate readiness failure?
- Should fresh certification acceptance create approved-output candidates, or only readiness
  evidence until a separate review/writeback pass?

## Suggested Next Move

Update the active plan in `docs/plans/metadata-enrichment-current-plan.md` to incorporate the
prerequisite reconciliation and the sharper readiness gates above, then create
`docs/metadata/enrichment/readiness-gates.md` as the canonical threshold document before touching
implementation.
