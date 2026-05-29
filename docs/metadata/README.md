# Metadata Reference

Start here for structured finding-model metadata.

This section defines what the metadata fields mean and how to decide values for them. The enrichment
tool uses these standards, but this is not only tool documentation. These rules describe the
metadata we want on finding models.

## Current Documents

- Field reference and decision standards: `docs/metadata/fields.md`
- RSNA subspecialty policy: `docs/metadata/subspecialties.md`
- Enrichment process: `docs/metadata/enrichment/README.md`
- Current cleanup plan: `docs/plans/metadata-enrichment-current-plan.md`

## Authority

Human review is authoritative. Generated outputs, extracted candidate hints, sub-agent triage, and
old source diffs are evidence only.

When artifacts conflict, prefer the latest explicit human adjudication over earlier extracted hints.
For example, earlier extracted candidate hints suggested etiologies for transudative pleural
effusion, but later adjudication settled that transudative pleural effusion does not carry enough
information to assign etiology reliably.

## Field Rules

Only `entity_type` is required for enrichment output. Other metadata fields may be null when the
finding does not justify an assignment.

Null can mean:

- not applicable;
- genuinely indeterminate from the finding;
- dependent on an unresolved underlying cause;
- insufficiently supported by the source record.

Unsupported additions are usually more harmful than omissions because they create false grouping,
false relationships, or false source-code confidence.

## Relationship To Enrichment

The enrichment tool proposes values for these fields. Source commits must still be based on human
review or a separately justified field-limited repair rule. The current approved data baseline is
67 human-approved metadata records plus 11 index-code display-only repairs.

Read the enrichment process docs when you need review evidence, eval scoring, prompt guidance, or
source writeback rules. Read the field reference when you need to know what the metadata should
mean.
