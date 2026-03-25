# Test Suite Parallelization Follow-up

## Status

Deferred for a later maintenance pass.

## Why this is worth doing

Our package test tasks in [Taskfile.yml](/Users/talkasab/repos/findingmodel-enrich/Taskfile.yml) currently run sequentially:

- `test`
- `test-full`

For the normal non-callout path, package-level parallelism should reduce wall-clock time with low risk.

## Current callout profile

The package suites are not all equal:

- `test:findingmodel-ai-full` has the heaviest real external callouts and model/API usage.
- `test:findingmodel-full` has a smaller number of real external callouts, including real embedding and remote-download tests.
- `test:oidm-common-full` has a small amount of real external callout work for embedding integration.
- `test:anatomic` and `test:maintenance` are effectively local-only in normal operation.

## Recommended future approach

1. Parallelize the non-callout aggregate task first.
2. Keep `test-full` sequential initially, or only partially parallelize it after measuring stability.
3. Use Task `deps:`-based aggregation rather than sequential `cmds:` where practical.
4. Recheck for shared writable artifacts before enabling parallel full-suite execution.

## Concrete next step when we pick this up

Update [Taskfile.yml](/Users/talkasab/repos/findingmodel-enrich/Taskfile.yml) so:

- `test` runs package suites in parallel
- `test-full` stays conservative until we verify the callout-heavy packages behave well under concurrent execution

