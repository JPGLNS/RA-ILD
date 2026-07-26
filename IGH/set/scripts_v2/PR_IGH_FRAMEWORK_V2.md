# IGH Framework V2 — Pull Request Delivery Guide

## Intended source-only commits

The IGH upgrade branch should contain the IGH-only Batch 00–06 source,
configuration, tests, documentation, manifests, and the server-generated
`IGH/set/FRAMEWORK_V2_COMPLETE.json` marker.

Do not include training bundles, split outputs, experiment task directories,
sparse matrices, catalogs, model results, ZIP deliveries, `__pycache__`, or the
unrelated `TRB/set/scripts/analyze_result_coefficient.R` file.

## Merge order

1. merge the TRB framework branch into `main`;
2. update the IGH delivery branch by rebasing or cherry-picking the IGH-only
   commits onto the updated `main`;
3. verify the final diff is restricted to `IGH/` plus any explicitly approved
   shared ignore-rule changes;
4. rerun the quick release checker after conflict resolution;
5. open the IGH PR without claiming that repeated holdouts are an independent
   validation set or that a final model has been automatically selected.

## Suggested PR title

`Add configurable leakage-controlled IGH repeated-holdout framework`

## Suggested PR summary

- modular IGH V2 configuration and scientific engine;
- configurable models and additional feature tables;
- frozen patient-level repeated holdouts;
- partition-local public references and exact fitting LOO;
- automatic inner-CV alpha/lambda and inner-OOF threshold selection;
- completed 3 × 120/49 training tasks and descriptive aggregation;
- release checks and provenance marker;
- no deployable final model selected in this PR.
