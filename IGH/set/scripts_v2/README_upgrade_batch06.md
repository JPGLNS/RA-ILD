# IGH Scheme Upgrade Batch 06 — Final Regression and Release

Batch 06 is the engineering closeout for the IGH V2 upgrade. It does not select
or fit a final deployable model. The current scientific result remains a
three-split repeated-holdout comparison with descriptive rankings only.

## Release gates

The unified checker covers:

- the existing 120 IGH unit tests and Batch 02–06 focused tests;
- the frozen historical IGH baseline and independent-test provenance;
- public-reference, preprocessing, Elastic Net, specification, and nested-CV regressions;
- frozen split and training-bundle hashes;
- all three completed 120/49 task trees;
- the completed 9-metric-row / 441-prediction-row aggregation;
- an IGH-only runtime path scan and a synthetic two-NT-to-one-AA SHM contract;
- source-only Git cleanliness and upstream synchronization before release marking.

Quick validation is used before the source commit. Full validation reruns the
complete historical nested-CV regression and is required before writing the
completion marker.

## Scientific interpretation

`FRAMEWORK_V2_COMPLETE.json` records framework completion, not selection of a
scientific winner. `final_model.status` remains `not_selected`. A deployable
full-169-patient model requires a separately approved selection protocol.

## Git sequence

1. install and run quick acceptance;
2. commit and push Batch 06 source files;
3. run full validation from the clean synchronized branch and write the marker;
4. commit and push only `IGH/set/FRAMEWORK_V2_COMPLETE.json`;
5. merge the TRB framework PR first, then rebase or cherry-pick the IGH-only
   commits onto the updated `main` before opening the IGH PR.
