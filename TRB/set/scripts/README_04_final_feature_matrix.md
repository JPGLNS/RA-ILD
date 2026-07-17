# 04 final feature matrix

## Correct expected dimensions

Because `patient` is retained as a trace column, the expected dimensions are:

```text
04_train_base_feature_matrix.csv          123 × 1103
04_test_base_feature_matrix.csv            51 × 1103
04_train_descriptive_feature_matrix.csv   123 × 1127
04_test_descriptive_feature_matrix.csv     51 × 1127
04_test_final_feature_matrix.csv           51 × 1121
```

The earlier 1102 / 1126 / 1120 expectations correspond to omitting `patient`.

No fixed `04_train_final_feature_matrix.csv` is generated.

## Install

Copy:

```text
build_04_final_feature_matrix.py
run_04_build_feature_matrices.sh
```

to:

```text
/data/users/chenhaisheng/RA-ILD/TRB/set/scripts/
```

Then:

```bash
chmod +x set/scripts/build_04_final_feature_matrix.py
chmod +x set/scripts/run_04_build_feature_matrices.sh
python3 -m py_compile set/scripts/build_04_final_feature_matrix.py
```

## Run

```bash
cd /data/users/chenhaisheng/RA-ILD/TRB
bash set/scripts/run_04_build_feature_matrices.sh
```

## Validation performed by the script

- Train/test samples must be disjoint.
- All IDs must be unique.
- All merges use `validate="one_to_one"`.
- Train/test step-02 columns and order must match exactly.
- Train/test base columns and order must match exactly.
- Train/test descriptive columns and order must match exactly.
- Step-02 and step-03 `aa_clone_number` must agree.
- Missing values and unexpected duplicated columns are rejected.
- Static train descriptive public features are not added to the train base matrix.
- Test reference-public features are added only to the test final matrix.
