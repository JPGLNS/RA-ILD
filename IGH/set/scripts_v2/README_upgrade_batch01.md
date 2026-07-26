# IGH Scheme Upgrade Batch 01

## Title

IGH modular foundation and historical baseline configuration

## Installation

Run from the repository root on `feature/igh-scheme-upgrade-v1`:

```bash
python3 apply_IGH_framework_upgrade_batch01.py \
  --repository-root /data/users/chenhaisheng/RA-ILD \
  --dry-run

python3 apply_IGH_framework_upgrade_batch01.py \
  --repository-root /data/users/chenhaisheng/RA-ILD
```

Then run the structural, unit, and configuration checks described in the main
response and in `IGH/set/README_V2.md`.

Batch 02 will add the user-facing model replacement and additional feature-table
workflow. Batch 03 will add configurable frozen repeated holdout.
