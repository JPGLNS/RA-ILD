#!/usr/bin/env python3
"""Run three independently tuned TCR-only ablations; do not touch frozen results."""
from __future__ import annotations
import argparse, concurrent.futures, json, os, subprocess, sys, time
from pathlib import Path
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = ROOT / "TRB/set/configs/layered_feature_experiments/tcr_only_ablation"
MANIFEST = CONFIG_ROOT / "TCR_ONLY_ABLATION_MANIFEST.json"
SCRIPT = ROOT / "TRB/set/scripts/run_06_unified_ml.py"
REGISTRY = ROOT / "TRB/set/configs/feature_framework/trb_feature_registry_v3_179_top10000.yaml"
META = ROOT / "TRB/set/data_versions/trb_v3_179_top10000/ml_compat/total"
CACHE = ROOT / "TRB/result/feature_framework/cache/step02/top10000"
RESULT = ROOT / "TRB/result/feature_framework/layered_feature_experiments"
OUT = RESULT / "tcr_only_ablation"
LOG = RESULT / "logs/tcr_only_ablation"
STATUS = RESULT / "status/tcr_only_ablation"

def entries_verified():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entries = data["entries"]
    if len(entries) != 3 or {e["cohort"] for e in entries} != {"total", "pbmc", "buffycoat"}:
        raise ValueError("Expected exactly three ablation cohorts")
    if data["outer_repeats"] != 100 or data["inner_folds"] != 5:
        raise ValueError("Repeat/fold contract mismatch")
    for e in entries:
        cfg = yaml.safe_load((ROOT / e["config"]).read_text(encoding="utf-8"))
        f = cfg["features"]
        if f["clinical_groups"] != []:
            raise ValueError(f"Clinical predictors remain enabled: {e['config']}")
        if f["static_tcr_groups"] != e["feature_groups"] or not e["feature_groups"]:
            raise ValueError(f"TCR group mismatch: {e['config']}")
        if f["kmer"]["enabled"] or f["enriched_dictionary"]["enabled"]:
            raise ValueError("Dynamic features unexpectedly enabled")
        if cfg["model"]["algorithm"] != "elastic_net" or cfg["cohort"]["scope"] != e["cohort"]:
            raise ValueError("Algorithm/cohort mismatch")
        if cfg["validation"]["repeats"] != 100 or cfg["validation"]["folds"] != 5:
            raise ValueError("Validation contract mismatch")
        if cfg["validation"]["split_assignments"] != e["split_assignments"]:
            raise ValueError("Frozen split path mismatch")
        if not (ROOT / e["split_assignments"]).is_file():
            raise FileNotFoundError(e["split_assignments"])
    return entries

def run_one(e, repeat):
    cohort = e["cohort"]
    tag = "full100" if repeat is None else f"smoke_repeat_{repeat:03d}"
    out = ROOT / e["output_dir"]
    if repeat is not None:
        out = out / tag
    log = LOG / f"{cohort}.{tag}.log"
    status = STATUS / f"{cohort}.{tag}.json"
    key = f"{cohort}__{tag}"
    if status.is_file():
        try:
            previous = json.loads(status.read_text())
            if previous.get("status") == "COMPLETE" and (out / "06_outer_metrics.csv").is_file():
                return {"key": key, "status": "SKIPPED_COMPLETE"}
        except (OSError, ValueError):
            pass
    if out.exists() and any(
        not (p.is_dir() and p.name.startswith("smoke_repeat_"))
        for p in out.iterdir()
    ):
        return {"key": key, "status": "BLOCKED_OUTPUT_EXISTS", "output": str(out)}
    required = [SCRIPT, REGISTRY, META / "metadata_train_70.csv",
                META / "metadata_test_30.csv", CACHE / "train/02_core_sample_features.csv",
                CACHE / "test/02_core_sample_features.csv"]
    for p in required:
        if not p.is_file():
            raise FileNotFoundError(p)
    cmd = [sys.executable, str(SCRIPT), "--config", str(ROOT / e["config"]),
           "--train-step02-dir", str(CACHE / "train"), "--test-step02-dir", str(CACHE / "test"),
           "--train-metadata", str(META / "metadata_train_70.csv"),
           "--test-metadata", str(META / "metadata_test_30.csv"),
           "--registry", str(REGISTRY), "--output-dir", str(out), "--workers", "1"]
    if repeat is not None:
        cmd += ["--repeat", str(repeat)]
    log.parent.mkdir(parents=True, exist_ok=True)
    status.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[name] = "1"
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as fh:
        code = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    result = {"key": key, "status": "COMPLETE" if code == 0 else "FAILED",
              "returncode": code, "runtime_seconds": round(time.monotonic() - started, 2),
              "output": str(out), "log": str(log)}
    if code == 0:
        metrics = pd.read_csv(out / "06_outer_metrics.csv")
        expected = list(range(1, 101)) if repeat is None else [repeat]
        if sorted(metrics["outer_repeat"].astype(int).tolist()) != expected:
            result["status"] = "FAILED_AUDIT"
    status.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", choices=["total", "pbmc", "buffycoat"])
    ap.add_argument("--repeat", type=int, help="Single frozen repeat for smoke test")
    ap.add_argument("--max-experiments", type=int, default=3)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    if args.repeat is not None and not 1 <= args.repeat <= 100:
        ap.error("--repeat must be 1..100")
    if not 1 <= args.max_experiments <= 3:
        ap.error("--max-experiments must be 1..3")
    entries = entries_verified()
    if args.cohort:
        entries = [e for e in entries if e["cohort"] == args.cohort]
    print(f"Configs={len(entries)}; repeats={args.repeat or 'all 100'}; concurrency={args.max_experiments}")
    for e in entries:
        print(f"{e['cohort']}: {e['feature_groups']}; clinical=[]; split={e['split_assignments']}")
    if not args.execute:
        print("DRY-RUN PASS: no training was started")
        return 0
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_experiments) as pool:
        futures = [pool.submit(run_one, e, args.repeat) for e in entries]
        for fut in concurrent.futures.as_completed(futures):
            r = fut.result()
            results.append(r)
            print(f"[{r['status']}] {r['key']}", flush=True)
    report = OUT / ("smoke_batch_status.csv" if args.repeat is not None else "batch_status.csv")
    report.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(report, index=False)
    print(f"Status report: {report}")
    return 1 if any(r["status"] not in {"COMPLETE", "SKIPPED_COMPLETE"} for r in results) else 0

if __name__ == "__main__":
    raise SystemExit(main())
