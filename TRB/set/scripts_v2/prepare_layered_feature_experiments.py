#!/usr/bin/env python3
"""Generate Layer-1/Layer-2 manifests; does not launch model training."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"src"))
from ra_ild_trb.layered_experiments import build_layer1_manifest,rank_layer1_results,build_layer2_manifest

def write_rows(path,rows):
    if not rows: raise ValueError(f"refusing empty manifest: {path}")
    path.parent.mkdir(parents=True,exist_ok=True)
    cols=list(rows[0])
    with path.open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=cols); w.writeheader()
        for row in rows:
            w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-root",type=Path,default=ROOT)
    p.add_argument("--top-n",type=int,default=5)
    p.add_argument("--layer1-results",type=Path,default=None,
        help="complete 100-repeat aggregate CSV; required to generate Layer 2")
    p.add_argument("--apply",action="store_true")
    a=p.parse_args(); root=a.repository_root.resolve()
    l1=build_layer1_manifest(); l2=None; selected=None
    if a.layer1_results:
        import pandas as pd
        selected=rank_layer1_results(pd.read_csv(a.layer1_results).to_dict("records"),top_n=a.top_n)
        l2=build_layer2_manifest(selected)
    print("TRB V3 Layered Feature Experiment Manifests")
    print(f"Layer 1: {len(l1)} = 3 cohorts × Elastic Net × 127 Core83 combinations")
    print(f"Layer 1 outer task evaluations: {len(l1)*100}")
    if l2 is None: print("Layer 2: deferred until complete Layer-1 aggregate is supplied")
    else: print(f"Layer 2: {len(l2)} = {len(selected)} promoted combinations × 4 paired states")
    print("RA-hit: prevalence >=20%; between-class prevalence delta >=10 percentage points")
    if not a.apply: print("DRY-RUN: no files written; add --apply to write manifests"); return 0
    out=root/"TRB/set/configs/layered_feature_experiments"
    write_rows(out/"LAYER1_127_COMBINATIONS_3COHORT_ELASTIC_NET.csv",l1)
    if l2 is not None:
        write_rows(out/"LAYER1_PROMOTED_TOPN.csv",selected)
        write_rows(out/"LAYER2_INCREMENTAL_MANIFEST.csv",l2)
    print(f"Wrote manifests under {out}")
    return 0
if __name__=="__main__": raise SystemExit(main())
