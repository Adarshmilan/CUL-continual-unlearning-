"""Phase F analysis: every method vs the retrain reference on the same stream.

For each run and each forgotten item we compare its probe curve with the reference run's
curve on the same rows (the reference never learned the item):
    excess after U = probe(run) - probe(ref) right after the U request
    excess end     = the same at the end of the stream
    excess rise    = how much the gap grew from after U to the end  (> 0 = feature relapse
                     beyond what normal learning gives a never-seen class)

Usage:  python programs/phase_f_analysis.py      (prints tables, writes results/F_table.json)
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.clu import load_result  # noqa: E402

# runs made before Phase F under other names (same configs as Phase F's)
ALIASES = {
    ("fixes", "task15_seed0"): "17_simple_fixes",
    ("proj", "task15_seed0"): "19_projection",
    ("fixes", "handmade_worst"): "20_fixes_handmade_worst",
    ("proj", "handmade_worst"): "20_final_handmade_worst",
}
METHODS = ["fixes", "proj", "layer", "feat", "feat_layer", "layer_proj"]


def get(method, stream):
    return load_result(ALIASES.get((method, stream), f"F_{method}_{stream}"))


def excess(run, ref):
    items = list(np.asarray(run["items"]))
    reqs = [str(s) for s in run["requests"]]
    out = {"after": [], "end": [], "rise": [], "peak_rise": []}
    for r, s in enumerate(reqs):
        if s[0] != "U":
            continue
        j = items.index(int(s[1:]))
        d = run["probe"][r:, j] - ref["probe"][r:, j]
        out["after"].append(d[0])
        out["end"].append(d[-1])
        out["rise"].append(d[-1] - d[0])
        out["peak_rise"].append(np.nanmax(d) - d[0])
    return {k: float(np.mean(v)) for k, v in out.items()}


def row(run, ref):
    s = run["summary"]
    r = {"RA": s["til"]["RA"], "FA": s["til"]["FA"], "Spill": s["til"]["Spill"], "Relapse": s["til"]["Relapse"],
         "RA_cil": s["cil"]["RA"], "probe_after": s["probe"]["after_U"], "probe_end": s["probe"]["end"],
         "minutes": float(run["minutes"])}
    if ref is not None:
        e = excess(run, ref)
        r.update({"ref_end": float(np.mean([ref["probe"][-1, list(ref["items"]).index(int(x[1:]))]
                                            for x in map(str, run["requests"]) if x[0] == "U"])),
                  **{f"excess_{k}": v for k, v in e.items()}})
    return r


COLS = ["RA", "FA", "Relapse", "RA_cil", "probe_after", "probe_end", "ref_end", "excess_after", "excess_end", "excess_rise"]


def fmt(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return "     -     "
    if len(vals) == 1:
        return f"{vals[0]*100:6.1f}     "
    return f"{np.mean(vals)*100:5.1f}±{np.std(vals)*100:3.1f}"


def table(title, streams):
    print(f"\n=== {title}  ({', '.join(streams)})")
    print(f"{'method':>11} {'n':>2} | " + " ".join(f"{c:>11}" for c in COLS))
    out = {}
    for m in METHODS:
        rows = []
        for st in streams:
            run, ref = get(m, st), get("ref", st)
            if run is not None:
                rows.append(row(run, ref))
        if not rows:
            continue
        out[m] = rows
        print(f"{m:>11} {len(rows):>2} | " + " ".join(f"{fmt([r.get(c) for r in rows]):>11}" for c in COLS))
    return out


if __name__ == "__main__":
    allt = {
        "task15 (3 seeds)": table("task-level, 15 requests", [f"task15_seed{s}" for s in range(3)]),
        "worst": table("worst case: forget task 0, then learn 9 tasks (3 run seeds)",
                       ["handmade_worst", "handmade_worst_s1", "handmade_worst_s2"]),
        "class30": table("class-level, 30 requests", ["class30_seed0"]),
    }
    print("\nexcess = probe(method) - probe(retrain reference) on forgotten items; rise > 0 means feature relapse")
    (ROOT / "results" / "F_table.json").write_text(json.dumps(allt, indent=1))
