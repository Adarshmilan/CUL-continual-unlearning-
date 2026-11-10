"""Every table of the paper, from results/paper/ (both laptops' results copied together).

    python experiments/paper_tables.py            -> prints all tables, writes results/paper/tables.json

Missing runs are simply left out (the "n" column says how many seeds a number is based on), so
this can be run any time to see how far the experiments are.

Key quantity: EXCESS = probe(method) - probe(retrain reference), for the same forgotten item on
the same request row. 0 = as forgotten as a model that never learned it.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.clu import load_result  # noqa: E402
from common.features import BLOCKS  # noqa: E402

SEEDS = [0, 1, 2]
NAMES = {"ref": "retrain reference", "bt": "BadTeacher", "ng": "NegGrad+", "rl": "random labels",
         "scrub": "SCRUB", "feat": "BadTeacher + feature unlearning", "layer": "BadTeacher + layer guard"}
OUT = {}


def get(group, method, stream, seed):
    return load_result(f"paper/{group}_{method}_{stream}_s{seed}")


def ms(vals, pct=True):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], dtype=float) * (100 if pct else 1)
    if len(v) == 0:
        return "    -     "
    return f"{v.mean():5.1f}±{v.std():3.1f}" if len(v) > 1 else f"{v.mean():5.1f}     "


def u_rows(res):
    items = [int(i) for i in np.asarray(res["items"])]
    return [(r, items.index(int(s[1:]))) for r, s in enumerate(map(str, res["requests"])) if s[0] == "U"]


def block_excess(run, ref, offset):
    """Per block: mean over forgotten items of probe - ref probe, `offset` rows after each U (-1 = before U)."""
    vals = []
    for r, j in u_rows(run):
        if offset < 0:
            a, b = run["probe_before_blocks"][j], ref["probe_before_blocks"][j]
        else:
            rr = min(r + offset, len(run["requests"]) - 1)
            a, b = run["probe_blocks"][rr, j], ref["probe_blocks"][rr, j]
        vals.append(a - b)
    return np.nanmean(vals, axis=0)


def block_raw(run, offset):
    vals = []
    for r, j in u_rows(run):
        vals.append(run["probe_before_blocks"][j] if offset < 0 else
                    run["probe_blocks"][min(r + offset, len(run["requests"]) - 1), j])
    return np.nanmean(vals, axis=0)


# ---------------------------------------------------------------- T1: where the forgotten task hides
def table_blocks(title, group, stream, methods):
    print(f"\n=== {title}")
    print("probe on the forgotten task at each block; rows: never-trained-on-it reference, then EXCESS over it")
    when = [("right after U", 0), ("after the next task", 1), ("end of stream", 99)]
    out = {}
    for label, off in when:
        print(f"\n  {label}")
        print(f"  {'method':>32} {'n':>2} | " + " ".join(f"{b:>11}" for b in BLOCKS))
        refs = [get(group, "ref", stream, s) for s in SEEDS]
        raw = [block_raw(r, off) for r in refs if r is not None and "probe_blocks" in r]
        if raw:
            print(f"  {'reference (raw probe)':>32} {len(raw):>2} | " + " ".join(f"{ms([x[b] for x in raw]):>11}" for b in range(4)))
        for m in methods:
            ex = []
            for s in SEEDS:
                run, ref = get(group, m, stream, s), get(group, "ref", stream, s)
                if run is not None and ref is not None and "probe_blocks" in run:
                    ex.append(block_excess(run, ref, off))
            if ex:
                print(f"  {NAMES[m]:>32} {len(ex):>2} | " + " ".join(f"{ms([x[b] for x in ex]):>11}" for b in range(4)))
                out.setdefault(label, {})[m] = [[float(v) for v in x] for x in ex]
    OUT[title] = out


# ---------------------------------------------------------------- T2: main metrics table
def table_main(title, group, stream, methods, aliases=None):
    print(f"\n=== {title}")
    cols = ["RA", "FA", "Spill", "Relapse", "RA_cil", "probe end", "excess after U", "excess end"]
    print(f"  {'method':>32} {'n':>2} | " + " ".join(f"{c:>14}" for c in cols))
    out = {}
    for m in methods:
        rows = []
        for s in SEEDS:
            run = (aliases or {}).get((m, s)) or get(group, m, stream, s)
            ref = (aliases or {}).get(("ref", s)) or get(group, "ref", stream, s)
            if isinstance(run, str):
                run = load_result(run)
            if isinstance(ref, str):
                ref = load_result(ref)
            if run is None:
                continue
            S = run["summary"]
            row = {"RA": S["til"]["RA"], "FA": S["til"]["FA"], "Spill": S["til"]["Spill"],
                   "Relapse": S["til"]["Relapse"], "RA_cil": S["cil"]["RA"], "probe end": S["probe"]["end"] if "probe" in S else None}
            if ref is not None and m != "ref":
                d0, d1 = [], []
                for r, j in u_rows(run):
                    d0.append(run["probe"][r, j] - ref["probe"][r, j])
                    d1.append(run["probe"][-1, j] - ref["probe"][-1, j])
                row["excess after U"], row["excess end"] = float(np.nanmean(d0)), float(np.nanmean(d1))
            rows.append(row)
        if rows:
            print(f"  {NAMES[m]:>32} {len(rows):>2} | " + " ".join(f"{ms([r.get(c) for r in rows]):>14}" for c in cols))
            out[m] = rows
    OUT[title] = out


# ---------------------------------------------------------------- T4: causes
def table_causes():
    title = "Causes of relapse (wolf, CIFAR-100): buffer kept vs purged, similar vs dissimilar later task"
    print(f"\n=== {title}")
    print("EXCESS over the reference (never learned wolf), layer3 and layer4 probe, after U and after the later task")
    print(f"  {'condition':>22} {'n':>2} | {'L3 after U':>11} {'L3 after L':>11} {'L4 after U':>11} {'L4 after L':>11}")
    out = {}
    for later in ["similar", "dissimilar"]:
        for cond in ["purged", "kept"]:
            ex = []
            for s in SEEDS:
                run = load_result(f"paper/cause_{cond}_{later}_s{s}")
                ref = load_result(f"paper/cause_ref_{later}_s{s}")
                if run is not None and ref is not None:
                    ex.append(np.r_[run["probe_blocks"][1, 0] - ref["probe_blocks"][1, 0],
                                    run["probe_blocks"][2, 0] - ref["probe_blocks"][2, 0]])
            if ex:
                v = [ms([x[k] for x in ex]) for k in (2, 6, 3, 7)]
                print(f"  {cond + ', ' + later:>22} {len(ex):>2} | " + " ".join(f"{x:>11}" for x in v))
                out[f"{cond}_{later}"] = [x.tolist() for x in ex]
    OUT[title] = out


if __name__ == "__main__":
    P = "paper/phase_f/"
    # Phase F runs (BadTeacher + fixes, guards) used for the guard claim
    guards = {("bt", 0): P + "17_simple_fixes", ("proj", 0): P + "19_projection", ("layer", 0): P + "F_layer_task15_seed0",
              ("ref", 0): P + "F_ref_task15_seed0"}
    for s in (1, 2):
        guards.update({("bt", s): P + f"F_fixes_task15_seed{s}", ("proj", s): P + f"F_proj_task15_seed{s}",
                       ("layer", s): P + f"F_layer_task15_seed{s}", ("ref", s): P + f"F_ref_task15_seed{s}"})
    NAMES["proj"] = "BadTeacher + projection guard"

    table_blocks("T1a CIFAR-100, worst case: where the forgotten task hides", "c100", "worst",
                 ["bt", "ng", "rl", "scrub", "feat"])
    table_blocks("T1b Tiny-ImageNet-100, worst case", "tiny", "worst", ["bt", "ng", "scrub"])
    table_blocks("T1c ER learner (CIFAR-100), worst case", "er", "worst", ["bt"])
    table_blocks("T1d CIFAR-100, 15 requests: block probes", "c100", "task15", ["bt", "ng", "rl", "scrub"])
    table_main("T2a CIFAR-100, 15 requests: all unlearning methods", "c100", "task15", ["bt", "ng", "rl", "scrub"])
    table_main("T2b CIFAR-100, 15 requests: guards (Phase F)", None, None, ["bt", "proj", "layer"], aliases=guards)
    table_main("T2c Tiny-ImageNet-100, 15 requests", "tiny", "task15", ["bt", "layer"])
    table_main("T2d ER learner, 15 requests", "er", "task15", ["bt"])
    table_main("T2e CIFAR-100, worst case: all methods", "c100", "worst", ["bt", "ng", "rl", "scrub", "feat"])
    cls = {("bt", 0): P + "F_fixes_class30_seed0", ("layer", 0): P + "F_layer_class30_seed0", ("ref", 0): P + "F_ref_class30_seed0"}
    table_main("T3 CIFAR-100, class level, 30 requests", "c100", "class30", ["bt", "layer"], aliases=cls)
    table_main("T5 CIFAR-100, long stream: 100 class-level requests (1 seed)", "c100", "class100", ["bt"])
    table_causes()
    (ROOT / "results" / "paper" / "tables.json").write_text(json.dumps(OUT, indent=1))
    print("\nwrote results/paper/tables.json")
