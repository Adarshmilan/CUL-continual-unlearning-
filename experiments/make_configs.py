"""Writes every paper experiment as a config file, split between the two laptops.

    python experiments/make_configs.py
        -> experiments/adarsh/configs/*.json   (CIFAR-100: unlearning methods, seeds, class level, causes)
        -> experiments/arpan/configs/*.json    (Tiny-ImageNet-100, the ER learner, the long stream)

Every run saves into results/paper/ (results/paper/<name>.npz, _summary.json, _model.pt), so the two
results folders can simply be copied together. Names: <group>_<method>_<stream>_s<seed>.

Methods (all with the "simple fixes": buffer purge + head reset + logit mask, program 17):
    ref     retrain reference on the same stream (forgotten items never learned)
    bt      BadTeacher (program 09)
    ng      NegGrad+ (gradient ascent + retain CE)
    rl      random labels
    scrub   SCRUB (simplified)
    feat    BadTeacher + feature unlearning (Phase F)
    layer   BadTeacher + the weight-level layer guard (Phase F)
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
base = json.loads((ROOT / "results/configs/relapse_guard.json").read_text())
for k in ("name", "stream", "seed", "guard"):
    base.pop(k)
FIXES = dict(base, purge_buffer=True, head_reset=True, mask_forgotten_logits=True, guard="none")

METHODS = {
    "ref": dict(reference=True),
    "bt": dict(unlearn_method="badteacher"),
    "ng": dict(unlearn_method="neggrad"),
    "rl": dict(unlearn_method="randlabel"),
    "scrub": dict(unlearn_method="scrub"),
    "feat": dict(unlearn_method="badteacher", feature_steps=50),
    "layer": dict(unlearn_method="badteacher", guard_layers="layer3,layer4"),
}
SEEDS = [0, 1, 2]

# program 15's wolf experiment: forget wolf, then learn a similar or a dissimilar task
TASK_A = ["wolf", "camel", "cattle", "chimpanzee", "elephant", "kangaroo", "bus", "bicycle", "orchid", "apple"]
SIMILAR = ["bear", "leopard", "lion", "tiger", "fox", "porcupine", "possum", "raccoon", "skunk", "otter"]
DISSIMILAR = ["bed", "chair", "couch", "table", "wardrobe", "lawn_mower", "rocket", "streetcar", "tank", "tractor"]


def stream_of(stream, seed):
    """(stream file, short name) for a stream family and a seed."""
    if stream == "worst":
        return "handmade_worst", "worst"
    return f"{stream}_seed{seed}", stream


def cfg(group, method, stream, seed, extra=None):
    file, short = stream_of(stream, seed)
    c = dict(FIXES, **METHODS[method], **(extra or {}), stream=file, seed=seed,
             name=f"paper/{group}_{method}_{short}_s{seed}")
    return c


def causes(group, seed, later, purge, ref=False):
    """Program 15 with the runner: L task A, U wolf, L similar/dissimilar task; buffer purged or kept."""
    c = dict(FIXES, **METHODS["ref" if ref else "bt"], seed=seed, probe_blocks=True, level="class",
             tasks_by_name={"0": TASK_A, "1": SIMILAR if later == "similar" else DISSIMILAR},
             requests=[["L", 0], ["U", "wolf"], ["L", 1]], purge_buffer=purge, stream="custom",
             name=f"paper/{group}_{'ref' if ref else ('purged' if purge else 'kept')}_{later}_s{seed}")
    return c


def adarsh():
    out = []
    # A1: where does the forgotten task hide, for every unlearning method? (worst case, every-block probe)
    for s in SEEDS:
        for m in ["ref", "bt", "ng", "rl", "scrub", "feat"]:
            out.append(("A1", cfg("c100", m, "worst", s, dict(probe_blocks=True))))
    # A2: main table, 15-request stream, every unlearning method (every-block probe too)
    for s in SEEDS:
        for m in ["ref", "bt", "ng", "rl", "scrub"]:
            out.append(("A2", cfg("c100", m, "task15", s, dict(probe_blocks=True))))
    # A3: class level, 30 requests, seeds 1 and 2 (seed 0 was run in Phase F)
    for s in [1, 2]:
        for m in ["ref", "bt", "layer"]:
            out.append(("A3", cfg("c100", m, "class30", s)))
    # A4: causes of relapse with a retrain reference (program 15 + reference)
    for s in SEEDS:
        for later in ["similar", "dissimilar"]:
            out.append(("A4", causes("cause", s, later, purge=True, ref=True)))
            out.append(("A4", causes("cause", s, later, purge=True)))
            out.append(("A4", causes("cause", s, later, purge=False)))
    return out


def arpan():
    out = []
    tiny = dict(dataset="tiny100", probe_blocks=True)
    # B1: second dataset, worst case
    for s in SEEDS:
        for m in ["ref", "bt", "ng", "scrub"]:
            out.append(("B1", cfg("tiny", m, "worst", s, tiny)))
    # B2: second dataset, 15-request stream
    for s in SEEDS:
        for m in ["ref", "bt", "layer"]:
            out.append(("B2", cfg("tiny", m, "task15", s, tiny)))
    # B3: second learner: plain experience replay (no logit replay: alpha=0, label replay beta=1)
    er = dict(alpha=0.0, beta=1.0, probe_blocks=True)
    for s in SEEDS:
        for m in ["ref", "bt"]:
            out.append(("B3", cfg("er", m, "worst", s, er)))
            out.append(("B3", cfg("er", m, "task15", s, er)))
    # B4: long stream, 100 class-level requests
    for m in ["ref", "bt"]:
        out.append(("B4", cfg("c100", m, "class100", 0)))
    return out


if __name__ == "__main__":
    for who, runs in [("adarsh", adarsh()), ("arpan", arpan())]:
        d = ROOT / "experiments" / who / "configs"
        d.mkdir(parents=True, exist_ok=True)
        for f in d.glob("*.json"):
            f.unlink()
        for i, (group, c) in enumerate(runs):
            (d / f"{i:03d}_{group}_{c['name'].split('/')[-1]}.json").write_text(json.dumps(c, indent=1))
        print(f"{who}: {len(runs)} runs -> {d}")
