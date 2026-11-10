"""Phase F diagnostic: where does the forgotten task hide right after unlearning?

Worst-case stream start: learn task 0, then unlearn it (simple fixes, optionally feature unlearning).
Probe task 0 (10-way, test images) at the output of every ResNet block, for:
    before U, after U, and a never-trained network (same init seed).
If after U is low at layer4 but stays high at layer2/layer3, the knowledge is only hidden,
and later learning can bring it back to the top (the jump seen after the next task).

Usage: python programs/phase_f_layer_probe.py [seed] [feature_steps]   -> results/F_layer_probe_s<seed>_f<steps>.json
"""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.clu import CLURun, make_config  # noqa: E402
from common.data import SplitCIFAR100, task_classes  # noqa: E402
from common.features import probe_accuracy  # noqa: E402
from common.models import resnet18_cifar  # noqa: E402
from common.utils import RESULTS_DIR, get_device  # noqa: E402

BLOCKS = ["layer1", "layer2", "layer3", "layer4"]


@torch.no_grad()
def block_features(model, X, chunk=250):
    model.eval()
    out = {b: [] for b in BLOCKS}
    for i in range(0, len(X), chunk):
        m, x = model, X[i:i + chunk]
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        for b in BLOCKS:
            x = getattr(m, b)(x)
            out[b].append(torch.flatten(torch.nn.functional.adaptive_avg_pool2d(x, 1), 1).float().cpu())
    return {b: torch.cat(v).numpy() for b, v in out.items()}


def probes(model, X, y):
    f = block_features(model, X)
    return {b: round(probe_accuracy(f[b], y, folds=5, seed=0)[0], 4) for b in BLOCKS}


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    fsteps = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    base = json.loads((RESULTS_DIR / "configs" / "relapse_guard.json").read_text())
    for k in ("name", "stream", "guard"):
        base.pop(k)
    cfg = make_config(**dict(base, name="diag", seed=seed, guard="none", head_reset=True,
                             mask_forgotten_logits=True, feature_steps=fsteps))
    device = get_device()
    data = SplitCIFAR100(seed=0, batch_size=cfg["batch_size"], num_workers=cfg["workers"])
    tasks = {t: task_classes(t) for t in range(10)}
    run = CLURun(cfg, data, device, [("L", 0), ("U", 0)], tasks, "task")
    rows = np.where(np.isin(run.Y, tasks[0]))[0]
    X, y = run.X[rows], run.Y[rows]

    torch.manual_seed(seed)
    never = resnet18_cifar(num_classes=100).to(device)
    res = {"never-trained": probes(never, X, y)}
    del never
    run.learn(0)
    res["before U"] = probes(run.model, X, y)
    run.unlearn(0, 1)
    res["after U"] = probes(run.model, X, y)
    print(f"{'':>14} " + " ".join(f"{b:>7}" for b in BLOCKS))
    for k, v in res.items():
        print(f"{k:>14} " + " ".join(f"{v[b]*100:6.1f}%" for b in BLOCKS))
    (RESULTS_DIR / f"F_layer_probe_s{seed}_f{fsteps}.json").write_text(json.dumps(res, indent=1))
