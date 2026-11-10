"""Relapse-Guard: one configurable Continual Learning-Unlearning runner (programs 17 to 20).

A run goes through a list of requests ("L", task) / ("U", item) and after every request
measures every class (class-IL and task-IL accuracy) and the feature probe of every forgotten item.

On "L t": DER++ (program 06) on task t, plus, once something is forgotten, the guard.
On "U i": in this order
    1. record the forget signature U_i from the current features (program 16)
    2. BadTeacher unlearning (program 09)
    3. head reset: the forgotten classes' rows of the head get fresh random weights and are frozen
    4. purge: delete the forgotten classes from the DER++ buffer
    5. logit mask: the forgotten classes' columns are left out of DER++'s logit-replay loss

Every switch is in the config (see DEFAULTS). Program 12 = purge only, program 17 = the simple
fixes (3+4+5), program 18 = + guard "penalty", program 19 = + guard "projection".

The guard (program 18): during learning, for the batch's features h (centered on the batch mean),
    guard = sum over forgotten items c of ||h_c U_c||^2 / ||h_c||^2   (share of the batch's feature
    variance that lies in the forgotten directions), added to the loss as lambda * guard.
The projection guard (program 19): the gradient that reaches h has its component inside the
forgotten directions removed, so no loss can push the features along them directly.

Signatures (program 16): "plain" = PCA of the forgotten class's features around its own mean.
"specific" = contrastive PCA (Abid et al., 2018): the top eigenvectors of C_forget - C_retain,
both second moments around the retain mean, i.e. directions where the forgotten class has much
more energy than the kept classes. Program 16 found plain signatures overlap strongly with kept classes.

The layer guard (Phase F, "guard_layers"): a weight-level guard in the style of GPM (Saha et al., 2021),
used the other way round. After an item is unlearned, for each guarded conv layer we store the input
directions (unfolded patches) where the forgotten classes have much more energy than the kept
classes ("specific" directions: generalized eigenvectors of C_forget vs C_retain with ratio > layer_ratio).
During later learning every guarded weight gradient has those input directions removed, so the layer's
response to forgotten-class-specific inputs cannot change: later learning can't rebuild the class there.
It stores no images, only a few directions per layer.

Reference mode ("reference": True): the retrain target on the same stream. Forgotten items are never
learned (their L requests are skipped, class-level: their images are left out), U requests only mark
them, so every row lines up with the real run and the probe curves can be compared row by row.

One command:  python -m common.clu results/configs/relapse_guard.json
"""

import copy
import gc
import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common.badteacher import badteacher_unlearn, make_bad_teacher
from common.data import IndexedSubset, SplitCIFAR100, SplitTiny100, task_classes
from common.derpp import DERBuffer
from common.features import BLOCKS, block_features, features_of, penultimate, probe_accuracy
from common.unlearn_methods import unlearn_other
from common.metrics import all_metrics, item_view
from common.models import resnet18_cifar
from common.stream import load_stream
from common.utils import RESULTS_DIR, get_device, set_seed

DEFAULTS = {
    "name": "run",                 # results are saved as results/<name>.npz / _model.pt / _summary.json
    "stream": "task15_seed0",      # a file in results/streams/ (ignored if requests are passed directly)
    "seed": 0,
    "workers": 2,
    # DER++ (program 06)
    "epochs": 5, "lr": 0.03, "batch_size": 32,
    "buffer_size": 2000, "replay_batch": 32, "alpha": 0.5, "beta": 0.5,
    # BadTeacher (program 09)
    "unlearn_lr": 1e-4, "unlearn_steps": 50, "forget_batch": 64, "retain_batch": 192,
    "forget_data": "all",          # "all" = every training image of the item; "buffer" = its buffer images (program 12)
    "min_forget": 64, "bad_teacher_seed": 0,
    # simple fixes (program 17)
    "purge_buffer": True,
    "head_reset": False,
    "mask_forgotten_logits": False,
    # guard (programs 18, 19)
    "guard": "none",               # "none" | "penalty" | "projection"
    "guard_lambda": 1.0,
    "k": 5,
    "signature": "specific",       # "plain" | "specific"
    "signature_images": 1000,      # at most this many forget images to compute a signature
    # layer guard (Phase F): weight-level, GPM-style, see the module docstring
    "guard_layers": "",            # "" = off, or e.g. "layer4" / "layer3,layer4"
    "layer_ratio": 2.0,            # keep input directions where forget energy > ratio x retain energy
    "layer_max_frac": 0.1,         # at most this share of a layer's input dimensions per forgotten item
    "layer_total_frac": 0.3,       # at most this share of a layer's input dimensions in total (all items)
    "layer_images": 500,           # images (forget and retain each) used to measure a layer's inputs
    # feature unlearning after BadTeacher (Phase F): collapse the forgotten classes' features
    "feature_steps": 0,            # 0 = off
    "feature_lr": 1e-4,
    "feature_retain_weight": 1.0,  # weight of the retain feature-distillation term
    # paper options
    "dataset": "cifar100",         # "cifar100" | "tiny100" (data/tiny100_32.npz, see experiments/prepare_tiny_imagenet.py)
    "unlearn_method": "badteacher",  # "badteacher" | "neggrad" | "randlabel" | "scrub" (common/unlearn_methods.py)
    "unlearn_tau": 4.6,            # clamp of the forget term for neggrad / scrub (ln 100)
    "probe_blocks": False,         # also probe the forgotten items at every ResNet block after every request
    "requests": None,              # a request list [["L", 0], ["U", 3], ...] instead of a stream file
    "tasks_by_name": None,         # {"0": [class names], ...}: custom tasks (CIFAR-100 names), else the 10 standard tasks
    "level": None,                 # "task" | "class", needed with "requests"
    # retrain reference on the same stream (Phase F)
    "reference": False,
    # measuring
    "probe_folds": 5,
}


def make_config(**overrides):
    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise KeyError(f"unknown config keys: {sorted(unknown)}")
    return {**DEFAULTS, **overrides}


# ------------------------------------------------------------------ signatures and guard


def compute_signature(H_forget, H_retain, k, kind):
    """A 512 x k orthonormal matrix of forbidden feature directions (numpy)."""
    if kind == "plain" or H_retain is None or len(H_retain) < 2:
        Hc = H_forget - H_forget.mean(0)
        _, _, Vt = np.linalg.svd(Hc, full_matrices=False)
        return Vt[:k].T.copy()
    m = H_retain.mean(0)
    Cf = (H_forget - m).T @ (H_forget - m) / len(H_forget)
    Cr = (H_retain - m).T @ (H_retain - m) / len(H_retain)
    w, V = np.linalg.eigh(Cf - Cr)
    return V[:, np.argsort(w)[::-1][:k]].copy()


def signature_share(H, U, m):
    """Share of each row's (centered) feature energy inside the directions U."""
    D = H - m
    return (np.linalg.norm(D @ U, axis=1) ** 2) / (np.linalg.norm(D, axis=1) ** 2 + 1e-12)


def guard_penalty(h, signatures):
    """Sum over forgotten items of the batch's feature-variance share inside their signature."""
    hc = h - h.mean(0, keepdim=True)
    denom = hc.pow(2).sum().detach() + 1e-8
    return sum((hc @ U).pow(2).sum() for U in signatures) / denom


def feature_unlearn(model, good, sample_forget, sample_retain, steps, lr, retain_weight, device):
    """Pull forget features to the kept classes' mean feature (their class structure collapses),
    while keeping retain features where the model before unlearning had them. BatchNorm frozen."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.eval()
    last = (float("nan"), float("nan"))
    for _ in range(steps):
        xf = sample_forget().to(device, non_blocking=True)
        hf = penultimate(model, xf)
        if sample_retain is not None:
            xr = sample_retain().to(device, non_blocking=True)
            hr = penultimate(model, xr)
            with torch.no_grad():
                hr_good = penultimate(good, xr)
                target = hr_good.mean(0, keepdim=True)
                scale = (hr_good - target).pow(2).sum(1).mean() + 1e-8
            lf = (hf - target).pow(2).sum(1).mean() / scale
            lr_ = (hr - hr_good).pow(2).sum(1).mean() / scale
        else:                                   # nothing kept yet: collapse the forget features to their mean
            with torch.no_grad():
                target = hf.mean(0, keepdim=True)
                scale = (hf - target).pow(2).sum(1).mean() + 1e-8
            lf = (hf - target.detach()).pow(2).sum(1).mean() / scale
            lr_ = torch.zeros((), device=device)
        loss = lf + retain_weight * lr_
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last = (lf.item(), lr_.item())
    return last


def guarded_convs(model, spec):
    """name -> conv module for every conv inside the listed blocks (e.g. "layer3,layer4")."""
    out = {}
    for block in [b.strip() for b in spec.split(",") if b.strip()]:
        for name, m in getattr(model, block).named_modules():
            if isinstance(m, torch.nn.Conv2d):
                out[f"{block}.{name}"] = m
    return out


@torch.no_grad()
def conv_input_moments(model, convs, X, chunk=100, patches_per_image=16, seed=0):
    """Second moment of the unfolded input patches of each conv: name -> [d, d] (GPU, float32)."""
    model.eval()
    C = {n: None for n in convs}
    count = {n: 0 for n in convs}
    g = torch.Generator(device="cpu").manual_seed(seed)

    def hook(name, conv):
        def f(_, inputs, __):
            x = inputs[0].float()
            P = F.unfold(x, conv.kernel_size, dilation=conv.dilation, padding=conv.padding, stride=conv.stride)
            P = P.transpose(1, 2)                                            # [B, L, d]
            pick = torch.randint(0, P.shape[1], (patches_per_image,), generator=g).to(P.device)
            P = P[:, pick].reshape(-1, P.shape[2])
            C[name] = P.T @ P if C[name] is None else C[name] + P.T @ P
            count[name] += P.shape[0]
        return f

    handles = [m.register_forward_hook(hook(n, m)) for n, m in convs.items()]
    for i in range(0, len(X), chunk):
        model(X[i:i + chunk])
    for h in handles:
        h.remove()
    return {n: C[n] / count[n] for n in convs}


def specific_input_basis(Cf, Cr, ratio, max_frac, eps=1e-4):
    """Orthonormal [d, m] basis of input directions where forget energy > ratio x retain energy.
    Computed on the GPU in float64; falls back to the CPU if the GPU is full (same result)."""
    try:
        return _specific_input_basis(Cf, Cr, ratio, max_frac, eps)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return _specific_input_basis(Cf.cpu(), Cr.cpu(), ratio, max_frac, eps).to(Cf.device)


def _specific_input_basis(Cf, Cr, ratio, max_frac, eps):
    d = Cf.shape[0]
    Cr = Cr.double() + eps * torch.trace(Cr.double()) / d * torch.eye(d, device=Cr.device, dtype=torch.float64)
    w, V = torch.linalg.eigh(Cr)
    Wh = V @ torch.diag(w.clamp_min(1e-12).rsqrt()) @ V.T                  # Cr^(-1/2)
    mu, E = torch.linalg.eigh(Wh @ Cf.double() @ Wh)
    order = torch.argsort(mu, descending=True)
    keep = order[mu[order] > ratio][: max(1, int(max_frac * d))]
    if len(keep) == 0:
        return torch.zeros(d, 0, device=Cf.device)
    B = Wh @ E[:, keep]                                                    # back to input space
    Q, _ = torch.linalg.qr(B)
    return Q.float()


# ------------------------------------------------------------------ the runner


class CLURun:
    """Holds the model, buffer and measurements of one run. Use run_clu() unless you need the parts."""

    def __init__(self, cfg, data, device, requests, tasks, level, log=print):
        self.cfg, self.data, self.device, self.log = cfg, data, device, log
        self.requests, self.level = requests, level
        self.tasks = {int(t): list(map(int, labels)) for t, labels in tasks.items()}
        self.task_ids = sorted(self.tasks)
        self.task_of_class = np.full(100, -1)
        for t, labels in self.tasks.items():
            self.task_of_class[labels] = t
        self.items = self.task_ids if level == "task" else sorted({i for k, i in requests if k == "U"})

        set_seed(cfg["seed"])
        self.rng = np.random.RandomState(cfg["seed"])
        self.model = resnet18_cifar(num_classes=100).to(device)
        self.scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
        self.buffer = DERBuffer(cfg["buffer_size"], dataset=data.train_set)

        # head freezing: gradients of reset rows are multiplied by 0
        self.row_keep = torch.ones(100, device=device)
        self.model.fc.weight.register_hook(lambda g: g * self.row_keep[:, None])
        self.model.fc.bias.register_hook(lambda g: g * self.row_keep)
        self.keep_cols = torch.ones(100, dtype=torch.bool, device=device)   # DER++ logit-replay columns
        self.signatures = {}                                                   # item -> torch [512, k]
        self.guard_basis = None                                                # orthonormal union (projection)
        self.forgotten = np.zeros(100, dtype=bool)
        self.learned = np.zeros(100, dtype=bool)
        # reference mode: every class that is ever unlearned is never learned
        self.never = set()
        if cfg["reference"]:
            for k, i in requests:
                if k == "U":
                    self.never |= set(self.classes_of(int(i)))
        # layer guard: conv name -> orthonormal basis of protected input directions [d, m]
        self.convs = guarded_convs(self.model, cfg["guard_layers"]) if cfg["guard_layers"] else {}
        self.layer_basis = {}

        self.train_targets = np.asarray(data.train_set.targets)
        # test images on the GPU, task columns for task-IL
        ts = data.test_set
        self.X = torch.stack([ts[i][0] for i in range(len(ts))]).to(device)
        self.Y = np.array(ts.targets)
        self.Yt = torch.from_numpy(self.Y).to(device)
        self.cols = torch.tensor([self.tasks[t] for t in self.task_ids], device=device)   # [n_tasks, 10]
        pos = {t: i for i, t in enumerate(self.task_ids)}
        tpos = np.array([pos.get(int(t), -1) for t in self.task_of_class[self.Y]])
        self.test_task_pos = torch.from_numpy(tpos).to(device)
        # fixed balanced probe sets for class items: the class vs as many images of its task's other classes
        r = np.random.RandomState(0)
        self.class_probe_rows = {}
        for c in (self.items if level == "class" else []):
            own = np.where(self.Y == c)[0]
            others = np.where(np.isin(self.Y, self.tasks[self.task_of_class[c]]) & (self.Y != c))[0]
            self.class_probe_rows[c] = (own, r.choice(others, len(own), replace=False))

    # ---- helpers
    def classes_of(self, item):
        return self.tasks[item] if self.level == "task" else [item]

    def load_images(self, idx, plain=False):
        ds = self.data.train_set_plain if plain else self.data.train_set
        return torch.stack([ds[j][0] for j in idx])

    @torch.no_grad()
    def per_class_acc(self):
        self.model.eval()
        logits = torch.cat([self.model(self.X[i:i + 1000]) for i in range(0, len(self.X), 1000)]).float()
        n = torch.bincount(self.Yt, minlength=100).float()
        cil = torch.bincount(self.Yt[logits.argmax(1) == self.Yt], minlength=100).float() / n
        has = self.test_task_pos >= 0
        cols = self.cols[self.test_task_pos[has]]                                 # [N', 10]
        pred = cols.gather(1, logits[has].gather(1, cols).argmax(1, keepdim=True)).squeeze(1)
        y = self.Yt[has]
        til = torch.bincount(y[pred == y], minlength=100).float() / n
        til[torch.bincount(y, minlength=100) == 0] = float("nan")
        return cil.cpu().numpy(), til.cpu().numpy()

    def probe(self, item, feats):
        if self.level == "task":
            rows = np.where(np.isin(self.Y, self.tasks[item]))[0]
            return probe_accuracy(feats[rows], self.Y[rows], folds=self.cfg["probe_folds"], seed=0)[0]
        own, others = self.class_probe_rows[item]
        rows = np.concatenate([own, others])
        y = np.r_[np.ones(len(own)), np.zeros(len(others))]
        return probe_accuracy(feats[rows], y, folds=self.cfg["probe_folds"], seed=0)[0]

    def test_features(self):
        return features_of(self.model, self.X).numpy()

    # ---- learning
    def learn(self, task):
        cfg = self.cfg
        classes = [c for c in self.tasks[task] if c not in self.never]
        if not classes:
            return "skipped (reference: never learned)"
        subset = IndexedSubset(self.data.train_set, self.data.train_indices(classes))
        loader = DataLoader(subset, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["workers"],
                            pin_memory=torch.cuda.is_available())
        optimizer = torch.optim.SGD(self.model.parameters(), lr=cfg["lr"], momentum=0.9)
        sig_list = list(self.signatures.values())
        G = self.guard_basis
        use_penalty = cfg["guard"] == "penalty" and sig_list
        use_proj = cfg["guard"] == "projection" and G is not None
        model, buffer, device = self.model, self.buffer, self.device
        guard_sum, steps = 0.0, 0
        for epoch in range(cfg["epochs"]):
            model.train()
            correct = seen = 0
            for images, labels, idx in loader:
                n = labels.size(0)
                use_buffer = len(buffer) > 0
                x = images
                if use_buffer:
                    xa, _, za = buffer.sample(cfg["replay_batch"])
                    xb, yb, _ = buffer.sample(cfg["replay_batch"])
                    x = torch.cat([images, xa, xb])
                x = x.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    h = penultimate(model, x)
                    if use_proj:
                        h.register_hook(lambda g: g - ((g.float() @ G) @ G.T).to(g.dtype))
                    out = model.fc(h)
                out = out.float()
                loss = F.cross_entropy(out[:n], labels)
                if use_buffer:
                    na = xa.size(0)
                    diff = (out[n:n + na] - za.to(device))[:, self.keep_cols]
                    loss = loss + cfg["alpha"] * diff.pow(2).mean() + cfg["beta"] * F.cross_entropy(out[n + na:], yb.to(device))
                if use_penalty:
                    g = guard_penalty(h.float(), sig_list)
                    loss = loss + cfg["guard_lambda"] * g
                    guard_sum += g.item()
                optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                for name, M in self.layer_basis.items():
                    wg = self.convs[name].weight.grad
                    g2 = wg.view(wg.shape[0], -1)
                    g2 -= (g2.float() @ M @ M.T).to(g2.dtype)
                self.scaler.step(optimizer)
                self.scaler.update()
                buffer.add(idx, labels.cpu(), out[:n])
                correct += (out[:n].argmax(1) == labels).sum().item()
                seen += n
                steps += 1
            if not torch.isfinite(loss):
                raise RuntimeError(f"task {task}, epoch {epoch}: loss is not finite")
        del loader, optimizer
        gc.collect()
        self.learned[classes] = True
        note = f"train acc {correct / seen * 100:.1f}%"
        if use_penalty:
            note += f", guard share {guard_sum / steps:.3f}"
        if len(classes) < len(self.tasks[task]):
            note += f" ({len(self.tasks[task]) - len(classes)} classes left out: reference)"
        return note

    # ---- unlearning
    def unlearn(self, item, r):
        cfg, buffer, model = self.cfg, self.buffer, self.model
        cls = self.classes_of(item)
        if cfg["reference"]:
            self.forgotten[cls] = True
            return {"request": r, "item": int(item), "forget_images": 0, "in_buffer": 0, "retain_images": 0,
                    "bad_teacher_seed": -1, "removed": int(buffer.remove(cls))}
        all_forgotten = set(np.where(self.forgotten)[0].tolist()) | set(cls)
        n = len(buffer)

        # forget data
        if cfg["forget_data"] == "all":
            forget_idx = np.array(self.data.train_indices(cls))
            from_buffer = len(buffer.positions_of(cls))
        else:
            fl = buffer.indices[buffer.positions_of(cls)].tolist()
            from_buffer = len(fl)
            if from_buffer < cfg["min_forget"]:
                pool = sorted(set(self.data.train_indices(cls)) - set(fl))
                fl += self.rng.choice(pool, cfg["min_forget"] - from_buffer, replace=False).tolist()
            forget_idx = np.array(fl)
        retain_idx = buffer.indices[:n][~np.isin(buffer.labels[:n], list(all_forgotten))]

        # 1. signature, from the features BEFORE unlearning
        if cfg["guard"] != "none":
            # own random generator, so runs with and without a guard unlearn with identical batches
            pick = np.random.RandomState(1000 + r).choice(forget_idx, min(cfg["signature_images"], len(forget_idx)), replace=False)
            Hf = features_of(model, self.load_images(pick, plain=True).to(self.device)).numpy()
            Hr = features_of(model, self.load_images(retain_idx, plain=True).to(self.device)).numpy() if len(retain_idx) else None
            U = compute_signature(Hf, Hr, cfg["k"], cfg["signature"])
            self.signatures[item] = torch.from_numpy(U.astype(np.float32)).to(self.device)
            Q, _ = torch.linalg.qr(torch.cat(list(self.signatures.values()), dim=1))
            self.guard_basis = Q

        # 2. BadTeacher
        fb, rb = cfg["forget_batch"], cfg["retain_batch"]
        sample_forget = lambda: self.load_images(self.rng.choice(forget_idx, fb, replace=len(forget_idx) < fb))
        sample_retain = None
        if len(retain_idx):
            sample_retain = lambda: self.load_images(self.rng.choice(retain_idx, min(rb, len(retain_idx)), replace=False))
        good = copy.deepcopy(model).eval()
        for p in good.parameters():
            p.requires_grad_(False)
        bad, bad_seed = None, -1
        if cfg["unlearn_method"] == "badteacher":
            among = self.tasks[self.task_of_class[item]] if self.level == "class" else None
            if among is not None and set(among) <= all_forgotten:
                among = None          # the whole task is forgotten now: same as task-level forgetting
            bad, bad_seed, _ = make_bad_teacher(100, self.load_images(forget_idx[:256]), all_forgotten,
                                                cfg["bad_teacher_seed"] + r, self.device, among=among)
            badteacher_unlearn(model, good, bad, sample_forget, sample_retain, cfg["unlearn_steps"], cfg["unlearn_lr"], self.device)
        else:
            T = self.train_targets
            def with_labels(pool, n):
                idx = self.rng.choice(pool, n, replace=len(pool) < n)
                return self.load_images(idx), torch.as_tensor(T[idx], dtype=torch.long)
            lab_forget = lambda: with_labels(forget_idx, fb)
            lab_retain = (lambda: with_labels(retain_idx, min(rb, len(retain_idx)))) if len(retain_idx) else None
            kept = np.where(self.learned & ~np.isin(np.arange(100), list(all_forgotten)))[0]
            pool = kept if len(kept) else np.setdiff1d(np.arange(100), list(all_forgotten))
            unlearn_other(cfg["unlearn_method"], model, good, lab_forget, lab_retain, cfg["unlearn_steps"],
                          cfg["unlearn_lr"], self.device, random_label_pool=pool, tau=cfg["unlearn_tau"])
        feat_loss = None
        if cfg["feature_steps"]:
            feat_loss = feature_unlearn(model, good, sample_forget, sample_retain, cfg["feature_steps"],
                                        cfg["feature_lr"], cfg["feature_retain_weight"], self.device)
        del good, bad
        torch.cuda.empty_cache()
        self.forgotten[cls] = True

        # 3. head reset
        if cfg["head_reset"]:
            with torch.no_grad():
                bound = 1 / np.sqrt(model.fc.in_features)
                model.fc.weight[cls] = torch.empty_like(model.fc.weight[cls]).uniform_(-bound, bound)
                model.fc.bias[cls] = torch.empty_like(model.fc.bias[cls]).uniform_(-bound, bound)
            self.row_keep[cls] = 0.0
        # 4. purge
        removed = buffer.remove(cls) if cfg["purge_buffer"] else 0
        # 5. logit mask
        if cfg["mask_forgotten_logits"]:
            self.keep_cols[cls] = False
        # 6. layer guard: protect the forgotten classes' specific input directions (measured AFTER unlearning)
        layer_dims = {}
        if self.convs:
            ri = buffer.indices[:len(buffer)]
            ri = ri[~np.isin(buffer.labels[:len(buffer)], np.where(self.forgotten)[0])]
            r2 = np.random.RandomState(2000 + r)
            Xf = self.load_images(r2.choice(forget_idx, min(cfg["layer_images"], len(forget_idx)), replace=False), plain=True).to(self.device)
            Cf = conv_input_moments(model, self.convs, Xf, seed=r)
            if len(ri):
                Xr = self.load_images(r2.choice(ri, min(cfg["layer_images"], len(ri)), replace=False), plain=True).to(self.device)
                Cr = conv_input_moments(model, self.convs, Xr, seed=r + 1)
            else:   # nothing kept yet: compare with an even spread of the forget energy
                Xr = None
                Cr = {n: torch.trace(C) / C.shape[0] * torch.eye(C.shape[0], device=C.device) for n, C in Cf.items()}
            for name in self.convs:
                B = specific_input_basis(Cf[name], Cr[name], cfg["layer_ratio"], cfg["layer_max_frac"])
                if name in self.layer_basis:
                    old = self.layer_basis[name]
                    B = B - old @ (old.T @ B)                                  # only the new part
                    B = B[:, torch.linalg.norm(B, dim=0) > 0.5]                 # drop near-duplicates
                    B = torch.linalg.qr(B)[0] if B.shape[1] else B
                    room = max(0, int(cfg["layer_total_frac"] * old.shape[0]) - old.shape[1])
                    B = torch.cat([old, B[:, :room]], dim=1)          # total cap: the guard must leave room to learn
                self.layer_basis[name] = B
                layer_dims[name] = int(B.shape[1])
            del Xf, Xr, Cf, Cr
            torch.cuda.empty_cache()
        return {"layer_dims": layer_dims, "feature_loss": feat_loss, "request": r, "item": int(item), "forget_images": int(len(forget_idx)), "in_buffer": int(from_buffer),
                "retain_images": int(len(retain_idx)), "bad_teacher_seed": int(bad_seed), "removed": int(removed)}

    # ---- the whole stream
    def run(self):
        R = len(self.requests)
        n_items = len(self.items)
        item_pos = {it: j for j, it in enumerate(self.items)}
        out = {"acc_cil": np.full((R, 100), np.nan), "acc_til": np.full((R, 100), np.nan),
               "learned": np.zeros((R, 100), bool), "forgotten": np.zeros((R, 100), bool),
               "probe": np.full((R, n_items), np.nan), "probe_before": np.full(n_items, np.nan),
               "probe_blocks": np.full((R, n_items, len(BLOCKS)), np.nan),
               "probe_before_blocks": np.full((n_items, len(BLOCKS)), np.nan)}
        cfg = self.cfg
        unlearn_log, start = [], time.time()
        for r, (kind, item) in enumerate(self.requests):
            if kind == "L":
                note = self.learn(item)
            else:
                out["probe_before"][item_pos[item]] = self.probe(item, self.test_features())
                if cfg["probe_blocks"]:
                    bf = block_features(self.model, self.X)
                    for b, B in enumerate(BLOCKS):
                        out["probe_before_blocks"][item_pos[item], b] = self.probe(item, bf[B])
                u = self.unlearn(item, r)
                unlearn_log.append(u)
                note = f"forget imgs {u['forget_images']} ({u['in_buffer']} in buffer), deleted {u['removed']}"
            cil, til = self.per_class_acc()
            out["acc_cil"][r], out["acc_til"][r] = cil, til
            out["learned"][r], out["forgotten"][r] = self.learned, self.forgotten
            gone = [it for it in self.items if self.forgotten[self.classes_of(it)].all()]
            if gone:
                feats = self.test_features()
                for it in gone:
                    out["probe"][r, item_pos[it]] = self.probe(it, feats)
                if cfg["probe_blocks"]:
                    bf = block_features(self.model, self.X)
                    for it in gone:
                        for b, B in enumerate(BLOCKS):
                            out["probe_blocks"][r, item_pos[it], b] = self.probe(it, bf[B])
                    del bf
            kept = self.learned & ~self.forgotten
            txt = f"kept {np.nanmean(til[kept])*100:5.1f}%" if kept.any() else "kept   -  "
            if self.forgotten.any():
                txt += (f"  forgot head {np.nanmean(til[self.forgotten])*100:5.1f}%"
                        f"  probe {np.nanmean(out['probe'][r, [item_pos[g] for g in gone]])*100:5.1f}%")
            self.log(f"[{r:3d}] {kind}{item:<3} {txt}  {note}  ({time.time() - start:.0f}s)")
        feats = self.test_features()
        out["probe_end_all"] = np.array([self.probe(it, feats) for it in self.items])
        out["items"] = np.array(self.items)
        out["requests"] = np.array([f"{k}{i}" for k, i in self.requests])
        out["level"] = self.level
        out["minutes"] = (time.time() - start) / 60
        out["unlearn_log"] = unlearn_log
        return out


def run_clu(cfg, data=None, requests=None, tasks=None, level=None, log=print, save=True):
    """Run one configuration. Returns the results dict (and saves results/<name>.npz, _model.pt, _summary.json)."""
    device = get_device()
    own_data = data is None
    if own_data:
        Data = {"cifar100": SplitCIFAR100, "tiny100": SplitTiny100}[cfg["dataset"]]
        data = Data(seed=0, batch_size=cfg["batch_size"], num_workers=cfg["workers"])
    if requests is None and cfg["requests"] is not None:          # items may be class names: ["U", "wolf"]
        requests = [(k, data.class_names.index(i) if isinstance(i, str) else int(i)) for k, i in cfg["requests"]]
        level = cfg["level"]
    if requests is None:
        info = load_stream(RESULTS_DIR / "streams" / f"{cfg['stream']}.json")
        requests, level = info["requests"], info["level"]
    if tasks is None and cfg["tasks_by_name"] is not None:
        tasks = {int(t): [data.class_names.index(n) for n in names] for t, names in cfg["tasks_by_name"].items()}
    if tasks is None:
        tasks = {t: task_classes(t) for t in range(10)}
    run = CLURun(cfg, data, device, requests, tasks, level, log=log)
    res = run.run()
    res["summary"] = summarize(res)
    if save:
        name = cfg["name"]
        (RESULTS_DIR / name).parent.mkdir(parents=True, exist_ok=True)       # names like "paper/xyz"
        np.savez(RESULTS_DIR / f"{name}.npz", **{k: v for k, v in res.items() if k not in ("unlearn_log", "summary")})
        torch.save(run.model.state_dict(), RESULTS_DIR / f"{name}_model.pt")
        (RESULTS_DIR / f"{name}_summary.json").write_text(json.dumps(
            {"config": cfg, "summary": res["summary"], "unlearn": res["unlearn_log"]}, indent=1))
    del run
    gc.collect()
    torch.cuda.empty_cache()
    if own_data:
        del data
    return res


def load_result(name):
    """Reload a saved run: the arrays of results/<name>.npz plus its summary and config, or None if missing."""
    npz, js = RESULTS_DIR / f"{name}.npz", RESULTS_DIR / f"{name}_summary.json"
    if not (npz.exists() and js.exists()):
        return None
    d = np.load(npz, allow_pickle=True)
    res = {k: d[k] for k in d.files}
    res["level"] = str(res["level"])
    info = json.loads(js.read_text())
    res["summary"], res["config"] = info["summary"], info["config"]
    return res


def run_or_load(cfg, data=None, rerun=False, **kw):
    """Load results/<name> if it was already run with the same config, otherwise run it."""
    old = None if rerun else load_result(cfg["name"])
    if old is not None and old["config"] == cfg:
        print(f"{cfg['name']}: loaded saved results (pass rerun=True to run again)")
        return old
    return run_clu(cfg, data, **kw)


def summarize(res):
    """The four metrics (task-IL and class-IL) plus probe numbers for the forgotten items."""
    out = {}
    for kind in ("til", "cil"):
        acc, learned, forgotten, unlearn_rows = item_view(res, kind)
        m = all_metrics(acc, learned, forgotten, unlearn_rows)
        out[kind] = {k: (round(v, 4) if isinstance(v, float) else [round(x, 4) for x in v]) for k, v in m.items()}
    items = list(np.asarray(res["items"]))
    reqs = [str(s) for s in res["requests"]]
    rows = [(r, int(s[1:])) for r, s in enumerate(reqs) if s[0] == "U"]
    p = res["probe"]
    after = [p[r, items.index(i)] for r, i in rows]
    peak = [np.nanmax(p[r:, items.index(i)]) for r, i in rows]
    end = [p[-1, items.index(i)] for r, i in rows]
    before = [res["probe_before"][items.index(i)] for r, i in rows]
    out["probe"] = {"before": float(np.mean(before)), "after_U": float(np.mean(after)),
                    "end": float(np.mean(end)), "peak_rise": float(np.mean(np.array(peak) - np.array(after))),
                    "end_rise": float(np.mean(np.array(end) - np.array(after))),
                    "per_item": {int(i): [round(float(b), 4), round(float(a), 4), round(float(e), 4)]
                                 for (r, i), b, a, e in zip(rows, before, after, end)}}
    return out


if __name__ == "__main__":
    cfg = make_config(**json.loads(open(sys.argv[1]).read()))
    res = run_clu(cfg)
    print(json.dumps(res["summary"], indent=1))
