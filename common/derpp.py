"""DER++ (Dark Experience Replay++, Buzzega et al., NeurIPS 2020): our base continual learner.

Program 06 explains and tests it. Every program from Phase D on imports it from here.

Each training step uses three loss terms:

    loss = CE(new batch)                                   learn the current task
         + alpha * MSE(outputs on buffer batch A, stored logits)   behave as before on old images
         + beta  * CE(outputs on buffer batch B, labels)           still get old labels right

We follow the paper's naming: alpha weights the logit (MSE) term, beta the label (CE) term.

Usage:
    buffer = DERBuffer(size=2000, dataset=data.train_set)
    for task in ...:
        loader = data.get_task_loader_with_index(task)   # batches of (images, labels, dataset indices)
        for epoch in ...:
            losses, train_acc = train_one_epoch_derpp(model, loader, optimizer, buffer, scaler, device)
"""

import numpy as np
import torch
import torch.nn.functional as F


class DERBuffer:
    """Reservoir-sampled memory of (dataset index, label, logits).

    Stores the dataset index instead of a copy of the image, so every replay loads the
    image fresh with a new random augmentation, and we can see exactly which images
    (and classes) are in the buffer. The logits are the model's outputs when the image
    was added; they are stored once and never updated.

    Reservoir sampling: every image ever offered has the same chance of being in the
    buffer, so after task t each task holds about size / (t + 1) images.
    """

    def __init__(self, size: int, dataset, n_classes: int = 100):
        self.size = size
        self.dataset = dataset        # where images are loaded from (augmented training set)
        self.indices = np.zeros(size, dtype=np.int64)
        self.labels = np.zeros(size, dtype=np.int64)
        self.logits = torch.zeros(size, n_classes)
        self.n_kept = 0               # slots filled
        self.n_seen = 0               # images offered so far

    def __len__(self):
        return self.n_kept

    def add(self, indices, labels, logits):
        """Offer a batch to the buffer. indices/labels: CPU tensors; logits: any device."""
        logits = logits.detach().float().cpu()
        for k, (idx, label) in enumerate(zip(indices.tolist(), labels.tolist())):
            if self.n_kept < self.size:                 # not full yet: keep it
                slot = self.n_kept
                self.n_kept += 1
            else:                                       # full: keep with chance size / (n_seen + 1)
                slot = np.random.randint(0, self.n_seen + 1)
                if slot >= self.size:
                    slot = None
            if slot is not None:
                self.indices[slot], self.labels[slot] = idx, label
                self.logits[slot] = logits[k]
            self.n_seen += 1

    def sample(self, n: int):
        """n random entries: (freshly augmented images, labels, stored logits), all on CPU."""
        pick = np.random.choice(self.n_kept, size=min(n, self.n_kept), replace=False)
        images = torch.stack([self.dataset[i][0] for i in self.indices[pick]])
        return images, torch.from_numpy(self.labels[pick]), self.logits[pick]

    def per_task(self, n_tasks: int = 10, classes_per_task: int = 10):
        """How many buffer entries belong to each task."""
        return np.bincount(self.labels[: self.n_kept] // classes_per_task, minlength=n_tasks)


def train_one_epoch_derpp(model, loader, optimizer, buffer: DERBuffer, scaler, device,
                          alpha: float = 0.5, beta: float = 0.5, replay_batch: int = 32):
    """One epoch of DER++ over a loader of (images, labels, dataset indices).

    New batch + buffer batch A + buffer batch B are joined into ONE forward pass, so
    BatchNorm statistics cover old and new tasks. Losses are computed in float32.

    Returns (average of each loss term, training accuracy on the new images).
    """
    model.train()
    sums = {"ce_new": 0.0, "mse_buf": 0.0, "ce_buf": 0.0}
    correct, seen, steps = 0, 0, 0
    for images, labels, idx in loader:
        n = labels.size(0)
        use_buffer = len(buffer) > 0
        x = images
        if use_buffer:
            xa, _, za = buffer.sample(replay_batch)     # A: images + stored logits
            xb, yb, _ = buffer.sample(replay_batch)     # B: images + labels
            x = torch.cat([images, xa, xb])
        x = x.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(x)
        out = out.float()
        out_new = out[:n]

        ce_new = F.cross_entropy(out_new, labels)
        loss = ce_new
        if use_buffer:
            na = xa.size(0)
            mse_buf = F.mse_loss(out[n:n + na], za.to(device))
            ce_buf = F.cross_entropy(out[n + na:], yb.to(device))
            loss = loss + alpha * mse_buf + beta * ce_buf
            sums["mse_buf"] += mse_buf.item()
            sums["ce_buf"] += ce_buf.item()

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        buffer.add(idx, labels.cpu(), out_new)        # store the new images with their current logits

        sums["ce_new"] += ce_new.item()
        correct += (out_new.argmax(1) == labels).sum().item()
        seen += n
        steps += 1
    return {k: v / steps for k, v in sums.items()}, correct / seen
