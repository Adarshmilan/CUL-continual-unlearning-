"""Training and testing loops shared by every program from 03 on.

Same loop as program 01. Later programs add extra loss terms (EWC, replay,
DER++, unlearning) through the `extra_loss` hook instead of copying the loop.
"""

import numpy as np
import torch
import torch.nn as nn

from common.data import N_TASKS, task_classes


def train_one_epoch(model, loader, optimizer, scaler, device, extra_loss=None):
    """One pass over a loader. Returns (average loss, training accuracy).

    extra_loss(model, images, labels, logits) -> tensor, optional. It is added to
    the cross-entropy loss. Program 03 does not use it; programs 04+ will.
    """
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, correct, seen = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(images)
            loss = loss_fn(logits, labels)
            if extra_loss is not None:
                loss = loss + extra_loss(model, images, labels, logits)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += labels.size(0)

    return total_loss / seen, correct / seen


@torch.no_grad()
def evaluate(model, loader, device, allowed_labels=None):
    """Accuracy on a loader, as a fraction between 0 and 1.

    allowed_labels=None   -> class-incremental: pick the best of all 100 outputs.
    allowed_labels=[...]  -> task-incremental: only these outputs may win
                             (the model is told which task the image is from).
    """
    model.eval()
    mask = None
    if allowed_labels is not None:
        mask = torch.full((model.fc.out_features,), float("-inf"), device=device)
        mask[list(allowed_labels)] = 0.0

    correct, seen = 0, 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(images)
        if mask is not None:
            logits = logits + mask
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += labels.size(0)
    return correct / seen


@torch.no_grad()
def per_class_accuracy(model, loader, device, n_classes=None):
    """Accuracy of each class separately, as a numpy array (NaN for classes not in the loader)."""
    model.eval()
    n_classes = n_classes or model.fc.out_features
    correct = torch.zeros(n_classes)
    total = torch.zeros(n_classes)
    for images, labels in loader:
        preds = model(images.to(device, non_blocking=True)).argmax(1).cpu()
        total += torch.bincount(labels, minlength=n_classes).float()
        correct += torch.bincount(labels[preds == labels], minlength=n_classes).float()
    acc = correct / total
    acc[total == 0] = float("nan")
    return acc.numpy()


@torch.no_grad()
def predict_all(model, loader, device):
    """Softmax probabilities and labels for every image in a loader: (probs [N, C], labels [N])."""
    model.eval()
    probs, labels = [], []
    for images, y in loader:
        probs.append(torch.softmax(model(images.to(device, non_blocking=True)).float(), dim=1).cpu())
        labels.append(y)
    return torch.cat(probs), torch.cat(labels)


def evaluate_tasks(model, data, up_to_task, device):
    """Test on every task 0..up_to_task.

    Returns two lists of accuracies (one per task):
      class_il: the model must pick among all 100 outputs (our setting)
      task_il:  the model only picks among that task's 10 outputs (easier)
    """
    class_il, task_il = [], []
    for t in range(up_to_task + 1):
        _, test_loader = data.get_task_loaders(t)
        class_il.append(evaluate(model, test_loader, device))
        task_il.append(evaluate(model, test_loader, device, allowed_labels=task_classes(t)))
    return class_il, task_il


def new_acc_matrix():
    """acc[after_task][on_task]. NaN means 'task not seen yet'."""
    return np.full((N_TASKS, N_TASKS), np.nan)


def average_accuracy(acc, after_task):
    """Mean accuracy over tasks 0..after_task, measured after training on after_task."""
    return float(np.mean(acc[after_task, : after_task + 1]))


def forgetting(acc):
    """Average forgetting at the end: for each old task, best accuracy it ever had
    minus its final accuracy. The last task is left out (it cannot be forgotten yet)."""
    last = acc.shape[0] - 1
    drops = [np.nanmax(acc[:last, t]) - acc[last, t] for t in range(last)]
    return float(np.mean(drops))
