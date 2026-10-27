"""Program 02: split CIFAR-100 into 10 tasks of 10 classes and check the split.

The real code lives in common/data.py (every later program imports it).
This script uses it, checks the "Done when" conditions, and saves a picture
of each task so you can see what the model will learn, and when.

Run from the project root:
    python programs/02_split_dataset.py

Done when: each task has 5,000 train and 1,000 test images, and no class
appears in two tasks.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # lets us import common/

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common.data import (CIFAR100_MEAN, CIFAR100_STD, N_TASKS, SplitCIFAR100,
                         get_class_order, task_classes)
from common.utils import RESULTS_DIR, log_run

SEED = 0


def check_split(data: SplitCIFAR100):
    """Assert every 'Done when' condition. Stops with an error if anything is wrong."""
    seen_labels = set()
    rows = []
    for t in range(N_TASKS):
        train_sub, test_sub = data.get_task_subsets(t)

        # Read the labels straight from the stored targets (fast, no image loading).
        train_labels = {data.train_set.targets[i] for i in train_sub.indices}
        test_labels = {data.test_set.targets[i] for i in test_sub.indices}

        assert len(train_sub) == 5000, f"task {t}: {len(train_sub)} train images"
        assert len(test_sub) == 1000, f"task {t}: {len(test_sub)} test images"
        assert train_labels == test_labels == set(task_classes(t)), f"task {t}: wrong labels"
        assert not (train_labels & seen_labels), f"task {t}: a class repeats from an earlier task"
        seen_labels |= train_labels

        names = [data.class_names[c] for c in task_classes(t)]
        rows.append((t, len(train_sub), len(test_sub), names))

    assert seen_labels == set(range(100)), "some classes are missing"
    return rows


def check_class_loader(data: SplitCIFAR100, label: int):
    """One class should give 500 train and 100 test images, all with that label."""
    for train, expected in [(True, 500), (False, 100)]:
        loader = data.get_class_loader(label, train=train)
        labels = torch.cat([y for _, y in loader])
        assert len(labels) == expected and (labels == label).all()
    return True


def check_loader_batch(data: SplitCIFAR100, task_id: int):
    """Pull one real batch from a task loader to make sure images and labels look right."""
    train_loader, _ = data.get_task_loaders(task_id)
    images, labels = next(iter(train_loader))
    assert images.shape[1:] == (3, 32, 32)
    assert set(labels.tolist()) <= set(task_classes(task_id))
    return images.shape


def save_task_grid(data: SplitCIFAR100, path):
    """One example image per class: rows are tasks, columns are the 10 classes of that task."""
    mean, std = np.array(CIFAR100_MEAN), np.array(CIFAR100_STD)
    fig, axes = plt.subplots(N_TASKS, 10, figsize=(14, 16))
    for t in range(N_TASKS):
        for j, label in enumerate(task_classes(t)):
            idx = data._test_idx[label][0]  # first test image of this class
            img, _ = data.test_set[idx]
            img = (img.permute(1, 2, 0).numpy() * std + mean).clip(0, 1)  # undo normalization
            ax = axes[t, j]
            ax.imshow(img)
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(f"{label}: {data.class_names[label]}", fontsize=7)
            if j == 0:
                ax.set_ylabel(f"task {t}", fontsize=10)
    fig.suptitle(f"Split CIFAR-100 (seed {data.seed}): label = position in class order", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    data = SplitCIFAR100(seed=SEED)

    print(f"Class order (seed {SEED}), first 10 original ids: {data.class_order[:10].tolist()}")
    same = np.array_equal(get_class_order(SEED), get_class_order(SEED))
    assert same, "class order is not reproducible"
    print("Same seed gives the same order again: OK\n")

    rows = check_split(data)
    print(f"{'task':>4} {'train':>6} {'test':>5}  classes")
    for t, n_train, n_test, names in rows:
        print(f"{t:>4} {n_train:>6} {n_test:>5}  {', '.join(names)}")

    check_class_loader(data, label=0)
    shape = check_loader_batch(data, task_id=3)
    print(f"\nget_class_loader(0): 500 train / 100 test images, all label 0  OK")
    print(f"get_task_loaders(3): one batch has shape {tuple(shape)}, labels in 30..39  OK")
    print("All checks passed: 10 tasks x (5,000 train, 1,000 test), no class in two tasks.")

    RESULTS_DIR.mkdir(exist_ok=True)
    grid_path = RESULTS_DIR / "02_task_grid.png"
    save_task_grid(data, grid_path)
    print(f"Saved picture of all tasks to {grid_path}")

    log_run("02_split_dataset", {"seed": SEED},
            {"class_order": data.class_order.tolist(),
             "tasks": {t: names for t, _, _, names in rows}})


if __name__ == "__main__":
    main()
