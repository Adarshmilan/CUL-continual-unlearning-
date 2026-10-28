"""Program 03: catastrophic forgetting.

Train one ResNet-18 on task 0, then task 1, ... up to task 9, with no protection
at all. After each task, test on every task seen so far and fill the accuracy
matrix acc[after_task][on_task].

The model has a single 100-way output (class-incremental): at test time it is
never told which task an image came from. We also record task-incremental
accuracy (only that task's 10 outputs may win) to show how much easier that is.

Run from the project root:
    python programs/03_naive_sequential.py               # full run (5 epochs per task)
    python programs/03_naive_sequential.py --epochs 1    # quick check

Done when: task 0 accuracy drops close to 0 after a few tasks, and you can read
the accuracy matrix.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # lets us import common/

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common.data import N_TASKS, SplitCIFAR100
from common.models import resnet18_cifar
from common.train import (average_accuracy, evaluate_tasks, forgetting,
                          new_acc_matrix, train_one_epoch)
from common.utils import RESULTS_DIR, get_device, log_run, set_seed


def to_json(acc):
    """Matrix as nested lists, with null for 'not seen yet' (NaN is not valid JSON)."""
    return [[None if np.isnan(v) else round(float(v), 4) for v in row] for row in acc]


def print_matrix(acc, title):
    print(f"\n{title}  (row = after training task, column = tested on task, %)")
    print("      " + "".join(f"{t:>6}" for t in range(N_TASKS)) + "   avg")
    for i in range(N_TASKS):
        cells = "".join("     ." if np.isnan(v) else f"{v * 100:6.1f}" for v in acc[i])
        print(f"after {i}{cells}  {average_accuracy(acc, i) * 100:5.1f}")


def plot_results(acc_cil, acc_til, path):
    """Left: one line per task (class-IL). Middle: the matrix as a heatmap.
    Right: average accuracy after each task, class-IL vs task-IL."""
    x = np.arange(N_TASKS)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 0.9, N_TASKS))
    for t in range(N_TASKS):
        ax.plot(x[t:], acc_cil[t:, t] * 100, marker="o", ms=3, color=colors[t], label=f"task {t}")
    ax.set(title="Accuracy on each task (class-incremental)",
           xlabel="after training task", ylabel="test accuracy (%)", ylim=(-2, 100), xticks=x)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    im = ax.imshow(acc_cil * 100, vmin=0, vmax=100, cmap="magma")
    for i in range(N_TASKS):
        for j in range(i + 1):
            v = acc_cil[i, j] * 100
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                    color="black" if v > 50 else "white")
    ax.set(title="acc[after_task][on_task] (%)", xlabel="tested on task",
           ylabel="after training task", xticks=x, yticks=x)
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[2]
    ax.plot(x, [average_accuracy(acc_til, i) * 100 for i in x], marker="o", label="task-IL (task id given)")
    ax.plot(x, [average_accuracy(acc_cil, i) * 100 for i in x], marker="o", label="class-IL (our setting)")
    ax.set(title="Average accuracy over seen tasks", xlabel="after training task",
           ylabel="accuracy (%)", ylim=(0, 100), xticks=x)
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5, help="epochs per task")
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    data = SplitCIFAR100(seed=args.seed, batch_size=args.batch_size, num_workers=args.workers)
    model = resnet18_cifar(num_classes=100).to(device)  # one head for all 100 classes
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    acc_cil, acc_til = new_acc_matrix(), new_acc_matrix()
    start = time.time()
    for task in range(N_TASKS):
        train_loader, _ = data.get_task_loaders(task)

        # Plain SGD, a fresh optimizer per task, nothing that remembers old tasks.
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        for epoch in range(1, args.epochs + 1):
            loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, device)
        print(f"task {task}: last epoch loss {loss:.3f}, train acc {train_acc * 100:.1f}%  "
              f"({time.time() - start:.0f}s)")

        cil, til = evaluate_tasks(model, data, task, device)
        acc_cil[task, : task + 1] = cil
        acc_til[task, : task + 1] = til
        print("   class-IL on seen tasks: " + " ".join(f"{a * 100:5.1f}" for a in cil))

    print_matrix(acc_cil, "Class-incremental accuracy")
    print_matrix(acc_til, "Task-incremental accuracy")

    f_cil, f_til = forgetting(acc_cil), forgetting(acc_til)
    print(f"\nTask 0 accuracy: {acc_cil[0, 0] * 100:.1f}% right after learning it, "
          f"{acc_cil[2, 0] * 100:.1f}% two tasks later, {acc_cil[-1, 0] * 100:.1f}% at the end")
    print(f"Final average accuracy: class-IL {average_accuracy(acc_cil, N_TASKS - 1) * 100:.1f}%, "
          f"task-IL {average_accuracy(acc_til, N_TASKS - 1) * 100:.1f}%")
    print(f"Average forgetting:     class-IL {f_cil * 100:.1f} points, task-IL {f_til * 100:.1f} points")

    RESULTS_DIR.mkdir(exist_ok=True)
    plot_path = RESULTS_DIR / "03_naive_sequential.png"
    plot_results(acc_cil, acc_til, plot_path)
    np.savez(RESULTS_DIR / "03_acc_matrix.npz", class_il=acc_cil, task_il=acc_til)
    print(f"Saved plot to {plot_path}")
    print(f"Saved matrices to {RESULTS_DIR / '03_acc_matrix.npz'} (program 06 compares against them)")

    log_run("03_naive_sequential", vars(args),
            {"acc_class_il": to_json(acc_cil),
             "acc_task_il": to_json(acc_til),
             "final_avg_class_il": average_accuracy(acc_cil, N_TASKS - 1),
             "final_avg_task_il": average_accuracy(acc_til, N_TASKS - 1),
             "forgetting_class_il": f_cil, "forgetting_task_il": f_til,
             "minutes": round((time.time() - start) / 60, 1)})


if __name__ == "__main__":
    main()
