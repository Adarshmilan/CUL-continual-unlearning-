"""Program 01: train ResNet-18 on all 10 classes of CIFAR-10.

Run from the project root:
    python programs/01_train_basic.py              # full run (30 epochs)
    python programs/01_train_basic.py --epochs 1   # quick check that everything works

Done when: test accuracy is about 85% or higher, and reloading the saved model
gives the same number.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # lets us import common/

import matplotlib
matplotlib.use("Agg")  # save plots to files, no window needed
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from common.data import get_cifar10_loaders
from common.models import resnet18_cifar
from common.utils import RESULTS_DIR, get_device, log_run, set_seed


def train_one_epoch(model, loader, optimizer, scaler, device):
    """One pass over the training data. Returns (average loss, training accuracy)."""
    model.train()  # turns on training behavior of BatchNorm
    loss_fn = nn.CrossEntropyLoss()
    total_loss, correct, seen = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        # Mixed precision: run the forward pass in float16 where safe. About 2x faster on
        # this GPU and uses less memory. It does not change what the model learns.
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(images)
            loss = loss_fn(logits, labels)

        # The three steps of learning: clear old gradients, compute new ones, update weights.
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += labels.size(0)

    return total_loss / seen, correct / seen


@torch.no_grad()  # no gradients needed when we only measure
def evaluate(model, loader, device):
    """Accuracy of the model on a loader, as a fraction between 0 and 1."""
    model.eval()  # BatchNorm uses its stored statistics, no randomness
    correct, seen = 0, 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(images)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += labels.size(0)
    return correct / seen


def plot_curves(history, path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epochs, history["train_loss"])
    ax1.set(title="Training loss", xlabel="epoch", ylabel="cross-entropy")
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], label="train")
    ax2.plot(epochs, [a * 100 for a in history["test_acc"]], label="test")
    ax2.set(title="Accuracy", xlabel="epoch", ylabel="accuracy (%)")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    train_loader, test_loader = get_cifar10_loaders(args.batch_size, args.workers)
    model = resnet18_cifar(num_classes=10).to(device)

    # SGD with momentum. Weight decay gently pulls weights toward 0, which reduces overfitting.
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=args.weight_decay)
    # Cosine schedule: start at lr 0.1 and smoothly lower it to ~0 by the last epoch.
    # Big steps early to make progress, small steps late to settle into a good solution.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    history = {"train_loss": [], "train_acc": [], "test_acc": []}
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, device)
        test_acc = evaluate(model, test_loader, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        print(f"epoch {epoch:2d}/{args.epochs}  loss {train_loss:.3f}  "
              f"train {train_acc*100:5.2f}%  test {test_acc*100:5.2f}%  "
              f"({time.time() - start:.0f}s)")

    # Save only the weights (state_dict). This is the standard, portable way.
    RESULTS_DIR.mkdir(exist_ok=True)
    model_path = RESULTS_DIR / "01_resnet18_cifar10.pt"
    torch.save(model.state_dict(), model_path)

    # "Done when" check: build a fresh model, load the saved weights, test again.
    reloaded = resnet18_cifar(num_classes=10).to(device)
    reloaded.load_state_dict(torch.load(model_path, map_location=device))
    reloaded_acc = evaluate(reloaded, test_loader, device)

    plot_curves(history, RESULTS_DIR / "01_curves.png")

    print(f"\nFinal test accuracy:    {test_acc*100:.2f}%")
    print(f"Reloaded test accuracy: {reloaded_acc*100:.2f}%  "
          f"({'same' if abs(reloaded_acc - test_acc) < 1e-9 else 'DIFFERENT, check this'})")
    print(f"Saved model to {model_path}")
    print(f"Saved plot to  {RESULTS_DIR / '01_curves.png'}")

    log_run("01_train_basic", vars(args),
            {"test_acc": test_acc, "reloaded_acc": reloaded_acc,
             "minutes": round((time.time() - start) / 60, 1)})


if __name__ == "__main__":  # needed because DataLoader workers start new processes
    main()
