"""UnCLe's four CLU metrics (program 13): RA, FA, Spill, Relapse.

They all read an accuracy table acc[request, item]: the accuracy of every item (a task, or a
class at class level) measured after every request, as saved by program 12.

    RA      average final accuracy on retained items (higher is better)
    FA      average final accuracy on forgotten items (close to chance is better)
    Spill   for each unlearn request: sum over all OTHER retained items of |change in accuracy|
            caused by that request (lower is better); reported as the mean over unlearn requests
    Relapse for each forgotten item: |accuracy right after its unlearn - accuracy at the end|
            (lower is better); reported as the mean over forgotten items

Accuracies are fractions (0-1). NaN in acc means "not learned yet".
"""

import numpy as np


def item_view(npz, kind="til", classes_per_task=10):
    """Turn program 12's saved file into per-item tables.

    kind: "til" (task-IL) or "cil" (class-IL).
    Returns acc [R, n_items], learned [R, n_items], forgotten [R, n_items] (booleans),
    and unlearn_rows: a list of (request row, item id).
    Task level: items are tasks (accuracy = mean over the task's classes). Class level: items are classes.
    """
    level = str(npz["level"])
    acc = np.asarray(npz[f"acc_{kind}"], dtype=float)
    learned, forgotten = np.asarray(npz["learned"]), np.asarray(npz["forgotten"])
    requests = [str(s) for s in npz["requests"]]
    R = len(requests)
    if level == "task":
        n_tasks = acc.shape[1] // classes_per_task
        acc = acc.reshape(R, n_tasks, classes_per_task).mean(2)
        learned = learned.reshape(R, n_tasks, classes_per_task).all(2)
        forgotten = forgotten.reshape(R, n_tasks, classes_per_task).all(2)
    acc = np.where(learned, acc, np.nan)
    unlearn_rows = [(r, int(s[1:])) for r, s in enumerate(requests) if s[0] == "U"]
    return acc, learned, forgotten, unlearn_rows


def retain_accuracy(acc, learned, forgotten):
    """RA: mean final accuracy over items learned and not forgotten at the end."""
    kept = learned[-1] & ~forgotten[-1]
    return float(np.mean(acc[-1, kept])) if kept.any() else float("nan")


def forget_accuracy(acc, forgotten):
    """FA: mean final accuracy over forgotten items."""
    gone = forgotten[-1]
    return float(np.mean(acc[-1, gone])) if gone.any() else float("nan")


def spill_per_unlearn(acc, learned, forgotten, unlearn_rows):
    """For each unlearn request at row r: sum of |acc[r] - acc[r-1]| over the items still retained after it."""
    out = []
    for r, _ in unlearn_rows:
        others = learned[r] & ~forgotten[r] & learned[r - 1]
        out.append(float(np.abs(acc[r, others] - acc[r - 1, others]).sum()))
    return out


def relapse_per_item(acc, unlearn_rows):
    """For each forgotten item: |final accuracy - accuracy right after its unlearn request|."""
    return [float(abs(acc[-1, item] - acc[r, item])) for r, item in unlearn_rows]


def all_metrics(acc, learned, forgotten, unlearn_rows):
    """The four numbers, plus the per-request lists behind Spill and Relapse."""
    spill = spill_per_unlearn(acc, learned, forgotten, unlearn_rows)
    relapse = relapse_per_item(acc, unlearn_rows)
    return {
        "RA": retain_accuracy(acc, learned, forgotten),
        "FA": forget_accuracy(acc, forgotten),
        "Spill": float(np.mean(spill)) if spill else float("nan"),
        "Relapse": float(np.mean(relapse)) if relapse else float("nan"),
        "spill_list": spill,
        "relapse_list": relapse,
    }
