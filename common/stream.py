"""Request streams for Continual Learning-Unlearning (CLU), Phase D on.

A stream is a list of requests, each a tuple:

    ("L", t)   learn task t (all its classes, trained with DER++)
    ("U", i)   unlearn item i: a task id (level="task") or a class label (level="class")

Rules every stream follows (check_stream enforces them):
  1. every task is learned exactly once;
  2. you can only unlearn something that has been learned and not yet forgotten;
  3. a forgotten item is never learned again (follows from rule 1, since tasks are learned once).

Program 11 explains and tests this. Programs 12+ load the saved JSON files, so every
method runs on exactly the same sequences.

Usage:
    stream = make_stream(n_tasks=10, n_requests=15, seed=0, level="task")
    save_stream(stream, path, n_tasks=10, level="task", seed=0)
    info = load_stream(path)            # dict with "requests" and the settings
"""

import json
from pathlib import Path

import numpy as np

CLASSES_PER_TASK = 10


def task_of(label: int, classes_per_task: int = CLASSES_PER_TASK) -> int:
    return label // classes_per_task


def max_requests(n_tasks: int, level: str, classes_per_task: int = CLASSES_PER_TASK) -> int:
    """Longest possible stream: every task learned once, every item forgotten once."""
    n_items = n_tasks if level == "task" else n_tasks * classes_per_task
    return n_tasks + n_items


def make_stream(n_tasks: int, n_requests: int, seed: int, level: str = "task",
                classes_per_task: int = CLASSES_PER_TASK):
    """A random stream of n_requests requests: n_tasks learn requests + the rest unlearn.

    Every task is learned once (in random order). The unlearn requests are spread at
    random positions; each one forgets a random item that is currently known.
    Same arguments -> same stream.
    """
    if level not in ("task", "class"):
        raise ValueError(f"level must be 'task' or 'class', got {level!r}")
    n_unlearn = n_requests - n_tasks
    longest = max_requests(n_tasks, level, classes_per_task)
    if not n_tasks <= n_requests <= longest:
        raise ValueError(f"with {n_tasks} tasks at level '{level}', n_requests must be between "
                         f"{n_tasks} (learn only) and {longest} (forget everything); got {n_requests}")
    items_per_task = 1 if level == "task" else classes_per_task
    rng = np.random.RandomState(seed)

    # 1. Where do the U's go? Shuffle L/U markers until no U comes before there is
    #    something to forget (at most items_per_task forgettable items per learned task).
    for _ in range(100_000):
        kinds = np.array(["L"] * n_tasks + ["U"] * n_unlearn)
        rng.shuffle(kinds)
        n_l = np.cumsum(kinds == "L")
        n_u = np.cumsum(kinds == "U")
        if np.all(n_u <= n_l * items_per_task):
            break
    else:
        raise RuntimeError("could not place the unlearn requests; try fewer of them")

    # 2. Fill in which task is learned and which known item is forgotten.
    learn_order = rng.permutation(n_tasks).tolist()
    known = []                       # items learned and not yet forgotten
    stream = []
    for kind in kinds:
        if kind == "L":
            t = learn_order.pop(0)
            stream.append(("L", int(t)))
            known += [t] if level == "task" else list(range(t * classes_per_task, (t + 1) * classes_per_task))
        else:
            item = known.pop(rng.randint(len(known)))
            stream.append(("U", int(item)))
    check_stream(stream, n_tasks, level, classes_per_task)
    return stream


def check_stream(stream, n_tasks: int, level: str, classes_per_task: int = CLASSES_PER_TASK):
    """Raise ValueError with a clear message if the stream breaks a rule. Returns True if fine."""
    learned_tasks, forgotten = set(), set()
    for pos, (kind, i) in enumerate(stream):
        if kind == "L":
            if not 0 <= i < n_tasks:
                raise ValueError(f"request {pos}: L{i}: there is no task {i}")
            if i in learned_tasks:
                raise ValueError(f"request {pos}: L{i}: task {i} was already learned (each task is learned once)")
            learned_tasks.add(i)
        elif kind == "U":
            task = i if level == "task" else task_of(i, classes_per_task)
            if task not in learned_tasks:
                raise ValueError(f"request {pos}: U{i}: not learned yet, so it cannot be unlearned")
            if i in forgotten:
                raise ValueError(f"request {pos}: U{i}: already forgotten")
            forgotten.add(i)
        else:
            raise ValueError(f"request {pos}: unknown kind {kind!r} (use 'L' or 'U')")
    return True


def stream_to_str(stream) -> str:
    return " → ".join(f"{k}{i}" for k, i in stream)


def learning_after_forget(stream):
    """For each unlearn request: how many learn requests come after it.

    More later learning = more chances for the forgotten item to come back (relapse).
    Returns a list of (position, item, number of later L requests).
    """
    out = []
    for pos, (kind, i) in enumerate(stream):
        if kind == "U":
            out.append((pos, i, sum(1 for k, _ in stream[pos + 1:] if k == "L")))
    return out


def save_stream(stream, path, n_tasks: int, level: str, seed: int, classes_per_task: int = CLASSES_PER_TASK):
    """Write the stream and its settings to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    info = {"n_tasks": n_tasks, "classes_per_task": classes_per_task, "level": level, "seed": seed,
            "n_requests": len(stream)}
    # Settings first, then one request per line, so the file is easy to read by eye.
    head = json.dumps(info, indent=1)[:-2]
    body = ",\n  ".join(json.dumps([k, int(i)]) for k, i in stream)
    path.write_text(f'{head},\n "requests": [\n  {body}\n ]\n}}\n')


def load_stream(path):
    """Read a saved stream. Returns the settings dict; info["requests"] is a list of (kind, id) tuples.
    The stream is checked again on loading, so a hand-edited file with a mistake is caught."""
    info = json.loads(Path(path).read_text())
    info["requests"] = [(k, int(i)) for k, i in info["requests"]]
    check_stream(info["requests"], info["n_tasks"], info["level"], info["classes_per_task"])
    return info
