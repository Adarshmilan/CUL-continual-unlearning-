"""Small helpers shared by every program: seeds, device, paths and run logging."""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch

# Project root = the folder that holds data/, programs/, common/, results/
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


def set_seed(seed: int) -> None:
    """Fix every random number generator so a run can be repeated exactly (roadmap rule 2)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log_run(program: str, settings: dict, results: dict) -> None:
    """Append one line (settings + results) to results/runs.jsonl, so every run is recorded."""
    RESULTS_DIR.mkdir(exist_ok=True)
    entry = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "program": program,
        "settings": settings,
        "results": results,
    }
    with open(RESULTS_DIR / "runs.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")
