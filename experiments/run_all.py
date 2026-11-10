"""Run every experiment of one person, one at a time (works on Linux, Windows and macOS).

    python experiments/run_all.py adarsh          (or arpan)
    python experiments/run_all.py arpan --only B1  (just one group)

Finished runs (results/paper/<name>_summary.json exists) are skipped, so you can stop it
(Ctrl+C) and start it again any time. Each run is a separate process, so memory is freed in
between. Logs: logs/paper/<name>.log. Progress: logs/run_<who>.txt.
On Linux, experiments/run.sh does the same plus a 4 GB memory cap per run.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("adarsh", "arpan"):
        sys.exit("usage: python experiments/run_all.py adarsh|arpan [--only GROUP]")
    who = sys.argv[1]
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    cfgs = sorted((ROOT / "experiments" / who / "configs").glob("*.json"))
    if only:
        cfgs = [c for c in cfgs if f"_{only}_" in c.name]
    (ROOT / "logs" / "paper").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "paper").mkdir(parents=True, exist_ok=True)
    progress = open(ROOT / "logs" / f"run_{who}.txt", "a")
    env = dict(os.environ, PYTHONPATH=str(ROOT))

    def say(msg):
        line = f"{time.strftime('%d %b %H:%M')}  {msg}"
        print(line, flush=True)
        progress.write(line + "\n")
        progress.flush()

    for i, c in enumerate(cfgs, 1):
        name = json.loads(c.read_text())["name"]
        short = name.split("/")[-1]
        if (ROOT / "results" / f"{name}_summary.json").exists():
            print(f"skip {short}")
            continue
        say(f"START  [{i}/{len(cfgs)}] {short}")
        t = time.time()
        with open(ROOT / "logs" / "paper" / f"{short}.log", "w") as log:
            code = subprocess.call([sys.executable, "-u", "-m", "common.clu", str(c)], cwd=ROOT, env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
        ok = code == 0 and (ROOT / "results" / f"{name}_summary.json").exists()
        say(f"{'DONE ' if ok else 'FAILED'} [{i}/{len(cfgs)}] {short}  ({(time.time() - t) / 60:.0f} min)"
            + ("" if ok else f"  -> see logs/paper/{short}.log"))
    say("ALL FINISHED")


if __name__ == "__main__":
    main()
