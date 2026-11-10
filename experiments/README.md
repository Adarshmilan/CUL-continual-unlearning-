# Paper experiments

**Paper:** *Unlearning in continual learning is shallow.* Forgotten knowledge stays in the early layers and comes back during later learning, and the usual metrics can't see it.

Two laptops (RTX 3050, 4 GB) share the work:

| Folder | Who | What | Runs | Time |
|---|---|---|---|---|
| `adarsh/` | Adarsh | CIFAR-100: every unlearning method, seeds, class level, causes of relapse | 57 | ~13–14 h (2 nights) |
| `arpan/` | Arpan | Tiny-ImageNet-100 (2nd dataset), ER (2nd learner), 100-request stream | 35 | ~12 h (2 nights) |

Every run saves into **`results/paper/`** with a unique name, so the two results folders can simply be copied together.

## What each experiment group proves

| Group | Claim in the paper |
|---|---|
| A1, B1, B3 (worst case, probe at every block) | Unlearning only changes the last block, for **every** method (BadTeacher, NegGrad+, random labels, SCRUB), dataset and learner. The knowledge in layers 1–3 comes back after the next task |
| A2, B2, B3 (15 requests) | Main table: output metrics are at chance for every method, while the features stay above the retrain reference |
| Phase F (already done, `results/paper/phase_f/`) | Guards (projection, layer guard) don't remove the hidden knowledge: they keep it |
| A3, B4 (class level, 30 and 100 requests) | Holds for single classes and long streams |
| A4 (causes) | Leftover buffer data vs learning a similar class: which brings the class back, measured against a reference |

## How to run (both people)

1. Get the latest code (`git pull`, or copy the folder).
2. Python 3.10+ with `torch torchvision numpy scikit-learn pillow matplotlib`. We use Python 3.14, torch 2.14, torchvision 0.29, scikit-learn 1.9.
3. Close other heavy programs (browser tabs, Jupyter kernels). One run needs about 3–4 GB of RAM.
4. Start your queue. Read your own README first (`adarsh/README.md`, `arpan/README.md`).

```
python experiments/run_all.py adarsh        # or: arpan
```

On Linux, this version also caps memory at 4 GB per run and keeps running after the terminal is closed:

```
setsid nohup ./experiments/run.sh adarsh > logs/run_adarsh.log 2>&1 < /dev/null & disown
grep -E "DONE|FAILED" logs/run_adarsh.log          # progress
```

You can stop it any time. Finished runs are skipped when you start it again.

## Putting the results together

When Arpan is done, he copies his `results/paper/` folder into Adarsh's `results/paper/`. Only the `*.npz` and `*_summary.json` files are needed. The `*_model.pt` files (45 MB each) are optional.

Then:

```
python experiments/paper_tables.py
```

This prints every table of the paper and writes `results/paper/tables.json`. It works with partial results too: the `n` column shows how many seeds each number is based on.

## Files

- `make_configs.py`: defines every experiment. Run it again only if you change the plan; it rewrites both `configs/` folders.
- `run_all.py` / `run.sh`: run one person's queue.
- `prepare_tiny_imagenet.py`: makes `data/tiny100_32.npz` (Arpan only).
- `paper_tables.py`: every table, from `results/paper/`.
- The method code is in `common/clu.py`. The new unlearning methods are in `common/unlearn_methods.py`.
