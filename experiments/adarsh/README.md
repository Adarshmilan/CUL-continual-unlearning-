# Adarsh's experiments (CIFAR-100): 57 runs, ~13–14 h

```
python experiments/run_all.py adarsh
```

Or detached on Linux: `setsid nohup ./experiments/run.sh adarsh > logs/run_adarsh.log 2>&1 < /dev/null & disown`

| Group | Runs | Each | Total | What |
|---|---|---|---|---|
| A1 | 18 | ~16 min | ~5 h | Worst case (forget task 0, learn 9 more), probe at **every block**: reference, BadTeacher, NegGrad+, random labels, SCRUB, BadTeacher + feature unlearning × 3 seeds |
| A2 | 15 | ~18 min | ~4.5 h | 15-request stream, every block: reference + 4 unlearning methods × 3 seeds |
| A3 | 6 | 15–47 min | ~3 h | Class level, 30 requests, seeds 1 and 2: reference, BadTeacher, layer guard |
| A4 | 18 | ~4 min | ~1.2 h | Causes: forget wolf, then learn a similar / dissimilar task, buffer kept / purged, + reference × 3 seeds |

To run one group only: `python experiments/run_all.py adarsh --only A1`. A1 is the most important group, so if time is short, run it first.

`cause_purged_similar_s0` was already run while testing, so it will be skipped.
