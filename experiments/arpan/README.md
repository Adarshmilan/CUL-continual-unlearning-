# Arpan's experiments (Tiny-ImageNet-100, ER learner, long stream): 35 runs, ~12 h

## One-time setup

1. Get the project folder (`git pull`, or a copy from Adarsh). CIFAR-100 downloads by itself on the first run.
2. Make the Tiny-ImageNet data (~250 MB download, about 5 minutes):
   ```
   python experiments/prepare_tiny_imagenet.py
   ```
   This creates `data/tiny100_32.npz`: the first 100 Tiny-ImageNet classes resized to 32×32. Faster: copy `data/tiny100_32.npz` (170 MB) from Adarsh's laptop instead.

## Run

```
python experiments/run_all.py arpan
```

Or detached on Linux: `setsid nohup ./experiments/run.sh arpan > logs/run_arpan.log 2>&1 < /dev/null & disown`

| Group | Runs | Each | Total | What |
|---|---|---|---|---|
| B1 | 12 | ~16 min | ~3.2 h | Tiny-ImageNet, worst case, probe at every block: reference, BadTeacher, NegGrad+, SCRUB × 3 seeds |
| B2 | 9 | ~18 min | ~2.8 h | Tiny-ImageNet, 15 requests: reference, BadTeacher, layer guard × 3 seeds |
| B3 | 12 | ~16 min | ~3.2 h | Second learner (plain experience replay instead of DER++): worst case + 15 requests, reference + BadTeacher × 3 seeds |
| B4 | 2 | ~1–1.5 h | ~3 h | Long stream: 100 class-level requests, reference + BadTeacher |

One group only: `python experiments/run_all.py arpan --only B1`

## When finished

Send Adarsh your `results/paper/` folder (the `*.npz` and `*_summary.json` files are enough). Also send `logs/run_arpan.txt` if anything shows FAILED.
