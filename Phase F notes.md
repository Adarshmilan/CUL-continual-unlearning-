# Phase F notes (started 2026-11-06)

## 1. Literature check: is Relapse-Guard already published?

Searched Nov 2026. Closest papers, and how we differ:

| Paper | What they do | Learn new tasks after forgetting? | Feature probe? | Overlap with us |
|---|---|---|---|---|
| UnCLe (Adhikari et al., arXiv 2509.17530) | Defines Relapse and Spill in CLU, task-level, hypernetwork method | yes | no | Our base paper. Their Relapse metric = abs(end − after U), which mixes relapse with later decay |
| UG-CLU (arXiv 2505.15178) | Task-agnostic CLU, class + sample level, purges the buffer | yes | no | Purges the buffer but never studies relapse |
| BID-LoRA (arXiv 2604.12686) | LoRA adapters for retain / learn / forget, "escape unlearning" of embeddings | yes | no | Studies leakage of retained knowledge, not relapse of forgotten knowledge |
| SAFER (CVPR 2026 Findings, arXiv 2604.19108) | "Forgetting reversal" over repeated unlearning; negative logit margin | **no**, unlearning only | no (cosine / DBI only) | Same phenomenon name, but no learning in between, no stored signature |
| CATA (arXiv 2605.18610) | Knowledge re-emergence in continual unlearning, task arithmetic | **no** | no | Same, unlearning only |
| FG-OrIU (arXiv 2601.13578) | SVD forgetting subspace per layer + gradient projection during unlearning | **no** | no (head retraining, image reconstruction) | **Closest to our programs 19 / layer guard.** They project during the unlearning updates; we project during later *learning* |
| "An Illusion of Unlearning?" (Gao et al., AISTATS 2026, arXiv 2604.08271) | Linear probes show forgotten classes survive in features after offline unlearning; retrain also has high probe accuracy | no (offline) | **yes** | Our "output forgetting ≠ feature forgetting" is known *offline*. Their point that retrain is high too is why we compare with a retrain reference |
| DECAF (arXiv 2607.23934), Gradient Concentration (arXiv 2607.21353), Source-Free Class Relearning (arXiv 2609.02018) | Representation-level class unlearning, probe/centroid attacks | no (offline) | yes | Tools for evaluating feature forgetting; none in CLU streams |
| GPM (Saha et al., ICLR 2021), PGU (2023) | Gradient projection to protect old/retained tasks | – | – | We use the GPM idea the other way round: protect the *forgotten* state |

**What still looks new (not found in any paper):**

1. Measuring **feature-level relapse inside learn+unlearn streams**, against a retrain reference run on the *same stream* (row by row).
2. **Splitting relapse into causes**: buffer leakage vs backward transfer from similar classes (program 15: +62.5 vs +7.2).
3. The **tie-winning mechanism** of class-level relapse under BadTeacher (program 15 calibration).
4. The **Relapse metric flaw** (end − after mixes relapse and decay; use peak rise / excess over reference).
5. A **guard applied during later learning** (feature-subspace or weight-level), not during unlearning. FG-OrIU is close; we must cite it and show the difference.

**What is NOT new:** head reset, buffer purge, probes for offline unlearning, gradient projection itself.

## 2. New code (Phase F)

- `common/clu.py`:
  - `reference: True`: the retrain target run on the *same* stream (forgotten items never learned, U only marks them), so every row lines up with the real run.
  - `guard_layers`: weight-level guard (GPM used the other way round). After U it stores the input directions where the forgotten classes have > `layer_ratio` × the kept classes' energy, for every conv in the listed blocks. Later weight gradients have those directions removed.
  - `feature_steps`: feature unlearning after BadTeacher. Pulls forget features to the kept classes' mean and distils kept features to the model from before this U.
- `programs/phase_f_queue.sh`, `programs/run_queue.sh <folder>`: run every config in `results/configs/<folder>/`, skipping finished ones. Logs go to `logs/`.
- `programs/phase_f_analysis.py`: tables against the reference ("excess" = probe(method) − probe(reference) on the same rows).
- `programs/phase_f_layer_probe.py`: probes task 0 at every ResNet block before and after U.

## 3. Results so far (1 seed each; queue stopped early on 2026-11-06 because the laptop ran low on memory)

Worst case (L0, U0, then L1..L9), probe on task 0:

| run | after U | after L1 | end | excess end | excess rise |
|---|---|---|---|---|---|
| retrain reference | 34.5 | 47.7 | 62.9 | – | – |
| simple fixes | 31.2 | 53.8 | 67.4 | +4.5 | +7.8 |
| projection guard (19/20) | 28.9 | 52.3 | 65.1 | +2.2 | +7.8 |
| layer guard (new) | 31.0 | 56.5 | 64.7 | +1.8 | +5.3 |

- **Most of the "+36 relapse" seen in program 20 is normal learning.** A model that never saw task 0 rises 34.5 → 62.9 too. Without a reference, probe relapse is overstated about 5×.
- **Relapse that does happen = hidden knowledge coming back.** Right after U, the penultimate probe is *below* never-trained (31 vs 34.5): it looks forgotten. One task later it is 6–9 points *above* retrain. The knowledge was hidden (probably in earlier layers, which `phase_f_layer_probe.py` tests), not erased.
- Task15 seed 0: forgotten tasks start +17.5 above the reference after U, and the gap *shrinks* to +6.4 at the end (rise −11). There, the problem is leftover knowledge that was never erased, not relapse.
- The guards change the end excess by 2–3 points: too small to call on one seed.

### Task15, 3 seeds (2026-11-06)

| method | RA | FA | Relapse | probe end | ref end | excess after U | excess end |
|---|---|---|---|---|---|---|---|
| simple fixes | 77.4±1.3 | 11.0±2.1 | 2.9±1.1 | 75.7±0.4 | 68.3±1.7 | 18.4±3.4 | 7.3±1.4 |
| projection | 77.8±1.1 | 9.7±1.2 | 1.9±0.3 | 75.2±0.3 | 68.3±1.7 | 18.0±3.2 | 6.8±1.5 |
| layer guard | 77.9±1.6 | 8.7±0.4 | 1.5±0.5 | 75.8±0.4 | 68.3±1.7 | 18.5±2.8 | 7.5±1.3 |

- After head reset, FA is at chance (10%) for every method, so FA/Relapse differences are noise around chance.
- Feature level: forgotten tasks stay +7 above the retrain reference at the end, and no guard changes it. The gap *shrinks* after U (rise −11): no feature relapse on this stream, only knowledge that was never erased. The step to fix is the unlearning step (feature unlearning), not the guard.
- Bug fixed: class-level runs crashed when a task's last class was forgotten (no bad teacher could avoid all of the task's classes). Now the within-task check is skipped then.

## 4. Final results (all runs finished 2026-11-06 14:29)

Methods: **fixes** = BadTeacher + buffer purge + head reset + logit mask (program 17). **proj** = fixes + feature-projection guard (19/20). **layer** = fixes + weight-level layer guard (new). **feat** = fixes + feature unlearning after BadTeacher (new). **ref** = retrain reference on the same stream. "Excess" = probe(method) − probe(ref) on the forgotten items, same rows.

### Task-level, 15 requests (3 seeds; feat 1 seed)

| method | RA | FA | Relapse | probe end | excess after U | excess end |
|---|---|---|---|---|---|---|
| fixes | 77.4±1.3 | 11.0±2.1 | 2.9±1.1 | 75.7±0.4 | 18.4±3.4 | 7.3±1.4 |
| proj | 77.8±1.1 | 9.7±1.2 | 1.9±0.3 | 75.2±0.3 | 18.0±3.2 | 6.8±1.5 |
| layer | 77.9±1.6 | 8.7±0.4 | 1.5±0.5 | 75.8±0.4 | 18.5±2.8 | 7.5±1.3 |
| feat | 80.0 | 10.1 | 1.5 | 76.6 | 17.9 | 6.9 |
| ref end probe | | | | 68.3±1.7 | | |

### Worst case: L0, U0, then L1..L9 (3 seeds; feat_layer and proj 1 seed)

| method | RA | probe after U | probe end | excess after U | excess end | excess rise |
|---|---|---|---|---|---|---|
| fixes | 78.1±1.0 | 40.7±6.7 | 65.1±1.7 | 6.0±6.6 | 2.4±1.6 | −3.6±8.0 |
| proj | 78.8 | 28.9 | 65.1 | −5.6 | 2.2 | 7.8 |
| layer | 78.3±0.5 | 42.1±7.8 | 65.7±2.4 | 7.4±7.7 | 3.0±2.6 | −4.3±7.1 |
| feat | 79.2±0.5 | 21.7±4.6 | 64.0±1.2 | −13.0±4.6 | 1.3±1.2 | **14.4±4.4** |
| feat + layer | 79.3 | 13.9 | 64.3 | −20.6 | 1.4 | 22.0 |
| ref end probe | | | 62.6±0.2 | | | |

Stronger layer guard (layers 2–4, ratio 1, up to 25%): probe end 65.3 (+2.4), head on task 0 drifts to 17% (chance 10%). No better.

### Class-level, 30 requests (1 seed; probe = class vs rest of its task, chance 50)

| method | RA | FA | Relapse | probe after U | probe end | ref end | excess end |
|---|---|---|---|---|---|---|---|
| fixes | 74.7 | 7.3 | 5.2 | 84.9 | 86.6 | 86.0 | +0.6 |
| layer | 74.0 | 4.4 | 3.7 | 84.4 | 86.1 | 86.0 | +0.2 |

### Where the forgotten task hides (`phase_f_layer_probe.py`, seed 0): probe on task 0 at each block

| | layer1 | layer2 | layer3 | layer4 |
|---|---|---|---|---|
| never-trained | 43.9 | 44.6 | 39.9 | 34.5 |
| before U | 52.1 | 54.4 | 55.3 | 56.9 |
| after U (BadTeacher + fixes) | 52.6 | 53.6 | 55.1 | 33.0 |
| after U (+ feature unlearning) | 52.4 | 52.7 | 51.0 | **10.0** |

## 5. Conclusions

1. **Unlearning only touches the last block.** After BadTeacher, layers 1–3 hold task 0 exactly as before (layer3: 55.1 vs 55.3 before, 39.9 never-trained). Only layer4 looks forgotten. Even feature unlearning, which drives layer4 to chance (10%), leaves layer3 at 51.0.
2. **That is the cause of feature relapse.** One task of later learning lifts the hidden knowledge back up to layer4. Feature unlearning goes from 21.7 to about 14 points above where it started, and ends above retrain. The more the last layer is wiped, the bigger the bounce (feat_layer: +22).
3. **Guards during later learning can't fix this, by construction.** A guard keeps the network as it was right after unlearning. If the knowledge is still in layers 1–3 then, the guard protects it. Projection and layer guards change the end excess by ≤ 1 point (within seed noise).
4. **Without a retrain reference, relapse is overstated about 5–10×.** A never-trained task's probe rises 34.5 → 62.6 just from learning other tasks. Real excess at the end is +1 to +3 (worst case), +7 (task15), +0.2 to +0.6 (class-level).
5. **After head reset, output metrics are at chance for every method**, so FA and Relapse can't rank methods any more. Only feature-level metrics can.
6. Class-level: single forgotten classes end level with retrain (+0.6). Leftover knowledge fades on its own as more tasks are learned.

### What this means for the paper

- The method claim "the guard stops feature relapse" is **not supported**. Don't build the paper on it.
- A strong **analysis paper** is supported: (a) the retrain-reference protocol for CLU, (b) unlearning is shallow (layer-wise probes), and shallow unlearning + later learning = relapse, (c) guards preserve hidden knowledge rather than block it, (d) output metrics saturate after head reset, (e) the relapse causes from program 15 (buffer +62.5 vs transfer +7.2).
- The method direction this points to: **unlearning that reaches layers 1–3** at U time, e.g. feature unlearning applied at every block's output (target: each block's probe near never-trained / retrain), then optionally a guard to keep it that way. Test it with the layer probe first: it takes 2 minutes.

## 6. Roadmap Phase F items NOT done yet

- 21: other baselines (SalUn / gradient ascent, UnCLe numbers), MIA in the table.
- 22: long streams (class100 exists; about 1.5 h per run with the layer guard).
- 23: k and λ sweeps (λ partly done in 18).
- 24: trade-off test on similar new classes.
- 25: Tiny-ImageNet (dataset not downloaded).
- 26: paper figures.
- More seeds for class-level and feature unlearning on task15.
