# Relapse-Guard Research Roadmap: 26 Programs

Oct 26, 2026 · @Adarsh

## How to use this roadmap

Build these 26 programs in order. Each one teaches one idea you need for the next, and together they take you from basic training to a full paper on **Relapse-Guard**.

**The research goal in one paragraph.** In Continual Learning-Unlearning (CLU), a model receives a stream of "learn this" and "forget this" requests. The base paper, UnCLe (Adhikari et al., 2025), showed that forgotten knowledge often *relapses*: it comes back when the model later learns similar tasks. Our method stops relapse in ordinary networks, at the level of single classes, using a stored "forget signature" and a guard loss during all later learning. We also check forgetting inside the features, not just at the output.

**Every program has five parts:**

- **What:** what the program does.
- **How:** the main steps to code it.
- **Why:** why it matters for the research.
- **Understanding note:** the one idea you must be able to explain in your own words before moving on.
- **Done when:** the result or plot that proves the program works.

**Setup**

- Python 3.10+, PyTorch, torchvision, numpy, matplotlib, scikit-learn (for PCA and the probe).
- One GPU is enough. Google Colab or Kaggle notebooks work for Phases A to D.
- Optional: the Mammoth library (aimagelab/mammoth on GitHub), written by the DER++ authors. Write programs 3 to 6 yourself first, then compare with Mammoth.
- Folder layout: `data/`, `programs/` (01 to 26), `common/` (shared model, data and metrics code), `results/` (plots and logs).

**Rules**

1. Do not move on until the current program's plot makes sense to you.
2. Fix random seeds and log every run (settings plus results) to a file.
3. Keep a lab notebook: one paragraph per program on what you saw and what surprised you. These notes become your paper.
4. Programs 3, 12 and 14 are the most important. Spend extra time there.

## Phase A: Basics (programs 1 to 2)

Phase A builds the training and data code that every later program reuses. Expect about one week.

### Program 01: `01_train_basic.py`, a basic classifier

- **What:** Train a ResNet-18 on all 10 classes of CIFAR-10 and report test accuracy.
- **How:** Load CIFAR-10 with torchvision (random crop plus horizontal flip for training). Use `torchvision.models.resnet18(num_classes=10)` and change the first conv to 3×3 stride 1 for 32×32 images. Train with SGD (lr 0.1, momentum 0.9) or Adam for 20 to 30 epochs. Write separate `train_one_epoch()` and `evaluate()` functions and save the model with `torch.save`.
- **Why:** Every later program is this loop plus extra loss terms. If this loop is clean, everything after is easier.
- **Understanding note:** A model is a function whose weights are shaped by the loss. Changing the loss is how you control what the model learns, and later, what it forgets.
- **Done when:** Test accuracy is roughly 85% or higher, and you can reload the saved model and get the same number.

### Program 02: `02_split_dataset.py`, task splits

- **What:** Split CIFAR-100 into 10 tasks of 10 classes each. Write a function that returns the train and test loader for any task id.
- **How:** Fix a class order with a seed. Task t holds classes 10t to 10t+9. Use `torch.utils.data.Subset` with index lists. Also write `get_class_loader(c)`, which you will need for unlearning single classes. Put all this in `common/data.py`.
- **Why:** Continual learning is defined by *what data the model sees when*. This file controls that for the whole project.
- **Understanding note:** In class-incremental learning, one output head covers all classes seen so far, and the model is never told which task a test image came from. That is harder than task-incremental learning, where the task id is given.
- **Done when:** Each task has 5,000 train and 1,000 test images, and no class appears in two tasks.

## Phase B: Continual learning (programs 3 to 6)

Phase B ends with DER++, the base learner for the whole project. Expect about two weeks.

### Program 03: `03_naive_sequential.py`, catastrophic forgetting

- **What:** Train on task 0, then task 1, up to task 9, with no protection at all.
- **How:** For each task, train a few epochs on that task only. After each task, test on *every* task seen so far and store the results in an accuracy matrix `acc[after_task][on_task]`. Plot one line per task.
- **Why:** This is the problem continual learning exists to solve. You need to see it yourself.
- **Understanding note:** The weights that solved task 0 get overwritten by gradients from task 1. Nothing in plain training protects old knowledge. Notice too that forgetting here is *accidental*, while unlearning is forgetting *on purpose*.
- **Done when:** Task 0 accuracy drops close to 0 after a few tasks, and you can read the accuracy matrix.

### Program 04: `04_ewc.py`, Elastic Weight Consolidation

- **What:** Add EWC, which penalizes changes to weights that were important for old tasks.
- **How:** After each task, estimate the diagonal Fisher information (average squared gradient of the log-likelihood per weight) and store a copy of the weights. While training later tasks, add λ·Σ F\_i·(θ\_i − θ\_old\_i)².
- **Why:** EWC is the classic regularization approach. The 2023 paper in your sheet even used EWC for unlearning, and the Fisher idea returns in unlearning methods like SSD.
- **Understanding note:** Some weights matter more than others for a task. Measuring that importance lets you protect knowledge, or later target it for removal.
- **Done when:** Forgetting is clearly lower than in program 03. Expect EWC to help less in class-incremental learning; note why.

### Program 05: `05_replay.py`, experience replay

- **What:** Keep a small memory buffer of old images (for example 2,000) and mix them into every training batch.
- **How:** Fill the buffer with reservoir sampling so every image seen has an equal chance of being kept. Each step, compute loss on the new batch plus a buffer batch.
- **Why:** Replay is the strongest simple approach. The buffer is also central to our problem: UnCLe found that old buffer data is one cause of relapse.
- **Understanding note:** Replay works because the model keeps seeing old data. That means any data still in the buffer can *teach the model again*, including data you were asked to delete.
- **Done when:** Average accuracy across all tasks clearly beats programs 03 and 04.

### Program 06: `06_derpp.py`, DER++ (our base learner)

- **What:** Upgrade replay to DER++, which also stores the model's logits (raw outputs) for each buffered image.
- **How:** Buffer entries become (image, label, logits). Loss = CE(new batch) + α·CE(buffer labels) + β·MSE(current logits, stored logits). Start with α = 0.5 and β = 0.5, buffer size 2,000. Save the final code as `common/derpp.py`.
- **Why:** UnCLe adapted all its baselines with DER++, so using it makes your results directly comparable.
- **Understanding note:** Matching old logits (distillation) preserves how the model *behaved*, not just its labels. That keeps richer knowledge, and it also means forgotten behavior can be distilled back in by mistake.
- **Done when:** One plot compares the average accuracy of 03, 04, 05 and 06 after each task, with DER++ best.

## Phase C: Unlearning, offline first (programs 7 to 10)

Phase C teaches unlearning on a normal, fully trained model before mixing it with continual learning. Use CIFAR-10 and forget one class. Expect about two weeks.

### Program 07: `07_retrain_baseline.py`, the gold standard

- **What:** Train a model from scratch on all CIFAR-10 classes except one (say class 3, "cat").
- **How:** Reuse program 01 with class 3 removed from training. Save it as `retrain_model.pt`.
- **Why:** Exact unlearning means "behave as if never trained on that data". The retrained model is that ideal, and every method is judged by how close it gets.
- **Understanding note:** Retraining is perfect but too expensive to repeat on every deletion request. All approximate unlearning methods trade some of that perfection for speed.
- **Done when:** Cat accuracy is 0% (the class was never learned) and the other classes are near program 01.

### Program 08: `08_simple_unlearn.py`, gradient ascent and random labels

- **What:** Start from program 01's model and remove class 3 with two simple methods.
- **How:** (a) Gradient ascent: maximize the loss on class 3 for a few steps. (b) Random labels: fine-tune on class 3 images with random wrong labels, mixed with normal batches from the other classes. Track Forget Accuracy (FA, on class 3) and Retain Accuracy (RA, other classes) after every step.
- **Why:** These are the basic tools that more advanced methods build on. Gradient ascent also shows how easily unlearning damages the rest of the model.
- **Understanding note:** Unlearning always pulls two ways: forget the target, but keep everything else. Pushing too hard on forgetting destroys retained knowledge.
- **Done when:** You have FA and RA curves for both methods and can point to where each one starts damaging RA.

### Program 09: `09_badteacher.py`, a real baseline

- **What:** Implement BadTeacher-style unlearning (Chundawat et al., 2023).
- **How:** Use two teachers: the trained model ("good") and a randomly initialized model ("bad"). The student starts as a copy of the trained model. On retain data, minimize KL to the good teacher. On forget data, minimize KL to the bad teacher.
- **Why:** UnCLe uses BadTeacher as a main baseline, and UniCLUN is built on it. You will reuse this code in Phase D.
- **Understanding note:** "Forgetting" can be defined as "answer like a random network would", which is gentler than gradient ascent and keeps the model stable.
- **Done when:** FA drops near chance level (about 10% for 10 classes), with RA close to the original.

### Program 10: `10_mia.py`, membership inference attack

- **What:** Build a simple attack that guesses whether an image was in the training set.
- **How:** Use the model's loss or confidence as a feature. Train a logistic regression to tell training images (label 1) from test images (label 0). Then apply it to the forgotten class: a score near 50% means the model looks like it never saw that data.
- **Why:** Low forget accuracy does not prove the data is gone. MIA is the standard privacy check, and UnCLe reports it.
- **Understanding note:** Accuracy is what the model *says*, while MIA probes what it *remembers*. Hold on to this idea, because program 14 extends it to the features.
- **Done when:** MIA is high on the original model's class 3, and closer to 50% on the retrain model and after BadTeacher.

## Phase D: Reproduce the CLU problem (programs 11 to 15)

Phase D proves the problem exists and finds its causes; this becomes the motivation section of your paper. Back to Split CIFAR-100 with DER++. Expect about two to three weeks.

### Program 11: `11_request_stream.py`, the request stream

- **What:** Generate random sequences of learn and unlearn requests, such as `L3 → L0 → U3 → L9 → U0 …`.
- **How:** A function `make_stream(n_tasks, n_requests, seed, level)` returns a list of `("L", id)` or `("U", id)`. Rules: you can only unlearn something already learned, and a forgotten item is never relearned. Support `level="task"` (forget a whole task) and `level="class"` (forget one class). Save streams to JSON so every method runs on identical sequences.
- **Why:** This *is* the CLU setting. Fixed, saved streams make comparisons fair.
- **Understanding note:** Order matters. Forgetting something early and then learning many similar tasks is the worst case for relapse.
- **Done when:** You can generate, save and reload streams of 15, 30 and 100 requests.

### Program 12: `12_clu_naive.py`, see spill and relapse

- **What:** Run DER++ with BadTeacher unlearning on a stream, and track every task's accuracy after every request.
- **How:** On `L`, train with DER++. On `U`, run BadTeacher using the forget class's buffer samples as forget data and the rest of the buffer as retain data, then remove those samples from the buffer. Log a full accuracy matrix (request × task). Plot task 0's accuracy across the whole sequence.
- **Why:** This reproduces UnCLe's key figure. Seeing it yourself is what turns the paper into *your* problem.
- **Understanding note:** Two separate failures. **Spill:** unlearning one task drops accuracy on others. **Relapse:** a forgotten task's accuracy climbs back after later learning.
- **Done when:** Your plot clearly shows at least one spill drop and one relapse rise.

### Program 13: `13_metrics.py`, the four metrics

- **What:** Turn the accuracy matrix into numbers you can put in a table.
- **How:** Write functions for: **RA**, average final accuracy on retained tasks; **FA**, average final accuracy on forgotten tasks (closer to chance is better); **Spill**, for each unlearn step, the sum of absolute accuracy changes on all other tasks; **Relapse**, for each forgotten task, |accuracy right after unlearning − accuracy at the end|. Put them in `common/metrics.py` and test them on a tiny hand-made matrix.
- **Why:** These are UnCLe's metrics. Using the same definitions lets you compare directly with their results.
- **Understanding note:** No single number is enough. A method can score a great FA simply by wrecking the whole model, which RA and Spill would expose.
- **Done when:** Your functions give the expected values on the hand-made matrix, then report all four for program 12.

### Program 14: `14_feature_probe.py`, check the hidden features

- **What:** Check whether the forgotten class is still recognizable inside the network's features.
- **How:** Freeze the model at the end of the stream. Extract penultimate-layer features for held-out images of the forgotten class plus some other classes. Train a small linear classifier (the probe) on half of them and test it on the other half. Compare with the same probe on a never-trained network and on a retrained-without-that-class model.
- **Why:** Once you reset a class's output head, output accuracy is trivially 0%, which proves nothing. The probe tests whether the knowledge is truly gone. This can be a contribution of its own.
- **Understanding note:** "The model can't output it" and "the model doesn't know it" are different claims. Only the second is real unlearning.
- **Done when:** You have one probe-accuracy number per method, and you can say whether program 12's forgotten class survives in the features.

### Program 15: `15_relapse_causes.py`, find out why relapse happens

- **What:** Run controlled experiments that separate the causes of relapse.
- **How:** Compare four conditions: (a) forget-class samples left in the buffer vs. purged; (b) the later task is *similar* to the forgotten class (for example forget "wolf", then learn "dog") vs. *dissimilar* (forget "wolf", then learn "truck"). Use CIFAR-100's superclasses to pick similar pairs. Measure relapse with both output accuracy and the probe.
- **Why:** Your method must target the real cause. This experiment also motivates each part of Relapse-Guard.
- **Understanding note:** Two routes to relapse. Buffer leakage is easy to fix by deleting data. Backward transfer from similar tasks is the hard one, and it is what the guard loss targets.
- **Done when:** A small table shows relapse for all four conditions, and you can explain which cause is bigger.

## Phase E: Our novelty, Relapse-Guard (programs 16 to 20)

Phase E builds the method one piece at a time, so you can measure what each piece adds. Expect about three weeks. The full training loss during later learning is:

```latex
L = \mathrm{CE}_{\text{new}} + \alpha\,\mathrm{CE}_{\text{buffer}} + \beta\,\mathrm{MSE}_{\text{logits}} + \lambda \sum_{c \in \text{forgotten}} \lVert U_c^{\top} h(x) \rVert^2
```

Here h(x) is the penultimate-layer feature vector of an input, and U\_c holds the stored feature directions of forgotten class c.

### Program 16: `16_signature.py`, the forget signature

- **What:** Before deleting a class, record the main directions its features occupy.
- **How:** Pass the class's buffer samples through the network and collect penultimate features (512-dim for ResNet-18). Center them and run PCA. Keep the top k directions as a 512×k matrix U\_c (start with k = 5). Also print how much variance those k directions explain.
- **Why:** This small matrix is the only thing you keep after deleting the data. It lets you guard against the class returning without storing any of its images.
- **Understanding note:** A class lives in a few directions of feature space. If later learning stops rebuilding those directions, the class can't come back.
- **Done when:** You can save and reload U\_c, and a handful of directions explain a large share of the class's feature variance.

### Program 17: `17_buffer_purge_head_reset.py`, the simple fixes

- **What:** On each unlearn request, run unlearning, delete the class from the buffer, and reset its row in the classifier head.
- **How:** Run BadTeacher (program 09) on the class's buffer samples, then remove them from the buffer, along with their stored logits. Set that class's head weights to a fresh random init and keep it frozen. Run the full stream and measure all four metrics plus the probe.
- **Why:** This is the honest baseline for your method. If these simple fixes already solve relapse, the guard is not needed, so you must measure them first.
- **Understanding note:** These steps fix relapse through the buffer and at the output. They do nothing about the features slowly rebuilding the class, which the probe will reveal.
- **Done when:** Output-level relapse drops compared with program 12, but probe accuracy for forgotten classes still rises during later learning.

### Program 18: `18_guard_loss.py`, the relapse guard

- **What:** Add the penalty λ·Σ‖U\_cᵀh(x)‖² to every training step after a class is forgotten.
- **How:** Keep a list of stored U\_c matrices. In the DER++ loss, compute features for the current batch, project them onto each U\_c, and add the squared norm times λ. Start with λ in {0.01, 0.1, 1}.
- **Why:** This is the main contribution: it directly blocks the backward-transfer route to relapse found in program 15.
- **Understanding note:** The guard tells the network, "learn the new task, but don't use the forgotten class's feature directions to do it."
- **Done when:** Probe-level relapse drops clearly compared with program 17, while RA stays within a few points.

### Program 19: `19_guard_gradproj.py`, the gradient-projection variant

- **What:** Replace the penalty with a hard constraint that removes forbidden directions from the gradients.
- **How:** Following the GPM paper (Saha et al., 2021), project each layer's gradient so its update cannot push features into U\_c. Start with the last layer only, then extend to more layers.
- **Why:** A penalty can be outweighed by a strong task loss; a projection cannot. Comparing the two is a natural ablation.
- **Understanding note:** A soft penalty trades forgetting against learning through λ. A hard projection guarantees the constraint but may block too much learning.
- **Done when:** You have a table comparing programs 18 and 19 on all metrics.

### Program 20: `20_relapse_guard_full.py`, the final method

- **What:** Combine the best pieces into one clean, configurable method.
- **How:** One config file selects buffer purge, head reset, guard type (penalty or projection), k and λ. The same script runs every stream and saves metrics to `results/`.
- **Why:** Phase F needs one reliable method that runs every experiment the same way.
- **Understanding note:** Each part has a job. Purge stops buffer leakage, head reset stops output relapse, and the guard stops feature relapse. The paper's ablations will prove each job.
- **Done when:** One command reproduces your best result on a fixed seed.

## Phase F: Experiments (programs 21 to 26)

Phase F produces every table and figure in the paper. Run everything on 3 seeds and report mean ± standard deviation. Expect about four weeks.

### Program 21: `21_run_baselines.py`, all methods on the same streams

- **What:** Run every baseline and your method on identical saved streams.
- **How:** Methods: DER++ with naive unlearning, DER++ with BadTeacher, DER++ with SalUn (or gradient ascent if SalUn is too much work), UnCLe (use their code if released; otherwise cite their reported numbers clearly as reported), and Relapse-Guard. Log RA, FA, Spill, Relapse, probe accuracy and MIA.
- **Why:** This is the main results table of your paper.
- **Understanding note:** A fair comparison means same data, same streams, same seeds, same tuning effort. An unfair comparison makes a paper worthless.
- **Done when:** A results CSV holds every method × metric × seed.

### Program 22: `22_long_sequences.py`, long streams

- **What:** Run streams of 50 and 100+ requests.
- **How:** Reuse program 21's runner with longer saved streams. Track how each metric changes as the stream grows.
- **Why:** UnCLe tested at most 30 requests, and BID-LoRA criticized earlier work for short sequences. Long streams are where relapse and plasticity collapse show up.
- **Understanding note:** A method that looks fine for 15 requests can slowly break over 100. Long-horizon stability is a key selling point.
- **Done when:** You have a plot of Relapse vs. stream length for each method.

### Program 23: `23_ablations.py`, what each piece contributes

- **What:** Turn off one part of Relapse-Guard at a time, and sweep its settings.
- **How:** Variants: no guard; no buffer purge; no head reset; penalty vs. projection. Sweeps: k in {1, 3, 5, 10, 20} and λ in {0.01, 0.1, 1, 10}.
- **Why:** Reviewers will ask "is every part needed?" Ablations answer that with evidence.
- **Understanding note:** If removing a part changes nothing, that part does not belong in the method. Be willing to cut it.
- **Done when:** An ablation table plus one sensitivity plot each for k and λ.

### Program 24: `24_tradeoff.py`, the honest-risk test

- **What:** Measure whether the guard hurts learning of new classes that are *similar* to forgotten ones.
- **How:** Forget a class, then learn a similar class from the same CIFAR-100 superclass. Compare the new class's accuracy with and without the guard, across λ values.
- **Why:** This is the method's main weakness. Measuring it yourself is far better than a reviewer finding it.
- **Understanding note:** Forgetting "wolf" should not stop the model from learning "dog". The question is how much shared feature space the guard blocks.
- **Done when:** A plot of forgotten-class probe accuracy vs. new-class accuracy, one point per λ: a trade-off curve.

### Program 25: `25_tinyimagenet.py`, a second dataset

- **What:** Repeat the main results and one ablation on Tiny-ImageNet (200 classes, 64×64 images).
- **How:** Same pipeline, with 20 tasks of 10 classes. Reduce epochs if compute is tight, and note it.
- **Why:** One dataset is weak evidence. A second shows your method generalizes, and UnCLe also used Tiny-ImageNet.
- **Understanding note:** Harder datasets have more classes sharing features, so relapse through similar classes should be stronger there.
- **Done when:** The main table exists for Tiny-ImageNet too.

### Program 26: `26_make_tables_plots.py`, paper figures

- **What:** Generate every table and figure in the paper from the saved results.
- **How:** Read all result CSVs and output LaTeX tables and PDF figures with consistent fonts, colors and method names.
- **Why:** Figures must be regenerated whenever results change. Automating this prevents copy-paste mistakes.
- **Understanding note:** Reproducibility starts here: anyone with your code and results folder should get the exact same figures.
- **Done when:** One command rebuilds every figure and table in the paper.

## Reading list, progress and next check-in

Read each paper when its phase starts, not all at once.

| When | Paper | Why read it |
| --- | --- | --- |
| Before Phase B | Buzzega et al., "Dark Experience for General Continual Learning" (DER++, NeurIPS 2020) | Our base learner, used in program 06 |
| Before Phase C | Chundawat et al., "Can Bad Teaching Induce Forgetting?" (BadTeacher, AAAI 2023) | Main unlearning baseline, program 09 |
| Before Phase D | [UnCLe: An Unlearning Framework for Continual Learning](https://arxiv.org/html/2509.17530v1) | **Base paper.** Setting, metrics and the relapse figure you reproduce |
| Before Phase D | [UniCLUN: A Unified Framework for Continual Learning and Unlearning](https://web3.arxiv.org/pdf/2408.11374) | First unified CLU method |
| Before Phase E | [UG-CLU: Task-agnostic Continual Learning-Unlearning](https://arxiv.org/abs/2505.15178v1) | Class- and sample-level CLU; a comparison point |
| Before Phase E | [BID-LoRA](https://arxiv.org/pdf/2604.12686) | Closest competitor. Confirm it doesn't already solve relapse |
| Before Phase E | Saha et al., "Gradient Projection Memory for Continual Learning" (GPM, ICLR 2021) | Basis for program 19 |

**Progress checklist**

- [ ] Phase A done (programs 01 to 02)
- [ ] Phase B done (programs 03 to 06)
- [ ] Phase C done (programs 07 to 10)
- [ ] Phase D done (programs 11 to 15)
- [ ] Phase E done (programs 16 to 20)
- [ ] Phase F done (programs 21 to 26)

**What to bring when you come back**

- The plot from your latest finished program, plus your lab-notebook paragraph about it.
- Anything that didn't match the "Done when" line, with the numbers you got.
- Questions you couldn't answer from the understanding notes.

The best checkpoints for a review are after program 06, after program 12 (once you see relapse yourself), and after program 18.
