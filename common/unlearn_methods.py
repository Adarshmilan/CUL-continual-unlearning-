"""Other unlearning methods for the paper (BadTeacher lives in badteacher.py).

All of them follow the same rules as our BadTeacher runs, so they can be compared fairly:
Adam, `steps` updates, BatchNorm frozen (model.eval()), forget batch + retain batch per step.
When there is nothing to keep yet (the worst-case stream), only the forget term is used.

- "neggrad"  : NegGrad+ (gradient ascent on forget + descent on retain, Kurmanji et al. 2023 /
               program 08). The forget CE is clamped at `tau` per image, so ascent stops once an
               image is at about chance level instead of blowing up (program 08 diverged without it).
- "randlabel": Random labels (program 08 / Hayase et al.): forget images are trained towards
               random labels from the classes still kept, plus CE on retain.
- "scrub"    : SCRUB (Kurmanji et al., NeurIPS 2023), simplified to one combined step:
               KL(teacher || student) + CE on retain, minus the KL on forget (clamped at `tau`).
               The teacher is the model before this unlearning request.

sample_forget() / sample_retain() return (images, labels) CPU batches; sample_retain may be None.
"""

import torch
import torch.nn.functional as F


def _kl(student_logits, teacher_logits):
    return F.kl_div(F.log_softmax(student_logits, 1), F.softmax(teacher_logits, 1), reduction="none").sum(1)


def unlearn_other(method, model, teacher, sample_forget, sample_retain, steps, lr, device,
                  random_label_pool=None, tau=4.6):
    """Run `steps` updates of `method` on `model`. Returns the last (forget term, retain term)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.eval()                                     # BatchNorm frozen, like BadTeacher
    pool = None if random_label_pool is None else torch.as_tensor(random_label_pool, device=device)
    last = (float("nan"), float("nan"))
    for _ in range(steps):
        xf, yf = (t.to(device, non_blocking=True) for t in sample_forget())
        out_f = model(xf).float()
        if method == "neggrad":
            lf = -F.cross_entropy(out_f, yf, reduction="none").clamp(max=tau).mean()
        elif method == "randlabel":
            yr = pool[torch.randint(len(pool), (len(xf),), device=device)]
            lf = F.cross_entropy(out_f, yr)
        elif method == "scrub":
            with torch.no_grad():
                tf = teacher(xf).float()
            lf = -_kl(out_f, tf).clamp(max=tau).mean()
        else:
            raise ValueError(f"unknown unlearning method {method!r}")
        lr_term = torch.zeros((), device=device)
        if sample_retain is not None:
            xr, yr_true = (t.to(device, non_blocking=True) for t in sample_retain())
            out_r = model(xr).float()
            lr_term = F.cross_entropy(out_r, yr_true)
            if method == "scrub":
                with torch.no_grad():
                    tr = teacher(xr).float()
                lr_term = lr_term + _kl(out_r, tr).mean()
        loss = lf + lr_term
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last = (lf.item(), lr_term.item())
    return last
