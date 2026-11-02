"""BadTeacher unlearning (Chundawat et al., AAAI 2023), shared version for Phase D on.

Program 09 explains it on CIFAR-10. Program 12 uses this version inside the CLU stream.

    student  = the model being unlearned (updated in place)
    good     = a frozen copy of the model before unlearning: the target on retain images
    bad      = a frozen, randomly initialized network: the target on forget images

    loss = KL(teacher || student), where each image's teacher is bad (forget) or good (retain)

BatchNorm stays frozen (student.eval()) while unlearning, as in programs 08 and 09.
"""

import torch
import torch.nn.functional as F

from common.models import resnet18_cifar


@torch.no_grad()
def favourite_class(model, images, among=None, chunk=500):
    """The class a model answers most often on these images.
    among: optional list of class ids; then only those outputs compete (e.g. one task's 10 classes)."""
    model.eval()
    logits = torch.cat([model(images[i:i + chunk]) for i in range(0, len(images), chunk)])
    if among is None:
        return int(logits.argmax(1).mode().values)
    among = torch.as_tensor(list(among), device=logits.device)
    return int(among[logits[:, among].argmax(1)].mode().values)


def make_bad_teacher(num_classes, probe_images, forbidden, seed, device, among=None, max_tries=100):
    """A random, never-trained ResNet-18, frozen.

    Random networks give almost every image the same answer (program 09, section 3), and they
    also rank the classes inside a task almost the same way for every image. If the favourite
    answer is a class we want to forget, the student would keep answering it, so such networks
    are rejected and the next seed is tried. The check is done over all outputs AND, if `among`
    is given (e.g. the task of a forgotten class), over those outputs only.
    For task-level forgetting every class of the task is forbidden, so pass among=None there.
    Returns (model, seed used, its favourite class over all outputs).
    """
    forbidden = set(int(c) for c in forbidden)
    probe_images = probe_images.to(device)
    for s in range(seed, seed + max_tries):
        with torch.random.fork_rng(devices=[]):    # don't disturb the main random sequence
            torch.manual_seed(s)
            model = resnet18_cifar(num_classes=num_classes)
        model = model.to(device).eval()
        fav = favourite_class(model, probe_images)
        fav_within = favourite_class(model, probe_images, among) if among is not None else None
        if fav not in forbidden and fav_within not in forbidden:
            for p in model.parameters():
                p.requires_grad_(False)
            return model, s, fav
        del model
    raise RuntimeError(f"no random network in {max_tries} tries avoided the forbidden classes")


def badteacher_loss(student_logits, good_logits, bad_logits, is_forget, T=1.0):
    """KL(teacher || student) averaged over the batch, plus the forget and retain parts (NaN if empty)."""
    teacher_logits = torch.where(is_forget[:, None], bad_logits, good_logits)
    log_p_student = F.log_softmax(student_logits / T, dim=1)
    p_teacher = F.softmax(teacher_logits / T, dim=1)
    per_image = F.kl_div(log_p_student, p_teacher, reduction="none").sum(1)
    kl_f = per_image[is_forget].mean().item() if is_forget.any() else float("nan")
    kl_r = per_image[~is_forget].mean().item() if (~is_forget).any() else float("nan")
    return per_image.mean(), kl_f, kl_r


def badteacher_unlearn(student, good_teacher, bad_teacher, sample_forget, sample_retain,
                       steps, lr, device, T=1.0):
    """Run `steps` BadTeacher updates on `student`.

    sample_forget() -> a CPU batch of forget images.
    sample_retain() -> a CPU batch of retain images, or pass sample_retain=None when there is
                       no retain data (then only the forget part of the loss is used).
    Returns {"kl_forget": [...], "kl_retain": [...]}, one value per step.
    """
    good_teacher.eval()
    bad_teacher.eval()
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    hist = {"kl_forget": [], "kl_retain": []}
    for _ in range(steps):
        xf = sample_forget()
        xr = sample_retain() if sample_retain is not None else None
        x = xf if xr is None else torch.cat([xf, xr])
        x = x.to(device, non_blocking=True)
        is_forget = torch.zeros(len(x), dtype=torch.bool, device=device)
        is_forget[: len(xf)] = True

        with torch.no_grad():
            good_logits, bad_logits = good_teacher(x), bad_teacher(x)
        student.eval()                                   # BatchNorm frozen
        loss, kl_f, kl_r = badteacher_loss(student(x), good_logits, bad_logits, is_forget, T)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        hist["kl_forget"].append(kl_f)
        hist["kl_retain"].append(kl_r)
    return hist
