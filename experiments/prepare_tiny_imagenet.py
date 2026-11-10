"""Make data/tiny100_32.npz: the first 100 Tiny-ImageNet classes (sorted wnids), resized to 32x32.

Run once (about 5 minutes, needs ~250 MB download + 160 MB output):
    python experiments/prepare_tiny_imagenet.py
If you already have tiny-imagenet-200.zip, put it in data/ first and the download is skipped.
Images are read straight from the zip (no 1 GB folder extract).
"""

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ZIP = DATA / "tiny-imagenet-200.zip"
OUT = DATA / "tiny100_32.npz"
URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
N_CLASSES = 100


def load(zf, name):
    img = Image.open(io.BytesIO(zf.read(name))).convert("RGB")
    return np.asarray(img.resize((32, 32), Image.BICUBIC), dtype=np.uint8)


def main():
    if OUT.exists():
        print(f"{OUT} already exists")
        return
    DATA.mkdir(exist_ok=True)
    if not ZIP.exists():
        print(f"downloading {URL} ...")
        urllib.request.urlretrieve(URL, ZIP)
    with zipfile.ZipFile(ZIP) as zf:
        wnids = sorted(zf.read("tiny-imagenet-200/wnids.txt").decode().split())[:N_CLASSES]
        pos = {w: i for i, w in enumerate(wnids)}
        names = zf.namelist()
        xtr, ytr = [], []
        for w in wnids:
            files = sorted(n for n in names if n.startswith(f"tiny-imagenet-200/train/{w}/images/") and n.endswith(".JPEG"))
            xtr += [load(zf, n) for n in files]
            ytr += [pos[w]] * len(files)
            print(f"\rtrain: {len(ytr)} images", end="")
        ann = zf.read("tiny-imagenet-200/val/val_annotations.txt").decode().splitlines()
        xte, yte = [], []
        for line in ann:
            f, w = line.split("\t")[:2]
            if w in pos:
                xte.append(load(zf, f"tiny-imagenet-200/val/images/{f}"))
                yte.append(pos[w])
        print(f"\ntest: {len(yte)} images")
        words = dict(l.split("\t") for l in zf.read("tiny-imagenet-200/words.txt").decode().splitlines())
    xtr, xte = np.stack(xtr), np.stack(xte)
    mean = xtr.reshape(-1, 3).mean(0) / 255
    std = xtr.reshape(-1, 3).std(0) / 255
    classes = np.array([words.get(w, w).split(",")[0] for w in wnids])
    np.savez(OUT, x_train=xtr, y_train=np.array(ytr), x_test=xte, y_test=np.array(yte),
             mean=mean, std=std, classes=classes, wnids=np.array(wnids))
    print(f"saved {OUT}: train {xtr.shape}, test {xte.shape}, mean {mean.round(3)}, std {std.round(3)}")


if __name__ == "__main__":
    sys.exit(main())
