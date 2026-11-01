"""Penultimate-layer features and linear probes (programs 14 to 16, and the guard later).

h(x) = the 512 numbers ResNet-18 computes just before its final linear layer (the head).
The head turns h(x) into class scores; the probe asks whether h(x) itself still separates a class.
"""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def penultimate(model, x):
    """h(x): the 512-dim feature vector of torchvision's ResNet-18, i.e. everything except model.fc."""
    m = model
    x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
    x = m.layer4(m.layer3(m.layer2(m.layer1(x))))
    return torch.flatten(m.avgpool(x), 1)


@torch.no_grad()
def features_of(model, X, chunk=500):
    """h(x) for a tensor of images already on the model's device. Returns a float32 CPU tensor [N, 512]."""
    model.eval()
    return torch.cat([penultimate(model, X[i:i + chunk]).float().cpu() for i in range(0, len(X), chunk)])


def probe_accuracy(feats, labels, folds=5, seed=0):
    """A linear probe: logistic regression on the features, cross-validated accuracy (mean, std).

    Train on (folds-1)/folds of the images, test on the rest, rotate. Chance = 1 / number of classes
    (for balanced labels).
    """
    feats = np.asarray(feats)
    labels = np.asarray(labels)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, feats, labels, cv=cv)
    return float(scores.mean()), float(scores.std())
