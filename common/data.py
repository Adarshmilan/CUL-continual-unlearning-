"""All dataset code for the project.

- CIFAR-10 loaders (program 01, and later the offline unlearning programs 07-10).
- Split CIFAR-100: 10 tasks of 10 classes (program 02 onward).

Label convention for Split CIFAR-100
------------------------------------
We shuffle the 100 classes once with a fixed seed, then *rename* every class to
its position in that order. After renaming, task t simply holds labels
10t .. 10t+9. So "label 0" means "the first class in our order", not the
original CIFAR-100 class 0. `class_order[new_label]` gives back the original id.
"""

import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from common.utils import DATA_DIR

# Per-channel mean and std of the training images, used to normalize inputs.
CIFAR10_MEAN, CIFAR10_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN, CIFAR100_STD = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)

N_TASKS = 10
CLASSES_PER_TASK = 10


def _transforms(mean, std, train: bool):
    """Training images get random crop + horizontal flip (augmentation); test images do not."""
    steps = []
    if train:
        steps += [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    steps += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(steps)


def _loader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


# ---------------------------------------------------------------- CIFAR-10


def get_cifar10_loaders(batch_size=128, num_workers=4):
    """Return (train_loader, test_loader) for all 10 CIFAR-10 classes."""
    train_set = datasets.CIFAR10(DATA_DIR, train=True, download=True,
                                 transform=_transforms(CIFAR10_MEAN, CIFAR10_STD, train=True))
    test_set = datasets.CIFAR10(DATA_DIR, train=False, download=True,
                                transform=_transforms(CIFAR10_MEAN, CIFAR10_STD, train=False))
    return (_loader(train_set, batch_size, True, num_workers),
            _loader(test_set, batch_size, False, num_workers))


def get_cifar10_unlearning_loaders(forget_class: int, batch_size=128, num_workers=2):
    """CIFAR-10 split into a FORGET set (one class) and a RETAIN set (the other nine).

    Used by the offline unlearning programs 07-10. Returns a dict of loaders:

      train_retain        training images of the 9 kept classes (augmented, shuffled)
      train_forget        training images of the forgotten class (augmented, shuffled)
      train_retain_plain  same images, no augmentation, not shuffled (for measuring)
      train_forget_plain  same images, no augmentation, not shuffled (for measuring, MIA)
      test_retain         test images of the 9 kept classes
      test_forget         test images of the forgotten class
      test_all            all 10,000 test images

    plus "classes" (the 10 class names).

    Memory (the project runs on an 8 GB laptop):
    - Only the two augmented training loaders use worker processes (num_workers each).
      Every worker is a separate Python process with its own copy of PyTorch and the
      images, a few hundred MB each. Measuring loaders use none: no augmentation, so
      the main process is fast enough.
    - The plain and augmented training sets share one copy of the image array.
    """
    def load(train, augment):
        return datasets.CIFAR10(DATA_DIR, train=train, download=True,
                                transform=_transforms(CIFAR10_MEAN, CIFAR10_STD, train=augment))

    train_aug, test = load(True, True), load(False, False)
    train_plain = copy.copy(train_aug)        # shallow copy: shares the 150 MB image array
    train_plain.transform = _transforms(CIFAR10_MEAN, CIFAR10_STD, train=False)

    def split(ds):
        targets = np.asarray(ds.targets)
        forget = np.where(targets == forget_class)[0].tolist()
        retain = np.where(targets != forget_class)[0].tolist()
        return retain, forget

    tr_retain, tr_forget = split(train_aug)
    te_retain, te_forget = split(test)
    return {
        "train_retain": _loader(Subset(train_aug, tr_retain), batch_size, True, num_workers),
        "train_forget": _loader(Subset(train_aug, tr_forget), batch_size, True, num_workers),
        "train_retain_plain": _loader(Subset(train_plain, tr_retain), batch_size, False, 0),
        "train_forget_plain": _loader(Subset(train_plain, tr_forget), batch_size, False, 0),
        "test_retain": _loader(Subset(test, te_retain), batch_size, False, 0),
        "test_forget": _loader(Subset(test, te_forget), batch_size, False, 0),
        "test_all": _loader(test, batch_size, False, 0),
        "classes": test.classes,
    }


# ---------------------------------------------------------- Split CIFAR-100


def get_class_order(seed: int = 0) -> np.ndarray:
    """A fixed random order of the 100 original class ids. Same seed -> same order."""
    return np.random.RandomState(seed).permutation(100)


def task_classes(task_id: int) -> list[int]:
    """The (renamed) labels that belong to a task: task t holds 10t .. 10t+9."""
    start = task_id * CLASSES_PER_TASK
    return list(range(start, start + CLASSES_PER_TASK))


def class_to_task(label: int) -> int:
    return label // CLASSES_PER_TASK


class IndexedSubset(Dataset):
    """Like Subset, but each item is (image, label, index in the full dataset).

    Replay buffers (program 05 on) store that index instead of a copy of the image.
    It lives here, not in a notebook, so DataLoader worker processes can import it.
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        image, label = self.dataset[idx]
        return image, label, idx


class SplitCIFAR100:
    """Loads CIFAR-100 once and hands out loaders for any task or single class.

    Usage:
        data = SplitCIFAR100(seed=0)
        train_loader, test_loader = data.get_task_loaders(3)
        cat_loader = data.get_class_loader(42, train=False)
    """

    def __init__(self, seed: int = 0, batch_size: int = 32, num_workers: int = 2):
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.class_order = get_class_order(seed)

        # Augmented copy for training, plain copies for testing. We also keep
        # a non-augmented view of the training set: later programs need it to
        # compute features/signatures of training images without random crops.
        self.train_set = self._load(train=True, augment=True)
        self.train_set_plain = self._load(train=True, augment=False)
        self.test_set = self._load(train=False, augment=False)

        # Original class names, listed in our new label order.
        self.class_names = [self.train_set.classes[c] for c in self.class_order]

        # For each new label, the indices of its images. Built once, reused by every loader.
        self._train_idx = self._index_by_label(self.train_set.targets)
        self._test_idx = self._index_by_label(self.test_set.targets)

    def _load(self, train: bool, augment: bool):
        ds = datasets.CIFAR100(DATA_DIR, train=train, download=True,
                               transform=_transforms(CIFAR100_MEAN, CIFAR100_STD, train=augment))
        # Rename labels: original id -> its position in class_order.
        old_to_new = np.argsort(self.class_order)
        ds.targets = [int(old_to_new[t]) for t in ds.targets]
        return ds

    @staticmethod
    def _index_by_label(targets) -> dict[int, list[int]]:
        targets = np.asarray(targets)
        return {c: np.where(targets == c)[0].tolist() for c in range(100)}

    def _indices(self, labels, train: bool) -> list[int]:
        table = self._train_idx if train else self._test_idx
        return [i for c in labels for i in table[c]]

    def _dataset(self, train: bool, augment: bool):
        if not train:
            return self.test_set
        return self.train_set if augment else self.train_set_plain

    def get_task_subsets(self, task_id: int):
        """(train_subset, test_subset) for one task, as torch Subsets."""
        labels = task_classes(task_id)
        return (Subset(self.train_set, self._indices(labels, train=True)),
                Subset(self.test_set, self._indices(labels, train=False)))

    def get_task_loaders(self, task_id: int):
        """(train_loader, test_loader) for one task. Training data is shuffled and augmented."""
        train_sub, test_sub = self.get_task_subsets(task_id)
        return (_loader(train_sub, self.batch_size, True, self.num_workers),
                _loader(test_sub, self.batch_size, False, self.num_workers))

    def get_task_loader_with_index(self, task_id: int):
        """Training loader for one task whose batches are (images, labels, dataset indices).
        Shuffled and augmented, like get_task_loaders. Used by replay buffers."""
        subset = IndexedSubset(self.train_set, self._indices(task_classes(task_id), train=True))
        return _loader(subset, self.batch_size, True, self.num_workers)

    def get_class_loader(self, label: int, train: bool = False, augment: bool = False, shuffle: bool = False):
        """Loader for the images of one single class (needed for class-level unlearning)."""
        subset = Subset(self._dataset(train, augment), self._indices([label], train))
        return _loader(subset, self.batch_size, shuffle, self.num_workers)

    def get_seen_test_loader(self, up_to_task: int):
        """Test loader over every task 0..up_to_task (class-incremental evaluation)."""
        labels = [c for t in range(up_to_task + 1) for c in task_classes(t)]
        subset = Subset(self.test_set, self._indices(labels, train=False))
        return _loader(subset, self.batch_size, False, self.num_workers)
