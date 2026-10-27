"""The network used in the whole project: ResNet-18 adapted to 32x32 images."""

import torch.nn as nn
import torchvision


def resnet18_cifar(num_classes: int = 10) -> nn.Module:
    """ResNet-18 for CIFAR.

    The ImageNet version starts with a 7x7 stride-2 conv plus a max-pool, which
    shrinks a 32x32 image to 8x8 before the first block and throws away detail.
    For small images we use a 3x3 stride-1 conv and remove the max-pool.
    """
    model = torchvision.models.resnet18(num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model
