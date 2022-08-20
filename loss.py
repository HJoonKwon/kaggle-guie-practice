import torch.nn as nn


def cross_entropy_loss(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)
