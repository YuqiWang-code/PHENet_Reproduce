import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceCrossEntropyLoss(nn.Module):
    """Paper loss: L_change = cross entropy + foreground/background Dice."""

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        target = target.long()
        ce = F.cross_entropy(logits, target)
        probabilities = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(target, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = (probabilities * one_hot).sum(dims)
        denominator = probabilities.sum(dims) + one_hot.sum(dims)
        dice = 1.0 - ((2.0 * intersection + self.smooth) / (denominator + self.smooth)).mean()
        return ce + dice, ce, dice
