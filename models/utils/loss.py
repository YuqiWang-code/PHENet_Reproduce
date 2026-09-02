import torch
import torch.nn as nn
import torch.nn.functional as F


def _validate_inputs(logits, target):
    if logits.ndim != 4:
        raise ValueError(
            f"logits must have shape [B,C,H,W], got {tuple(logits.shape)}"
        )

    if target.ndim != 3:
        raise ValueError(
            f"target must have shape [B,H,W], got {tuple(target.shape)}"
        )

    if logits.shape[0] != target.shape[0]:
        raise ValueError(
            f"Batch mismatch: logits={tuple(logits.shape)}, "
            f"target={tuple(target.shape)}"
        )

    if logits.shape[-2:] != target.shape[-2:]:
        raise ValueError(
            f"Spatial mismatch: logits={tuple(logits.shape)}, "
            f"target={tuple(target.shape)}"
        )


class DiceCrossEntropyLoss(nn.Module):
    """Current reproduction loss: 2-class CE + mean foreground/background Dice."""

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        _validate_inputs(logits, target)

        target = target.long()

        ce = F.cross_entropy(
            logits,
            target,
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        one_hot = F.one_hot(
            target,
            num_classes=logits.shape[1],
        ).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)

        intersection = (
            probabilities * one_hot
        ).sum(dims)

        denominator = (
            probabilities.sum(dims)
            + one_hot.sum(dims)
        )

        dice = 1.0 - (
            (
                2.0 * intersection
                + self.smooth
            )
            /
            (
                denominator
                + self.smooth
            )
        ).mean()

        return ce + dice, ce, dice


class BinaryBCEDiceLoss(nn.Module):
    """Binary BCEWithLogits + foreground/change-class Dice.

    PHENet keeps its existing two-channel output [background, change].
    The binary change logit is therefore represented as:

        change_logit = logits[:, 1] - logits[:, 0]

    This preserves the existing decoder and argmax inference behavior.
    """

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        _validate_inputs(logits, target)

        if logits.shape[1] != 2:
            raise ValueError(
                "BinaryBCEDiceLoss requires exactly two output channels, "
                f"got {logits.shape[1]}"
            )

        target_float = target.float()

        # Binary log-odds corresponding to the current two-class logits.
        change_logit = (
            logits[:, 1]
            - logits[:, 0]
        )

        bce = F.binary_cross_entropy_with_logits(
            change_logit,
            target_float,
        )

        change_probability = torch.sigmoid(
            change_logit
        )

        # Foreground/change-class Dice only.
        intersection = (
            change_probability
            * target_float
        ).sum()

        denominator = (
            change_probability.sum()
            + target_float.sum()
        )

        dice = 1.0 - (
            (
                2.0 * intersection
                + self.smooth
            )
            /
            (
                denominator
                + self.smooth
            )
        )

        return bce + dice, bce, dice


def build_change_loss(mode):
    if mode == "current":
        return DiceCrossEntropyLoss(), "CE"

    if mode == "bce_fg_dice":
        return BinaryBCEDiceLoss(), "BCE"

    raise ValueError(
        f"Unsupported change loss mode: {mode}"
    )