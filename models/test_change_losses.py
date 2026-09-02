"""Deterministic checks for PHENet change-detection losses."""

import torch
import torch.nn.functional as F

from utils.loss import (
    BinaryBCEDiceLoss,
    DiceCrossEntropyLoss,
)


def previous_current_formula(
    logits,
    target,
    smooth=1e-6,
):
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
    ).permute(
        0,
        3,
        1,
        2,
    ).float()

    dims = (0, 2, 3)

    intersection = (
        probabilities
        * one_hot
    ).sum(dims)

    denominator = (
        probabilities.sum(dims)
        + one_hot.sum(dims)
    )

    dice = 1.0 - (
        (
            2.0 * intersection
            + smooth
        )
        /
        (
            denominator
            + smooth
        )
    ).mean()

    return ce + dice, ce, dice


def main():
    torch.manual_seed(123)

    logits = torch.tensor(
        [
            [
                [
                    [2.0, -1.0],
                    [0.5, 1.2],
                ],
                [
                    [-0.5, 1.5],
                    [1.0, -0.3],
                ],
            ],
            [
                [
                    [1.0, 0.2],
                    [-0.8, 1.3],
                ],
                [
                    [-0.3, 1.4],
                    [1.2, -0.5],
                ],
            ],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    target = torch.tensor(
        [
            [
                [0, 1],
                [1, 0],
            ],
            [
                [0, 1],
                [1, 0],
            ],
        ],
        dtype=torch.long,
    )

    current = DiceCrossEntropyLoss()
    candidate = BinaryBCEDiceLoss()

    (
        current_total,
        current_ce,
        current_dice,
    ) = current(
        logits,
        target,
    )

    (
        expected_total,
        expected_ce,
        expected_dice,
    ) = previous_current_formula(
        logits,
        target,
    )

    torch.testing.assert_close(
        current_total,
        expected_total,
    )

    torch.testing.assert_close(
        current_ce,
        expected_ce,
    )

    torch.testing.assert_close(
        current_dice,
        expected_dice,
    )

    print(
        "[OK] current loss is numerically unchanged"
    )

    (
        candidate_total,
        candidate_bce,
        candidate_dice,
    ) = candidate(
        logits,
        target,
    )

    # For two-class logits, BCE on z1-z0 must match CE.
    reference_ce = F.cross_entropy(
        logits,
        target,
    )

    torch.testing.assert_close(
        candidate_bce,
        reference_ce,
        rtol=1e-6,
        atol=1e-7,
    )

    print(
        "[OK] binary BCE(z1-z0) matches 2-class CE"
    )

    change_logit = (
        logits[:, 1]
        - logits[:, 0]
    )

    probability = torch.sigmoid(
        change_logit
    )

    target_float = target.float()

    smooth = 1e-6

    intersection = (
        probability
        * target_float
    ).sum()

    denominator = (
        probability.sum()
        + target_float.sum()
    )

    expected_fg_dice = 1.0 - (
        (
            2.0 * intersection
            + smooth
        )
        /
        (
            denominator
            + smooth
        )
    )

    torch.testing.assert_close(
        candidate_dice,
        expected_fg_dice,
    )

    print(
        "[OK] candidate Dice uses foreground/change class only"
    )

    if not torch.isfinite(
        candidate_total
    ):
        raise AssertionError(
            "Candidate total loss is not finite"
        )

    candidate_total.backward()

    if logits.grad is None:
        raise AssertionError(
            "No gradient reached logits"
        )

    if not torch.isfinite(
        logits.grad
    ).all():
        raise AssertionError(
            "Non-finite gradient detected"
        )

    if float(
        logits.grad.abs().sum()
    ) <= 0.0:
        raise AssertionError(
            "Gradient is identically zero"
        )

    print(
        "[OK] candidate loss backward is finite and non-zero"
    )

    print()
    print(
        "CHANGE LOSS TESTS PASSED"
    )


if __name__ == "__main__":
    main()