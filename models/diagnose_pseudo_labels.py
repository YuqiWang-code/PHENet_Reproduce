"""Diagnose the PGARM pseudo-label quality against binary change GT.

This script does not train PHENet and does not modify the dataset.
It reproduces the current frozen-random ShallowCNN initialization order
used by models/train.py and evaluates pseudo masks on a deterministic split.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF

from modeling.PHENet import PHENet
from train import Discriminator, ShallowCNN, generate_pseudo_label


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def read_rgb(path):
    with Image.open(path) as image:
        image = image.convert("RGB")

    tensor = TF.to_tensor(image)
    tensor = TF.normalize(
        tensor,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )
    return tensor


def read_binary_label(path):
    with Image.open(path) as image:
        array = np.asarray(image)

    if array.ndim == 3:
        array = array[..., 0]

    return torch.from_numpy(
        (array > 0).astype(np.uint8)
    )


def collect_ids(a_dir, b_dir, label_dir):
    ids = sorted(
        path.name
        for path in a_dir.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_SUFFIXES
        )
    )

    if not ids:
        raise RuntimeError(
            f"No supported images found in {a_dir}"
        )

    missing = []

    for name in ids:
        for root in (b_dir, label_dir):
            if not (root / name).is_file():
                missing.append((name, root))

    if missing:
        preview = ", ".join(
            f"{name} @ {root}"
            for name, root in missing[:10]
        )
        raise FileNotFoundError(
            f"{len(missing)} paired files are missing: "
            f"{preview}"
        )

    return ids


def resolve_label_dir(split_root, configured):
    if configured != "auto":
        path = split_root / configured
        if not path.is_dir():
            raise FileNotFoundError(
                f"Label directory does not exist: {path}"
            )
        return path

    for name in ("label", "GT", "gt", "labels"):
        path = split_root / name
        if path.is_dir():
            return path

    raise FileNotFoundError(
        f"No label directory found under {split_root}"
    )


def build_training_equivalent_extractor(seed):
    """Reproduce the RNG initialization order in Trainer.__init__.

    train.py creates:
      1. PHENet
      2. Discriminator
      3. ShallowCNN

    Since ShallowCNN is random and frozen, its actual weights depend on
    all RNG consumption before its construction. Reproduce that order.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # PHENet is instantiated first in Trainer.__init__.
    model = PHENet(
        num_classes=2,
        backbone="mobilenet",
        output_stride=16,
        sync_bn=True,
        freeze_bn=False,
    )

    # Discriminator is instantiated before ShallowCNN.
    discriminator = Discriminator()

    extractor = ShallowCNN().eval()

    for parameter in extractor.parameters():
        parameter.requires_grad_(False)

    # These objects are needed only to advance RNG exactly as training does.
    del model
    del discriminator

    return extractor


def update_confusion(prediction, target):
    prediction = prediction.bool()
    target = target.bool()

    tp = int((prediction & target).sum())
    fp = int((prediction & ~target).sum())
    fn = int((~prediction & target).sum())
    tn = int((~prediction & ~target).sum())

    return tp, fp, fn, tn


def safe_div(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate current PGARM pseudo-labels "
            "against binary change labels."
        )
    )

    parser.add_argument(
        "--data-root",
        required=True,
    )

    parser.add_argument(
        "--split",
        default="test",
    )

    parser.add_argument(
        "--label-dir",
        default="auto",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help=(
            "0 means evaluate the complete split."
        ),
    )

    parser.add_argument(
        "--device",
        default="cpu",
        help=(
            "Use cpu by default. "
            "cuda:0 is also supported."
        ),
    )

    args = parser.parse_args()

    root = Path(args.data_root).expanduser().resolve()
    split_root = root / args.split

    a_dir = split_root / "A"
    b_dir = split_root / "B"

    if not a_dir.is_dir():
        raise FileNotFoundError(a_dir)

    if not b_dir.is_dir():
        raise FileNotFoundError(b_dir)

    label_dir = resolve_label_dir(
        split_root,
        args.label_dir,
    )

    ids = collect_ids(
        a_dir,
        b_dir,
        label_dir,
    )

    if args.max_samples > 0:
        ids = ids[: args.max_samples]

    device = torch.device(args.device)

    extractor = build_training_equivalent_extractor(
        args.seed
    ).to(device)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    pseudo_positive_pixels = 0
    gt_positive_pixels = 0
    total_pixels = 0

    empty_pseudo = 0
    full_pseudo = 0

    processed = 0

    print(
        f"Dataset: {root}",
        flush=True,
    )
    print(
        f"Split: {args.split}",
        flush=True,
    )
    print(
        f"Samples: {len(ids)}",
        flush=True,
    )
    print(
        f"Device: {device}",
        flush=True,
    )
    print(
        f"Seed: {args.seed}",
        flush=True,
    )

    for start in range(
        0,
        len(ids),
        args.batch_size,
    ):
        group = ids[
            start : start + args.batch_size
        ]

        image_a = torch.stack(
            [
                read_rgb(a_dir / name)
                for name in group
            ]
        ).to(device)

        image_b = torch.stack(
            [
                read_rgb(b_dir / name)
                for name in group
            ]
        ).to(device)

        targets = torch.stack(
            [
                read_binary_label(
                    label_dir / name
                )
                for name in group
            ]
        )

        with torch.inference_mode():
            pseudo = generate_pseudo_label(
                image_a,
                image_b,
                extractor,
            )

        pseudo = (
            pseudo[:, 0]
            .detach()
            .cpu()
            .to(torch.uint8)
        )

        if pseudo.shape != targets.shape:
            raise RuntimeError(
                f"Shape mismatch: "
                f"pseudo={tuple(pseudo.shape)}, "
                f"target={tuple(targets.shape)}"
            )

        if not torch.all(
            (pseudo == 0) | (pseudo == 1)
        ):
            raise RuntimeError(
                "Pseudo mask is not binary."
            )

        for prediction, target in zip(
            pseudo,
            targets,
        ):
            tp, fp, fn, tn = update_confusion(
                prediction,
                target,
            )

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

            positive = int(prediction.sum())
            target_positive = int(target.sum())
            pixels = prediction.numel()

            pseudo_positive_pixels += positive
            gt_positive_pixels += target_positive
            total_pixels += pixels

            if positive == 0:
                empty_pseudo += 1

            if positive == pixels:
                full_pseudo += 1

        processed += len(group)

        if (
            processed % 256 == 0
            or processed == len(ids)
        ):
            print(
                f"Processed "
                f"{processed}/{len(ids)}",
                flush=True,
            )

    precision = safe_div(
        total_tp,
        total_tp + total_fp,
    )

    recall = safe_div(
        total_tp,
        total_tp + total_fn,
    )

    f1 = safe_div(
        2 * precision * recall,
        precision + recall,
    )

    iou = safe_div(
        total_tp,
        total_tp + total_fp + total_fn,
    )

    oa = safe_div(
        total_tp + total_tn,
        total_tp
        + total_fp
        + total_fn
        + total_tn,
    )

    pseudo_positive_ratio = safe_div(
        pseudo_positive_pixels,
        total_pixels,
    )

    gt_positive_ratio = safe_div(
        gt_positive_pixels,
        total_pixels,
    )

    print()
    print("[Pseudo-label Diagnostics]")
    print(f"Samples: {len(ids)}")
    print(
        f"Confusion: "
        f"TP={total_tp} "
        f"FP={total_fp} "
        f"FN={total_fn} "
        f"TN={total_tn}"
    )
    print(
        f"Pseudo Positive Ratio(%): "
        f"{pseudo_positive_ratio * 100:.4f}"
    )
    print(
        f"GT Positive Ratio(%): "
        f"{gt_positive_ratio * 100:.4f}"
    )
    print(
        f"Precision(%): "
        f"{precision * 100:.4f}"
    )
    print(
        f"Recall(%): "
        f"{recall * 100:.4f}"
    )
    print(
        f"F1(%): "
        f"{f1 * 100:.4f}"
    )
    print(
        f"IoU(%): "
        f"{iou * 100:.4f}"
    )
    print(
        f"OA(%): "
        f"{oa * 100:.4f}"
    )
    print(
        f"Empty Pseudo Masks: "
        f"{empty_pseudo}/{len(ids)}"
    )
    print(
        f"Full Pseudo Masks: "
        f"{full_pseudo}/{len(ids)}"
    )


if __name__ == "__main__":
    main()