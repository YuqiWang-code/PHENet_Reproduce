"""Compare PGARM pseudo-label variants on change GT.

Variants:
1. frozen_random:
   Current PHENet_Reproduce behavior: one randomly initialized
   ShallowCNN is frozen for the whole run.

2. rgb_l2:
   Direct RGB temporal L2 difference followed by min-max + Otsu.

3. release_random:
   Original PHENet release behavior: create a fresh randomly
   initialized ShallowCNN for every batch.

The script reports metrics for:
- all test samples
- samples listed in test/A/fog.txt
- samples not listed in test/A/fog.txt

No training and no dataset modification are performed.
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from diagnose_pseudo_labels import (
    build_training_equivalent_extractor,
    collect_ids,
    read_binary_label,
    read_rgb,
    resolve_label_dir,
)
from modeling.PHENet import PHENet
from train import (
    Discriminator,
    ShallowCNN,
    generate_pseudo_label,
)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_release_rng(seed):
    """Advance RNG to approximately the point where the official
    release generates its first random ShallowCNN.

    The official release constructs PHENet and the discriminator
    before generate_pseudo_label() creates ShallowCNN.
    """
    seed_everything(seed)

    model = PHENet(
        num_classes=2,
        backbone="mobilenet",
        output_stride=16,
        sync_bn=True,
        freeze_bn=False,
    )

    discriminator = Discriminator()

    del model
    del discriminator


def difference_to_pseudo(difference, output_size):
    masks = []

    for sample in difference:
        array = sample.detach().float().cpu().numpy()

        minimum = float(array.min())
        maximum = float(array.max())
        span = maximum - minimum

        normalized = np.zeros_like(
            array,
            dtype=np.uint8,
        )

        if span > 0:
            normalized = np.clip(
                np.rint(
                    (array - minimum)
                    / span
                    * 255.0
                ),
                0,
                255,
            ).astype(np.uint8)

        _, mask = cv2.threshold(
            normalized,
            0,
            1,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        masks.append(
            torch.from_numpy(mask).unsqueeze(0)
        )

    pseudo = torch.stack(masks).to(
        difference.device,
        dtype=torch.float32,
    )

    return F.interpolate(
        pseudo,
        size=output_size,
        mode="nearest",
    )


@torch.no_grad()
def generate_rgb_pseudo(image_a, image_b):
    difference = torch.linalg.vector_norm(
        image_a - image_b,
        dim=1,
    )

    return difference_to_pseudo(
        difference,
        image_a.shape[-2:],
    )


@torch.no_grad()
def generate_release_pseudo(image_a, image_b):
    """Original release style: a new random CNN for every call."""
    extractor = ShallowCNN().to(
        image_a.device
    ).eval()

    for parameter in extractor.parameters():
        parameter.requires_grad_(False)

    pseudo = generate_pseudo_label(
        image_a,
        image_b,
        extractor,
    )

    del extractor

    return pseudo


def new_stats():
    return {
        "samples": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
        "pseudo_positive": 0,
        "gt_positive": 0,
        "pixels": 0,
        "empty": 0,
        "full": 0,
    }


def update_stats(stats, prediction, target):
    prediction = prediction.bool()
    target = target.bool()

    stats["samples"] += 1

    stats["tp"] += int(
        (prediction & target).sum()
    )
    stats["fp"] += int(
        (prediction & ~target).sum()
    )
    stats["fn"] += int(
        (~prediction & target).sum()
    )
    stats["tn"] += int(
        (~prediction & ~target).sum()
    )

    positive = int(prediction.sum())
    gt_positive = int(target.sum())
    pixels = prediction.numel()

    stats["pseudo_positive"] += positive
    stats["gt_positive"] += gt_positive
    stats["pixels"] += pixels

    if positive == 0:
        stats["empty"] += 1

    if positive == pixels:
        stats["full"] += 1


def safe_div(a, b):
    return a / b if b else 0.0


def report(title, stats):
    tp = stats["tp"]
    fp = stats["fp"]
    fn = stats["fn"]
    tn = stats["tn"]

    precision = safe_div(
        tp,
        tp + fp,
    )

    recall = safe_div(
        tp,
        tp + fn,
    )

    f1 = safe_div(
        2.0 * precision * recall,
        precision + recall,
    )

    iou = safe_div(
        tp,
        tp + fp + fn,
    )

    oa = safe_div(
        tp + tn,
        tp + fp + fn + tn,
    )

    pseudo_ratio = safe_div(
        stats["pseudo_positive"],
        stats["pixels"],
    )

    gt_ratio = safe_div(
        stats["gt_positive"],
        stats["pixels"],
    )

    print()
    print(f"[{title}]")
    print(
        f"Samples: {stats['samples']}"
    )
    print(
        f"Pseudo Positive Ratio(%): "
        f"{pseudo_ratio * 100:.4f}"
    )
    print(
        f"GT Positive Ratio(%): "
        f"{gt_ratio * 100:.4f}"
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
        f"{stats['empty']}/{stats['samples']}"
    )
    print(
        f"Full Pseudo Masks: "
        f"{stats['full']}/{stats['samples']}"
    )


def load_fog_names(path):
    if path is None:
        return set()

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Fog list does not exist: {path}"
        )

    names = {
        line.strip()
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    }

    return names


def main():
    parser = argparse.ArgumentParser()

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
        "--fog-list",
        default=None,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help=(
            "Use 8 to match formal PHENet training."
        ),
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    args = parser.parse_args()

    root = (
        Path(args.data_root)
        .expanduser()
        .resolve()
    )

    split_root = root / args.split
    a_dir = split_root / "A"
    b_dir = split_root / "B"

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

    if args.fog_list is None:
        candidate = a_dir / "fog.txt"

        fog_list_path = (
            candidate
            if candidate.is_file()
            else None
        )
    else:
        fog_list_path = Path(
            args.fog_list
        )

    fog_names = load_fog_names(
        fog_list_path
    )

    unknown_fog_names = sorted(
        fog_names - set(ids)
    )

    if unknown_fog_names:
        print(
            "Warning: fog list contains names "
            "outside the evaluated IDs: "
            f"{len(unknown_fog_names)}",
            flush=True,
        )

    device = torch.device(
        args.device
    )

    # Current reproduction's frozen-random extractor.
    frozen_extractor = (
        build_training_equivalent_extractor(
            args.seed
        )
        .to(device)
        .eval()
    )

    # Reset RNG independently for the release-style branch.
    prepare_release_rng(
        args.seed
    )

    variants = (
        "frozen_random",
        "rgb_l2",
        "release_random",
    )

    subsets = (
        "all",
        "fog",
        "nonfog",
    )

    stats = {
        variant: {
            subset: new_stats()
            for subset in subsets
        }
        for variant in variants
    }

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
        f"Fog-list samples: "
        f"{len(set(ids) & fog_names)}",
        flush=True,
    )
    print(
        f"Device: {device}",
        flush=True,
    )
    print(
        f"Batch size: {args.batch_size}",
        flush=True,
    )
    print(
        f"Seed: {args.seed}",
        flush=True,
    )

    processed = 0

    for start in range(
        0,
        len(ids),
        args.batch_size,
    ):
        group = ids[
            start:start + args.batch_size
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
            frozen = generate_pseudo_label(
                image_a,
                image_b,
                frozen_extractor,
            )

            rgb = generate_rgb_pseudo(
                image_a,
                image_b,
            )

            release = generate_release_pseudo(
                image_a,
                image_b,
            )

        predictions = {
            "frozen_random": frozen,
            "rgb_l2": rgb,
            "release_random": release,
        }

        for variant, pseudo in predictions.items():
            pseudo = (
                pseudo[:, 0]
                .detach()
                .cpu()
                .to(torch.uint8)
            )

            if pseudo.shape != targets.shape:
                raise RuntimeError(
                    f"{variant} shape mismatch: "
                    f"{tuple(pseudo.shape)} vs "
                    f"{tuple(targets.shape)}"
                )

            for index, name in enumerate(group):
                prediction = pseudo[index]
                target = targets[index]

                update_stats(
                    stats[variant]["all"],
                    prediction,
                    target,
                )

                subset = (
                    "fog"
                    if name in fog_names
                    else "nonfog"
                )

                update_stats(
                    stats[variant][subset],
                    prediction,
                    target,
                )

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

    for variant in variants:
        print()
        print("=" * 72)
        print(
            f"VARIANT: {variant}"
        )
        print("=" * 72)

        report(
            f"{variant} / ALL",
            stats[variant]["all"],
        )

        report(
            f"{variant} / FOG",
            stats[variant]["fog"],
        )

        report(
            f"{variant} / NONFOG",
            stats[variant]["nonfog"],
        )


if __name__ == "__main__":
    main()