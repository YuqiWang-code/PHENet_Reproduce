"""Diagnose whether the trained PGARM discriminator actually uses pseudo labels.

The trained discriminator is evaluated with:
1. current pseudo labels
2. all-zero pseudo labels
3. batch-shuffled pseudo labels

If discriminator outputs barely change, the pseudo-label channel is effectively ignored.
"""

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloaders.datasets.CD_dataset_heightmap import CDDataSet
from modeling.PHENet import PHENet
from train import Discriminator, generate_pseudo_label
from diagnose_pseudo_labels import build_training_equivalent_extractor


def resolve_checkpoint(value):
    path = Path(value).expanduser().resolve()

    if path.is_file():
        return path

    if not path.is_dir():
        raise FileNotFoundError(
            f"Checkpoint path does not exist: {path}"
        )

    candidates = sorted(
        path.glob("best_F1=*.pth")
    )

    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one best_F1 checkpoint in {path}, "
            f"found {len(candidates)}: {candidates}"
        )

    return candidates[0]


def set_batchnorm_eval(module):
    for child in module.modules():
        if isinstance(
            child,
            nn.modules.batchnorm._BatchNorm,
        ):
            child.eval()


def condition_metrics(
    discriminator,
    image_a,
    image_b,
    fog_a,
    fog_b,
    pseudo,
):
    real_logits = discriminator(
        torch.cat(
            (image_a, image_b, pseudo),
            dim=1,
        )
    )

    fake_logits = discriminator(
        torch.cat(
            (fog_a, fog_b, pseudo),
            dim=1,
        )
    )

    disc_loss = 0.5 * (
        F.binary_cross_entropy_with_logits(
            real_logits,
            torch.ones_like(real_logits),
        )
        +
        F.binary_cross_entropy_with_logits(
            fake_logits,
            torch.zeros_like(fake_logits),
        )
    )

    adv_g = F.binary_cross_entropy_with_logits(
        fake_logits,
        torch.ones_like(fake_logits),
    )

    return {
        "real_logits": real_logits,
        "fake_logits": fake_logits,
        "real_prob": torch.sigmoid(real_logits).mean(),
        "fake_prob": torch.sigmoid(fake_logits).mean(),
        "disc_loss": disc_loss,
        "adv_g": adv_g,
    }


def new_totals():
    return {
        "count": 0,
        "real_prob": 0.0,
        "fake_prob": 0.0,
        "disc_loss": 0.0,
        "adv_g": 0.0,
    }


def update_totals(totals, result, batch_size):
    totals["count"] += batch_size

    for key in (
        "real_prob",
        "fake_prob",
        "disc_loss",
        "adv_g",
    ):
        totals[key] += (
            float(result[key].detach())
            * batch_size
        )


def report_condition(name, totals):
    count = totals["count"]

    print()
    print(f"[Condition: {name}]")

    for key in (
        "real_prob",
        "fake_prob",
        "disc_loss",
        "adv_g",
    ):
        value = totals[key] / count
        print(f"{key}: {value:.6f}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "Checkpoint file or directory containing "
            "exactly one best_F1=*.pth."
        ),
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
        default=8,
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    args = parser.parse_args()

    checkpoint_path = resolve_checkpoint(
        args.checkpoint
    )

    device = torch.device(
        args.device
    )

    dataset_args = SimpleNamespace(
        data_root=args.data_root,
        crop_size=256,
        label_dir=args.label_dir,
    )

    dataset = CDDataSet(
        dataset_args,
        split=args.split,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
    )

    # Reproduce the frozen-random extractor used by v2.
    pseudo_extractor = (
        build_training_equivalent_extractor(
            args.seed
        )
        .to(device)
        .eval()
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    model = PHENet(
        num_classes=2,
        backbone="mobilenet",
        output_stride=16,
        sync_bn=True,
        freeze_bn=False,
    ).to(device)

    model.load_state_dict(
        checkpoint["state_dict"],
        strict=True,
    )

    discriminator = (
        Discriminator()
        .to(device)
        .eval()
    )

    discriminator.load_state_dict(
        checkpoint["discriminator_state_dict"],
        strict=True,
    )

    # PHENet only returns PGARM fog outputs while self.training=True.
    # Keep the top-level model in train mode but freeze BN behavior.
    model.train()
    set_batchnorm_eval(model)

    conditions = {
        "current": new_totals(),
        "zero": new_totals(),
        "shuffled": new_totals(),
    }

    delta_totals = {
        "zero_real": 0.0,
        "zero_fake": 0.0,
        "shuffled_real": 0.0,
        "shuffled_fake": 0.0,
        "count": 0,
    }

    processed = 0

    print(
        f"Checkpoint: {checkpoint_path}",
        flush=True,
    )
    print(
        f"Checkpoint epoch: "
        f"{checkpoint.get('epoch', 'unknown')}",
        flush=True,
    )
    print(
        f"Checkpoint best_f1: "
        f"{checkpoint.get('best_f1', 'unknown')}",
        flush=True,
    )
    print(
        f"Dataset samples: {len(dataset)}",
        flush=True,
    )
    print(
        f"Max samples: {args.max_samples}",
        flush=True,
    )
    print(
        f"Device: {device}",
        flush=True,
    )

    with torch.inference_mode():
        for batch in loader:
            (
                image_a,
                image_b,
                _,
                height_a,
                height_b,
                _,
            ) = batch

            remaining = (
                args.max_samples - processed
                if args.max_samples > 0
                else image_a.shape[0]
            )

            if remaining <= 0:
                break

            take = min(
                image_a.shape[0],
                remaining,
            )

            image_a = image_a[:take].to(device)
            image_b = image_b[:take].to(device)
            height_a = height_a[:take].to(device)
            height_b = height_b[:take].to(device)

            outputs = model(
                image_a,
                image_b,
                height_a,
                height_b,
            )

            (
                _,
                fog_a,
                _,
                _,
                _,
                fog_b,
                _,
                _,
                _,
            ) = outputs

            pseudo = generate_pseudo_label(
                image_a,
                image_b,
                pseudo_extractor,
            )

            zero = torch.zeros_like(
                pseudo
            )

            if take > 1:
                shuffled = pseudo.roll(
                    shifts=1,
                    dims=0,
                )
            else:
                shuffled = pseudo.clone()

            results = {
                "current": condition_metrics(
                    discriminator,
                    image_a,
                    image_b,
                    fog_a,
                    fog_b,
                    pseudo,
                ),
                "zero": condition_metrics(
                    discriminator,
                    image_a,
                    image_b,
                    fog_a,
                    fog_b,
                    zero,
                ),
                "shuffled": condition_metrics(
                    discriminator,
                    image_a,
                    image_b,
                    fog_a,
                    fog_b,
                    shuffled,
                ),
            }

            for name, result in results.items():
                update_totals(
                    conditions[name],
                    result,
                    take,
                )

            current = results["current"]

            delta_totals["zero_real"] += (
                float(
                    (
                        results["zero"]["real_logits"]
                        - current["real_logits"]
                    )
                    .abs()
                    .mean()
                )
                * take
            )

            delta_totals["zero_fake"] += (
                float(
                    (
                        results["zero"]["fake_logits"]
                        - current["fake_logits"]
                    )
                    .abs()
                    .mean()
                )
                * take
            )

            delta_totals["shuffled_real"] += (
                float(
                    (
                        results["shuffled"]["real_logits"]
                        - current["real_logits"]
                    )
                    .abs()
                    .mean()
                )
                * take
            )

            delta_totals["shuffled_fake"] += (
                float(
                    (
                        results["shuffled"]["fake_logits"]
                        - current["fake_logits"]
                    )
                    .abs()
                    .mean()
                )
                * take
            )

            delta_totals["count"] += take

            processed += take

            if (
                processed % 64 == 0
                or (
                    args.max_samples > 0
                    and processed >= args.max_samples
                )
            ):
                print(
                    f"Processed {processed}",
                    flush=True,
                )

            if (
                args.max_samples > 0
                and processed >= args.max_samples
            ):
                break

    print()
    print("=" * 72)
    print("DISCRIMINATOR CONDITIONING DIAGNOSTICS")
    print("=" * 72)

    for name in (
        "current",
        "zero",
        "shuffled",
    ):
        report_condition(
            name,
            conditions[name],
        )

    count = delta_totals["count"]

    print()
    print("[Mean Absolute Logit Change vs Current]")
    print(
        "zero / real: "
        f"{delta_totals['zero_real'] / count:.6f}"
    )
    print(
        "zero / fake: "
        f"{delta_totals['zero_fake'] / count:.6f}"
    )
    print(
        "shuffled / real: "
        f"{delta_totals['shuffled_real'] / count:.6f}"
    )
    print(
        "shuffled / fake: "
        f"{delta_totals['shuffled_fake'] / count:.6f}"
    )

    first_conv = discriminator.network[0]

    weight = (
        first_conv.weight
        .detach()
        .float()
        .cpu()
    )

    channel_rms = (
        weight.square()
        .mean(dim=(0, 2, 3))
        .sqrt()
    )

    rgb_rms = channel_rms[:6].mean()
    pseudo_rms = channel_rms[6]

    print()
    print("[First Conv Input-channel Weight RMS]")

    for index, value in enumerate(
        channel_rms.tolist()
    ):
        label = (
            "pseudo"
            if index == 6
            else f"rgb_{index}"
        )

        print(
            f"{label}: {value:.6f}"
        )

    print(
        "Pseudo / mean RGB RMS ratio: "
        f"{float(pseudo_rms / rgb_rms):.6f}"
    )


if __name__ == "__main__":
    main()