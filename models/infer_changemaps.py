"""Save binary PHENet change maps for a complete test split."""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from evaluate_generalization import (
    assert_batch_contract,
    build_loader,
    configure_reproducibility,
    load_model,
    metadata_mapping,
    resolve_checkpoint,
    resolve_model_config,
    validate_dataset_pairing,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PHENet on a complete test split and save binary change-map PNG files."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--label-dir", default="auto")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--out-stride", type=int, choices=(8, 16), default=None)
    parser.add_argument("--sync-bn", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--freeze-bn", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def validate_cli(args: argparse.Namespace) -> None:
    if args.split == "train":
        raise ValueError(
            "infer_changemaps.py refuses split='train' because CDDataSet applies random "
            "training augmentation. Use the formal test split."
        )
    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be > 0, got {args.batch_size}")
    if args.workers < 0:
        raise ValueError(f"--workers must be >= 0, got {args.workers}")
    if args.crop_size <= 0:
        raise ValueError(f"--crop-size must be > 0, got {args.crop_size}")


def existing_png_names(output_dir: Path) -> set[str]:
    if not output_dir.is_dir():
        return set()
    return {
        path.name
        for path in output_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    }


def prepare_output_dir(
    output_dir: Path,
    expected_names: set[str],
    overwrite: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = existing_png_names(output_dir)

    if not existing:
        return

    if not overwrite:
        raise FileExistsError(
            f"{output_dir} already contains {len(existing)} PNG files. "
            "Use --overwrite only for generated PHENet predictions."
        )

    extra = sorted(existing - expected_names)
    if extra:
        raise RuntimeError(
            f"{output_dir} contains unexpected PNG files: {extra[:10]}. "
            "Clean only the generated result_PHENet directory before re-running."
        )


def save_binary_mask(mask01: np.ndarray, destination: Path) -> None:
    if mask01.ndim != 2:
        raise RuntimeError(f"Prediction must be 2-D, got {mask01.shape}")
    unique = np.unique(mask01)
    if not np.isin(unique, (0, 1)).all():
        raise RuntimeError(f"Prediction is not binary 0/1: {unique.tolist()}")
    encoded = mask01.astype(np.uint8) * 255
    Image.fromarray(encoded, mode="L").save(destination, format="PNG")


@torch.inference_mode()
def run_inference(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    output_dir: Path,
    expected_pairs: int,
) -> tuple[int, float]:
    model.eval()
    processed = 0
    non_blocking = device.type == "cuda"

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()

    progress = tqdm(loader, desc="Save PHENet change maps", dynamic_ncols=True)
    for batch in progress:
        image_a, image_b, target, height_a, height_b, names = batch
        assert_batch_contract(image_a, image_b, target, height_a, height_b)

        image_a = image_a.to(device, non_blocking=non_blocking)
        image_b = image_b.to(device, non_blocking=non_blocking)
        height_a = height_a.to(device, non_blocking=non_blocking)
        height_b = height_b.to(device, non_blocking=non_blocking)

        logits = model(image_a, image_b, height_a, height_b)[0]
        if logits.ndim != 4 or logits.shape[1] != 2:
            raise RuntimeError(f"PHENet logits must be [B,2,H,W], got {tuple(logits.shape)}")
        if not torch.isfinite(logits).all():
            raise RuntimeError("Non-finite values detected in logits")

        prediction = logits.argmax(dim=1).detach().cpu().numpy().astype(np.uint8)

        if len(names) != prediction.shape[0]:
            raise RuntimeError("Batch filename/prediction count mismatch")

        for mask, name in zip(prediction, names):
            name = Path(str(name)).name
            if Path(name).suffix.lower() != ".png":
                raise RuntimeError(f"Expected PNG filename, got: {name}")
            save_binary_mask(mask, output_dir / name)

        processed += prediction.shape[0]
        progress.set_postfix(saved=processed)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    if processed != expected_pairs:
        raise RuntimeError(
            f"Inference did not consume complete split: processed={processed}, "
            f"expected={expected_pairs}"
        )
    return processed, elapsed


def verify_outputs(
    output_dir: Path,
    expected_names: set[str],
    expected_shape: tuple[int, int],
) -> None:
    actual_names = existing_png_names(output_dir)
    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - expected_names)
    if missing or extra:
        raise RuntimeError(
            f"Output filename verification failed: missing={len(missing)} {missing[:10]}, "
            f"extra={len(extra)} {extra[:10]}"
        )

    all_values = set()
    for name in sorted(expected_names):
        with Image.open(output_dir / name) as image:
            array = np.asarray(image)
        if array.shape != expected_shape:
            raise RuntimeError(
                f"Bad prediction shape: {output_dir / name}: {array.shape}"
            )
        if array.dtype != np.uint8:
            raise RuntimeError(
                f"Bad prediction dtype: {output_dir / name}: {array.dtype}"
            )
        values = set(int(v) for v in np.unique(array))
        if not values.issubset({0, 255}):
            raise RuntimeError(
                f"Prediction is not binary 0/255: {output_dir / name}: {sorted(values)}"
            )
        all_values.update(values)

    print(
        f"[PASS] Verified {len(expected_names)} PNG files: "
        f"shape={expected_shape}, dtype=uint8, values={sorted(all_values)}"
    )


def main() -> None:
    args = parse_args()
    validate_cli(args)

    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {data_root}")

    split_root = (data_root / args.split).resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.parent != split_root:
        raise ValueError(
            f"--output-dir must be a direct child of {split_root}, got {output_dir}"
        )

    checkpoint_path = resolve_checkpoint(args.checkpoint)

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA requested ({device}) but unavailable")
        torch.cuda.set_device(device)

    configure_reproducibility(args.seed, device)

    if device.type == "cuda" and os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError(
            "Formal server inference must use physical GPU 1 via CUDA_VISIBLE_DEVICES=1 "
            "and logical --device cuda:0."
        )

    label_dir_path, paired_count, counts = validate_dataset_pairing(
        data_root=data_root,
        split=args.split,
        configured_label_dir=args.label_dir,
    )

    expected_names = {
        path.name
        for path in (split_root / "A").iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    }
    if len(expected_names) != paired_count:
        raise RuntimeError(
            f"A PNG count ({len(expected_names)}) != paired count ({paired_count})"
        )

    prepare_output_dir(output_dir, expected_names, args.overwrite)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f"Checkpoint top-level object must be a mapping, got {type(checkpoint).__name__}"
        )

    checkpoint_args = metadata_mapping(checkpoint.get("args"))
    model_config, config_sources = resolve_model_config(args, checkpoint_args)

    dataset, loader = build_loader(
        args=args,
        data_root=data_root,
        label_dir_name=label_dir_path.name,
        device=device,
    )
    if len(dataset) != paired_count:
        raise RuntimeError(
            f"Pairing precheck={paired_count}, CDDataSet={len(dataset)}"
        )

    model = load_model(checkpoint, model_config, device)

    dataset_name = args.dataset_name or data_root.name
    print("[PHENet Test Changemap Inference]")
    print(f"Dataset: {dataset_name}")
    print(f"Data Root: {data_root}")
    print(f"Split: {args.split}")
    print(f"Strictly Paired Image Pairs: {paired_count}")
    print(
        "Supported Image Counts: "
        + ", ".join(f"{name}={count}" for name, count in counts.items())
    )
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Checkpoint Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Checkpoint Best F1: {checkpoint.get('best_f1', 'unknown')}")
    print(
        "Model Config: "
        + ", ".join(
            f"{name}={model_config[name]} ({config_sources[name]})"
            for name in ("out_stride", "sync_bn", "freeze_bn")
        )
    )
    print("Checkpoint state_dict: strict=True load passed")
    print("Model Mode: eval")
    print("Prediction Rule: argmax(two-class logits)")
    print("Output Encoding: uint8 PNG, unchanged=0, changed=255")
    print(f"Output Directory: {output_dir}")
    print()

    processed, elapsed = run_inference(
        model=model,
        loader=loader,
        device=device,
        output_dir=output_dir,
        expected_pairs=paired_count,
    )

    verify_outputs(
        output_dir=output_dir,
        expected_names=expected_names,
        expected_shape=(args.crop_size, args.crop_size),
    )

    throughput = processed / elapsed if elapsed > 0 else 0.0
    print()
    print(f"[PASS] Saved Image Pairs: {processed}")
    print(f"Inference Wall Time(s): {elapsed:.3f}")
    print(f"Inference Throughput(pairs/s): {throughput:.3f}")
    print(f"Result Directory: {output_dir}")


if __name__ == "__main__":
    main()
