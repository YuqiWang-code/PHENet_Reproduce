"""Zero-shot cross-dataset generalization evaluation for PHENet.

This script is intentionally evaluation-only:
- it does not create an optimizer, scheduler, discriminator, or loss;
- it never enters training mode;
- it loads only the PHENet ``state_dict`` from a checkpoint with ``strict=True``;
- it evaluates a complete non-training split with a single accumulated confusion matrix.

Expected split layout::

    <data-root>/<split>/
    ├── A/
    ├── B/
    ├── label/ or GT/
    ├── A_heightmap/
    └── B_heightmap/

Run from the PHENet project root, for example::

    CUDA_VISIBLE_DEVICES=1 /home/yqwang/miniconda3/envs/phenet/bin/python \
        models/evaluate_generalization.py \
        --data-root /storage/BCD-foggy/SYSU-CD-foggy \
        --checkpoint /storage/yqwang/PHENet/saved_models/Run1/LEVIR-CD-foggy/best_F1=0.7093.pth \
        --split test \
        --label-dir GT \
        --batch-size 8 \
        --workers 8 \
        --crop-size 256 \
        --device cuda:0 \
        --output-log /tmp/phenet_sysu_generalization.log
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloaders.datasets.CD_dataset_heightmap import CDDataSet, IMAGE_SUFFIXES
from modeling.PHENet import PHENet
from utils.metrics import Evaluator


class RunLogger:
    """Write the same evaluation messages to stdout and, optionally, a log file."""

    def __init__(self, path: Optional[str]) -> None:
        self.path: Optional[Path] = None
        self.handle = None
        if path:
            self.path = Path(path).expanduser().resolve()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = self.path.open("w", encoding="utf-8", buffering=1)

    def write(self, message: str = "") -> None:
        print(message, flush=True)
        if self.handle is not None:
            self.handle.write(message + "\n")

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained PHENet checkpoint on a complete non-training split "
            "for zero-shot cross-dataset generalization."
        )
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Dataset root containing <split>/A, B, label/GT and heightmap directories.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "Checkpoint file, or a directory containing exactly one best_F1=*.pth checkpoint."
        ),
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Evaluation split. For the formal generalization experiment use 'test'.",
    )
    parser.add_argument(
        "--label-dir",
        default="auto",
        help="Label directory name below the split: auto, label, GT, gt, or labels.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument(
        "--device",
        default="cuda:0",
        help=(
            "Torch device. On the project server, CUDA_VISIBLE_DEVICES=1 maps physical GPU 1 "
            "to logical cuda:0. Use cpu only for local/static smoke tests."
        ),
    )
    parser.add_argument(
        "--output-log",
        default=None,
        help="Optional evaluation log path. Prediction images are never saved.",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Optional display name. Defaults to the data-root directory name.",
    )
    parser.add_argument("--seed", type=int, default=1)

    # Normally these are restored from checkpoint['args']. The options below exist only
    # for an explicitly documented compatibility override if an older checkpoint lacks
    # reliable metadata.
    parser.add_argument(
        "--out-stride",
        type=int,
        choices=(8, 16),
        default=None,
        help="Override checkpoint out_stride. Omit to restore from checkpoint args.",
    )
    parser.add_argument(
        "--sync-bn",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override checkpoint sync_bn. Omit to restore from checkpoint args.",
    )
    parser.add_argument(
        "--freeze-bn",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override checkpoint freeze_bn. Omit to restore from checkpoint args.",
    )
    return parser.parse_args()


def resolve_checkpoint(value: str) -> Path:
    """Resolve a checkpoint file or a directory containing one best_F1 checkpoint."""
    path = Path(value).expanduser().resolve()
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
    if not path.is_dir():
        raise RuntimeError(f"Checkpoint path is neither a file nor a directory: {path}")

    candidates = sorted(path.glob("best_F1=*.pth"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one best_F1=*.pth checkpoint in {path}, "
            f"found {len(candidates)}: {candidates}"
        )
    return candidates[0]


def metadata_mapping(value: Any) -> Dict[str, Any]:
    """Convert checkpoint metadata such as args Namespace/dict to a plain dictionary."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {"value": value}


def json_safe(value: Any) -> Any:
    """Convert common checkpoint metadata values into JSON-serializable objects."""
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def coerce_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"Cannot interpret checkpoint field {field_name}={value!r} as bool")


def resolve_model_config(
    cli_args: argparse.Namespace,
    checkpoint_args: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Restore PHENet construction arguments from checkpoint metadata when possible."""
    config: Dict[str, Any] = {}
    sources: Dict[str, str] = {}

    if cli_args.out_stride is not None:
        config["out_stride"] = int(cli_args.out_stride)
        sources["out_stride"] = "CLI override"
    elif "out_stride" in checkpoint_args:
        config["out_stride"] = int(checkpoint_args["out_stride"])
        sources["out_stride"] = "checkpoint args"
    else:
        config["out_stride"] = 16
        sources["out_stride"] = "train.py default fallback"

    if config["out_stride"] not in (8, 16):
        raise ValueError(f"Unsupported out_stride restored from checkpoint: {config['out_stride']}")

    for field_name, cli_value, fallback in (
        ("sync_bn", cli_args.sync_bn, False),
        ("freeze_bn", cli_args.freeze_bn, False),
    ):
        if cli_value is not None:
            config[field_name] = bool(cli_value)
            sources[field_name] = "CLI override"
        elif field_name in checkpoint_args:
            config[field_name] = coerce_bool(checkpoint_args[field_name], field_name)
            sources[field_name] = "checkpoint args"
        else:
            config[field_name] = fallback
            sources[field_name] = "train.py default fallback"

    return config, sources


def supported_image_names(directory: Path) -> set[str]:
    return {
        path.name
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def resolve_label_directory(split_root: Path, configured: str) -> Path:
    if configured != "auto":
        candidate = split_root / configured
        if not candidate.is_dir():
            raise FileNotFoundError(f"Configured label directory does not exist: {candidate}")
        return candidate

    for name in ("label", "GT", "gt", "labels"):
        candidate = split_root / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"No label directory found below {split_root}; expected label/ or GT/."
    )


def validate_dataset_pairing(
    data_root: Path,
    split: str,
    configured_label_dir: str,
) -> Tuple[Path, int, Dict[str, int]]:
    """Require exact A/B/label/A_heightmap/B_heightmap filename-set equality."""
    split_root = data_root / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"Split directory does not exist: {split_root}")

    label_dir = resolve_label_directory(split_root, configured_label_dir)
    directories = {
        "A": split_root / "A",
        "B": split_root / "B",
        "label": label_dir,
        "A_heightmap": split_root / "A_heightmap",
        "B_heightmap": split_root / "B_heightmap",
    }

    missing_directories = [str(path) for path in directories.values() if not path.is_dir()]
    if missing_directories:
        raise FileNotFoundError(
            "Missing dataset directories: " + ", ".join(missing_directories)
        )

    name_sets = {name: supported_image_names(path) for name, path in directories.items()}
    counts = {name: len(names) for name, names in name_sets.items()}

    if not name_sets["A"]:
        raise RuntimeError(f"No supported images found in {directories['A']}")

    reference = name_sets["A"]
    mismatch_messages = []
    for name in ("B", "label", "A_heightmap", "B_heightmap"):
        names = name_sets[name]
        missing = sorted(reference - names)
        extra = sorted(names - reference)
        if missing or extra:
            mismatch_messages.append(
                f"{name}: missing={len(missing)} {missing[:10]}, "
                f"extra={len(extra)} {extra[:10]}"
            )

    if mismatch_messages:
        raise RuntimeError(
            "Dataset pairing check failed; A/B/label/A_heightmap/B_heightmap must have "
            "exactly matching supported-image filenames.\n" + "\n".join(mismatch_messages)
        )

    return label_dir, len(reference), counts


def build_loader(
    args: argparse.Namespace,
    data_root: Path,
    label_dir_name: str,
    device: torch.device,
) -> Tuple[CDDataSet, DataLoader]:
    dataset_args = SimpleNamespace(
        data_root=str(data_root),
        crop_size=args.crop_size,
        label_dir=label_dir_name,
    )
    dataset = CDDataSet(dataset_args, split=args.split)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.workers,
        "drop_last": False,
        "pin_memory": device.type == "cuda",
    }
    if args.workers > 0:
        loader_kwargs["persistent_workers"] = True

    loader = DataLoader(dataset, **loader_kwargs)
    return dataset, loader


def validate_cli(args: argparse.Namespace) -> None:
    if args.split == "train":
        raise ValueError(
            "evaluate_generalization.py refuses split='train' because the current CDDataSet "
            "enables random training augmentation for that split. Use the formal test split."
        )
    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be > 0, got {args.batch_size}")
    if args.workers < 0:
        raise ValueError(f"--workers must be >= 0, got {args.workers}")
    if args.crop_size <= 0:
        raise ValueError(f"--crop-size must be > 0, got {args.crop_size}")


def configure_reproducibility(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def describe_device(device: torch.device) -> Dict[str, str]:
    details = {
        "device": str(device),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        details["logical_cuda_index"] = str(index)
        details["gpu_name"] = torch.cuda.get_device_name(index)
    return details


def assert_batch_contract(
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    target: torch.Tensor,
    height_a: torch.Tensor,
    height_b: torch.Tensor,
) -> None:
    if image_a.ndim != 4 or image_a.shape[1] != 3:
        raise RuntimeError(f"image_a must be [B,3,H,W], got {tuple(image_a.shape)}")
    if image_b.shape != image_a.shape:
        raise RuntimeError(
            f"image_b shape must equal image_a shape, got {tuple(image_b.shape)} "
            f"vs {tuple(image_a.shape)}"
        )
    if target.ndim != 3:
        raise RuntimeError(f"target must be [B,H,W], got {tuple(target.shape)}")
    if height_a.ndim != 4 or height_a.shape[1] != 1:
        raise RuntimeError(f"height_a must be [B,1,H,W], got {tuple(height_a.shape)}")
    if height_b.shape != height_a.shape:
        raise RuntimeError(
            f"height_b shape must equal height_a shape, got {tuple(height_b.shape)} "
            f"vs {tuple(height_a.shape)}"
        )

    batch_size = image_a.shape[0]
    spatial = image_a.shape[-2:]
    if target.shape[0] != batch_size or target.shape[-2:] != spatial:
        raise RuntimeError(
            f"target batch/spatial shape must match RGB inputs, got {tuple(target.shape)} "
            f"vs RGB {tuple(image_a.shape)}"
        )
    if height_a.shape[0] != batch_size or height_a.shape[-2:] != spatial:
        raise RuntimeError(
            f"height batch/spatial shape must match RGB inputs, got {tuple(height_a.shape)} "
            f"vs RGB {tuple(image_a.shape)}"
        )

    for name, tensor in (
        ("image_a", image_a),
        ("image_b", image_b),
        ("height_a", height_a),
        ("height_b", height_b),
    ):
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"Non-finite values detected in {name}")

    if target.numel() > 0:
        target_min = int(target.min().item())
        target_max = int(target.max().item())
        if target_min < 0 or target_max > 1:
            raise RuntimeError(
                f"Target must be binary 0/1 after loading, got range [{target_min}, {target_max}]"
            )


def load_model(
    checkpoint: Mapping[str, Any],
    model_config: Mapping[str, Any],
    device: torch.device,
) -> PHENet:
    if "state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain required key 'state_dict'. "
            f"Available keys: {list(checkpoint.keys())}"
        )
    state_dict = checkpoint["state_dict"]
    if not isinstance(state_dict, Mapping):
        raise TypeError(
            f"checkpoint['state_dict'] must be a mapping, got {type(state_dict).__name__}"
        )

    # Construct on CPU first, load strictly, then move to the requested device.
    model = PHENet(
        num_classes=2,
        backbone="mobilenet",
        output_stride=int(model_config["out_stride"]),
        sync_bn=bool(model_config["sync_bn"]),
        freeze_bn=bool(model_config["freeze_bn"]),
    )
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def evaluate(
    model: PHENet,
    loader: DataLoader,
    dataset_size: int,
    device: torch.device,
    logger: RunLogger,
    description: str,
) -> Tuple[Dict[str, float], np.ndarray, int, float]:
    evaluator = Evaluator()
    processed_pairs = 0
    first_batch_logged = False
    non_blocking = device.type == "cuda"

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_time = time.perf_counter()

    progress = tqdm(loader, desc=description, dynamic_ncols=True)
    for batch in progress:
        image_a, image_b, target, height_a, height_b, names = batch
        assert_batch_contract(image_a, image_b, target, height_a, height_b)

        if not first_batch_logged:
            logger.write(
                "First Batch Shapes: "
                f"A={tuple(image_a.shape)}, B={tuple(image_b.shape)}, "
                f"Target={tuple(target.shape)}, "
                f"A_heightmap={tuple(height_a.shape)}, B_heightmap={tuple(height_b.shape)}"
            )
            first_batch_logged = True

        image_a = image_a.to(device, non_blocking=non_blocking)
        image_b = image_b.to(device, non_blocking=non_blocking)
        target = target.to(device, non_blocking=non_blocking)
        height_a = height_a.to(device, non_blocking=non_blocking)
        height_b = height_b.to(device, non_blocking=non_blocking)

        logits = model(image_a, image_b, height_a, height_b)[0]
        if logits.ndim != 4 or logits.shape[1] != 2:
            raise RuntimeError(f"PHENet logits must be [B,2,H,W], got {tuple(logits.shape)}")
        if logits.shape[0] != target.shape[0] or logits.shape[-2:] != target.shape[-2:]:
            raise RuntimeError(
                f"Logit/target shape mismatch: logits={tuple(logits.shape)}, "
                f"target={tuple(target.shape)}"
            )
        if not torch.isfinite(logits).all():
            raise RuntimeError("Non-finite values detected in PHENet logits")

        prediction = logits.argmax(dim=1)
        evaluator.add_batch(
            target.detach().cpu().numpy(),
            prediction.detach().cpu().numpy(),
        )

        batch_pairs = int(image_a.shape[0])
        processed_pairs += batch_pairs
        progress.set_postfix(pairs=processed_pairs)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_time

    if processed_pairs != dataset_size:
        raise RuntimeError(
            f"Evaluation did not consume the complete split: processed {processed_pairs}, "
            f"dataset contains {dataset_size}"
        )

    metrics = evaluator.compute()
    confusion = evaluator.confusion_matrix.copy()
    return metrics, confusion, processed_pairs, elapsed


def log_checkpoint_metadata(
    logger: RunLogger,
    checkpoint_path: Path,
    checkpoint: Mapping[str, Any],
    checkpoint_args: Mapping[str, Any],
) -> None:
    logger.write(f"Checkpoint: {checkpoint_path}")
    logger.write(f"Checkpoint Epoch: {checkpoint.get('epoch', 'unknown')}")
    logger.write(f"Checkpoint Best F1 (raw): {checkpoint.get('best_f1', 'unknown')}")
    logger.write(
        "Checkpoint Metrics (raw): "
        + json.dumps(json_safe(checkpoint.get("metrics", {})), ensure_ascii=False, sort_keys=True)
    )
    logger.write(
        "Checkpoint Args: "
        + json.dumps(json_safe(checkpoint_args), ensure_ascii=False, sort_keys=True)
    )


def log_results(
    logger: RunLogger,
    dataset_name: str,
    split: str,
    processed_pairs: int,
    confusion: np.ndarray,
    metrics: Mapping[str, float],
    elapsed: float,
) -> None:
    tn, fp, fn, tp = (int(value) for value in confusion.ravel())

    logger.write()
    logger.write("[Confusion Matrix]")
    logger.write("Rows=GroundTruth, Columns=Prediction")
    logger.write(f"TN: {tn}")
    logger.write(f"FP: {fp}")
    logger.write(f"FN: {fn}")
    logger.write(f"TP: {tp}")
    logger.write(f"Total Pixels: {tn + fp + fn + tp}")

    logger.write()
    logger.write("[Metrics]")
    for key in ("Recall", "Precision", "OA", "F1", "IoU", "Kappa"):
        logger.write(f"{key}(%): {metrics[key] * 100.0:.4f}")

    throughput = processed_pairs / elapsed if elapsed > 0 else 0.0
    logger.write()
    logger.write(f"Processed Image Pairs: {processed_pairs}")
    logger.write(f"Evaluation Wall Time(s): {elapsed:.3f}")
    logger.write(
        "Evaluation Throughput(pairs/s, includes data loading): "
        f"{throughput:.3f}"
    )
    logger.write("Prediction Saving: disabled")

    # One machine-friendly summary row. Metric values are percentages.
    logger.write()
    logger.write(
        "RESULT\tDataset\tSplit\tPairs\tTN\tFP\tFN\tTP\t"
        "Recall\tPrecision\tOA\tF1\tIoU\tKappa"
    )
    logger.write(
        f"RESULT\t{dataset_name}\t{split}\t{processed_pairs}\t{tn}\t{fp}\t{fn}\t{tp}\t"
        f"{metrics['Recall'] * 100.0:.4f}\t"
        f"{metrics['Precision'] * 100.0:.4f}\t"
        f"{metrics['OA'] * 100.0:.4f}\t"
        f"{metrics['F1'] * 100.0:.4f}\t"
        f"{metrics['IoU'] * 100.0:.4f}\t"
        f"{metrics['Kappa'] * 100.0:.4f}"
    )


def main() -> None:
    args = parse_args()
    validate_cli(args)

    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {data_root}")

    checkpoint_path = resolve_checkpoint(args.checkpoint)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device requested ({device}) but torch.cuda.is_available() is False"
            )
        torch.cuda.set_device(device)

    configure_reproducibility(args.seed, device)

    with RunLogger(args.output_log) as logger:
        try:
            logger.write("[PHENet Zero-Shot Generalization Evaluation]")
            logger.write(f"Timestamp: {datetime.now().astimezone().isoformat(timespec='seconds')}")
            logger.write(f"Data Root: {data_root}")
            logger.write(f"Split: {args.split}")
            logger.write(f"Crop Size: {args.crop_size}")
            logger.write(f"Batch Size: {args.batch_size}")
            logger.write(f"Workers: {args.workers}")
            logger.write(f"Seed: {args.seed}")

            device_details = describe_device(device)
            for key, value in device_details.items():
                logger.write(f"{key}: {value}")
            if device.type == "cuda" and os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
                logger.write(
                    "WARNING: Formal PHENet server evaluation must expose only physical GPU 1 "
                    "with CUDA_VISIBLE_DEVICES=1; then use logical --device cuda:0."
                )

            # Strictly validate all five image-name sets before constructing the loader.
            label_dir_path, paired_count, counts = validate_dataset_pairing(
                data_root=data_root,
                split=args.split,
                configured_label_dir=args.label_dir,
            )
            logger.write(f"Resolved Label Directory: {label_dir_path}")
            logger.write(
                "Supported Image Counts: "
                + ", ".join(f"{name}={count}" for name, count in counts.items())
            )
            logger.write(f"Strictly Paired Image Pairs: {paired_count}")

            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if not isinstance(checkpoint, Mapping):
                raise TypeError(
                    f"Checkpoint top-level object must be a mapping, got {type(checkpoint).__name__}"
                )
            checkpoint_args = metadata_mapping(checkpoint.get("args"))
            log_checkpoint_metadata(logger, checkpoint_path, checkpoint, checkpoint_args)

            model_config, config_sources = resolve_model_config(args, checkpoint_args)
            logger.write("Model: PHENet")
            logger.write("Backbone: mobilenet (PHENet current implementation)")
            logger.write("Num Classes: 2")
            for field_name in ("out_stride", "sync_bn", "freeze_bn"):
                logger.write(
                    f"Model Config {field_name}: {model_config[field_name]} "
                    f"({config_sources[field_name]})"
                )

            # CDDataSet accepts the concrete resolved label-directory name.
            dataset, loader = build_loader(
                args=args,
                data_root=data_root,
                label_dir_name=label_dir_path.name,
                device=device,
            )
            if len(dataset) != paired_count:
                raise RuntimeError(
                    f"Pairing precheck found {paired_count} pairs but CDDataSet loaded {len(dataset)}"
                )
            logger.write(f"Dataset Samples: {len(dataset)}")

            model = load_model(checkpoint, model_config, device)
            logger.write("Checkpoint state_dict: strict=True load passed")
            logger.write("Model Mode: eval")
            logger.write("Optimizer/Scheduler/Discriminator: not created")
            logger.write()

            dataset_name = args.dataset_name or data_root.name
            metrics, confusion, processed_pairs, elapsed = evaluate(
                model=model,
                loader=loader,
                dataset_size=len(dataset),
                device=device,
                logger=logger,
                description=f"Evaluate {dataset_name}/{args.split}",
            )
            log_results(
                logger=logger,
                dataset_name=dataset_name,
                split=args.split,
                processed_pairs=processed_pairs,
                confusion=confusion,
                metrics=metrics,
                elapsed=elapsed,
            )
        except Exception as error:
            logger.write()
            logger.write(f"[ERROR] {type(error).__name__}: {error}")
            raise


if __name__ == "__main__":
    main()
