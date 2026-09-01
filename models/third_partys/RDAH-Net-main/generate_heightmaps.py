"""Generate RDAH-Net height maps for every A/B image in BCD-foggy datasets."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from rdah_model import load_rdah_checkpoint


SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

IMAGENET_MEAN = torch.tensor(
    (0.485, 0.456, 0.406),
    dtype=torch.float32,
).view(1, 3, 1, 1)

IMAGENET_STD = torch.tensor(
    (0.229, 0.224, 0.225),
    dtype=torch.float32,
).view(1, 3, 1, 1)


def chunks(values, size):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def load_depth_anything(model_name, device):
    try:
        from transformers import (
            AutoImageProcessor,
            AutoModelForDepthEstimation,
        )
    except ImportError as error:
        raise RuntimeError(
            "Install compatible transformers 4.x and safetensors "
            "before height generation"
        ) from error

    model_path = Path(model_name).expanduser()

    if model_path.is_dir():
        model_path = model_path.resolve()

        required_files = (
            "config.json",
            "preprocessor_config.json",
            "model.safetensors",
        )

        missing = [
            name
            for name in required_files
            if not (model_path / name).is_file()
        ]

        if missing:
            raise FileNotFoundError(
                f"Incomplete local Depth Anything V2 model at "
                f"{model_path}; missing: {', '.join(missing)}"
            )

        print(
            f"Loading Depth Anything V2 locally from: {model_path}",
            flush=True,
        )

        processor = AutoImageProcessor.from_pretrained(
            model_path,
            local_files_only=True,
        )

        model = AutoModelForDepthEstimation.from_pretrained(
            model_path,
            local_files_only=True,
        )

    else:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForDepthEstimation.from_pretrained(model_name)

    model = model.to(device).eval()
    model.requires_grad_(False)

    return processor, model


def image_tensor(images, device):
    arrays = [
        np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        for image in images
    ]

    value = torch.from_numpy(np.stack(arrays)).to(
        device=device,
        dtype=torch.float32,
    )

    mean = IMAGENET_MEAN.to(device=device)
    std = IMAGENET_STD.to(device=device)

    return (value - mean) / std

@torch.no_grad()
def relative_depth(images, processor, model, device, size):
    inputs = processor(
        images=images,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    prediction = model(**inputs).predicted_depth.unsqueeze(1)

    prediction = F.interpolate(
        prediction,
        size=(size, size),
        mode="bicubic",
        align_corners=False,
    )

    flat = prediction.flatten(1)

    minimum = flat.min(dim=1).values.view(
        -1, 1, 1, 1
    )

    maximum = flat.max(dim=1).values.view(
        -1, 1, 1, 1
    )

    prediction = (
        (prediction - minimum)
        / (maximum - minimum).clamp_min(1e-6)
    )

    # RDAH-Net receives the relative depth in [0, 255].
    return prediction * 255.0


def discover_splits(dataset_root):
    """
    This project uses train for training and test for epoch validation.
    SYSU-CD-foggy/val is intentionally not used here.
    """
    splits = []

    for name in ("train", "test"):
        root = dataset_root / name

        if (
            (root / "A").is_dir()
            and (root / "B").is_dir()
        ):
            splits.append(name)

    if not splits:
        raise FileNotFoundError(
            f"No train/test A/B directories found under "
            f"{dataset_root}"
        )

    return splits


def discover_images(input_dir):
    return sorted(
        path
        for path in input_dir.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() in SUFFIXES
        )
    )

def encode_height_uint16(height):
    height = np.asarray(height, dtype=np.float32)

    if not np.isfinite(height).all():
        raise RuntimeError("Non-finite values found in RDAH-Net height output")

    minimum = float(height.min())
    maximum = float(height.max())
    span = maximum - minimum

    if span < 1e-6:
        return np.zeros(height.shape, dtype=np.uint16)

    normalized = (height - minimum) / span

    return np.clip(
        np.rint(normalized * 65535.0),
        0,
        65535,
    ).astype(np.uint16)

@torch.no_grad()
def generate_phase(
    args,
    dataset_root,
    split,
    phase,
    processor,
    depth_model,
    height_model,
    device,
):
    input_dir = dataset_root / split / phase
    output_dir = (
        dataset_root
        / split
        / f"{phase}_heightmap"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_paths = discover_images(input_dir)

    if not all_paths:
        raise RuntimeError(
            f"No supported images found in {input_dir}"
        )

    if args.overwrite:
        paths = all_paths
    else:
        paths = [
            path
            for path in all_paths
            if not (output_dir / path.name).is_file()
        ]

    description = (
        f"{dataset_root.name}/{split}/{phase}"
    )

    print(
        f"{description}: "
        f"{len(all_paths)} source images, "
        f"{len(paths)} pending",
        flush=True,
    )

    total_batches = (
        len(paths) + args.batch_size - 1
    ) // args.batch_size

    for batch_index, group in enumerate(
        tqdm(
            chunks(paths, args.batch_size),
            total=total_batches,
            desc=description,
        )
    ):
        images = []

        for path in group:
            with Image.open(path) as image:
                image = image.convert("RGB")

                if image.size != (
                    args.image_size,
                    args.image_size,
                ):
                    image = image.resize(
                        (
                            args.image_size,
                            args.image_size,
                        ),
                        Image.Resampling.BICUBIC,
                    )

                images.append(image)

        # RGB -> frozen Depth Anything V2 -> relative depth.
        depth = relative_depth(
            images=images,
            processor=processor,
            model=depth_model,
            device=device,
            size=args.image_size,
        )

        if depth.shape != (
            len(images),
            1,
            args.image_size,
            args.image_size,
        ):
            raise RuntimeError(
                f"Unexpected relative depth shape "
                f"for {description}: "
                f"{tuple(depth.shape)}"
            )

        if not torch.isfinite(depth).all():
            raise RuntimeError(
                f"Non-finite Depth Anything V2 output "
                f"detected for {description}"
            )

        rgb = image_tensor(
            images,
            device,
        )

        # Both models are inference-only.
        # AMP remains optional because some CUDA/cuDNN
        # configurations may not support every FP16 conv path.
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(
                args.amp
                and device.type == "cuda"
            ),
        ):
            heights = height_model(
                depth,
                rgb,
            )

        if heights.ndim != 4:
            raise RuntimeError(
                f"Unexpected RDAH-Net output ndim "
                f"for {description}: "
                f"{heights.ndim}"
            )

        if heights.shape != (
            len(images),
            1,
            args.image_size,
            args.image_size,
        ):
            raise RuntimeError(
                f"Unexpected RDAH-Net output shape "
                f"for {description}: "
                f"{tuple(heights.shape)}"
            )

        if not torch.isfinite(heights).all():
            raise RuntimeError(
                f"Non-finite RDAH-Net output "
                f"detected for {description}"
            )

        # Print raw statistics only during smoke tests.
        if args.debug_stats:
            print(
                f"\nRDAH raw output {description} "
                f"batch={batch_index + 1}: "
                f"min={heights.min().item():.6f}, "
                f"max={heights.max().item():.6f}, "
                f"mean={heights.mean().item():.6f}, "
                f"std={heights.std().item():.6f}",
                flush=True,
            )

        heights = (
            heights
            .float()
            .cpu()
            .numpy()[:, 0]
        )

        for path, height in zip(
            group,
            heights,
        ):
            encoded = encode_height_uint16(height)

            destination = (
                output_dir
                / path.name
            )

            if not cv2.imwrite(
                str(destination),
                encoded,
            ):
                raise IOError(
                    f"Failed to write {destination}"
                )

            if not cv2.imwrite(
                str(destination),
                encoded,
            ):
                raise IOError(
                    f"Failed to write "
                    f"{destination}"
                )

    expected_names = {
        path.name
        for path in all_paths
    }

    output_paths = discover_images(
        output_dir
    )

    actual_names = {
        path.name
        for path in output_paths
    }

    missing_names = sorted(
        expected_names - actual_names
    )

    extra_names = sorted(
        actual_names - expected_names
    )

    if missing_names:
        raise RuntimeError(
            f"Height-map filename mismatch "
            f"for {description}: "
            f"{len(missing_names)} missing; "
            f"first entries: "
            f"{missing_names[:10]}"
        )

    if extra_names:
        print(
            f"Warning: {description} has "
            f"{len(extra_names)} extra height-map "
            f"files not present in the source directory; "
            f"first entries: {extra_names[:10]}",
            flush=True,
        )

    actual = len(
        expected_names & actual_names
    )

    expected = len(
        expected_names
    )

    if actual != expected:
        raise RuntimeError(
            f"Height-map count mismatch "
            f"for {description}: "
            f"expected {expected}, "
            f"found {actual}"
        )

    sample_path = (
        output_dir
        / all_paths[0].name
    )

    sample = cv2.imread(
        str(sample_path),
        cv2.IMREAD_UNCHANGED,
    )

    if sample is None:
        raise RuntimeError(
            f"Failed to read generated "
            f"height map: {sample_path}"
        )

    expected_shape = (
        args.image_size,
        args.image_size,
    )

    if sample.shape != expected_shape:
        raise RuntimeError(
            f"Invalid height-map shape at "
            f"{sample_path}: "
            f"{sample.shape}, expected "
            f"{expected_shape}"
        )

    if sample.dtype != np.uint16:
        raise RuntimeError(
            f"Invalid height-map dtype at "
            f"{sample_path}: "
            f"{sample.dtype}, expected uint16"
        )

    if not np.isfinite(sample).all():
        raise RuntimeError(
            f"Non-finite values found in "
            f"{sample_path}"
        )

    print(
        f"Verified {description}: "
        f"{actual} maps, "
        f"shape={sample.shape}, "
        f"dtype={sample.dtype}, "
        f"range=[{sample.min()}, "
        f"{sample.max()}], "
        f"unique={len(np.unique(sample))}",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Depth Anything V2 + RDAH-Net "
            "height-map generation"
        )
    )

    parser.add_argument(
        "--dataset-roots",
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--rdah-checkpoint",
        required=True,
    )

    parser.add_argument(
        "--depth-model",
        default=(
            "depth-anything/"
            "Depth-Anything-V2-Large-hf"
        ),
        help=(
            "Hugging Face model ID or local "
            "Depth Anything V2 directory"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--output-scale",
        type=float,
        default=1.0,
        help=(
            "Scale RDAH checkpoint output "
            "before uint16 PNG encoding"
        ),
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--amp",
        action="store_true",
        help=(
            "Enable FP16 autocast for the "
            "RDAH-Net forward pass"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Regenerate existing height maps"
        ),
    )

    parser.add_argument(
        "--debug-stats",
        action="store_true",
        help=(
            "Print raw RDAH-Net output "
            "statistics for each batch; "
            "use for smoke tests only"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be > 0"
        )

    if args.image_size <= 0:
        raise ValueError(
            "--image-size must be > 0"
        )

    if args.image_size % 128:
        raise ValueError(
            "RDAH-Net image size must be "
            "divisible by 128 "
            "(use the dataset-native 256)"
        )

    if args.output_scale <= 0:
        raise ValueError(
            "--output-scale must be > 0"
        )

    if (
        torch.cuda.is_available()
        and not args.device.startswith("cuda")
    ):
        device = torch.device(
            args.device
        )
    elif torch.cuda.is_available():
        device = torch.device(
            args.device
        )
    else:
        print(
            "CUDA is unavailable; "
            "falling back to CPU.",
            flush=True,
        )
        device = torch.device("cpu")

    print(
        f"Height-map device: {device}",
        flush=True,
    )

    if device.type == "cuda":
        print(
            f"CUDA device: "
            f"{torch.cuda.get_device_name(device)}",
            flush=True,
        )

    processor, depth_model = (
        load_depth_anything(
            args.depth_model,
            device,
        )
    )

    height_model = (
        load_rdah_checkpoint(
            args.rdah_checkpoint,
            device,
        )
    )

    # Height estimation is inference-only.
    height_model.eval()
    height_model.requires_grad_(False)

    for root_value in args.dataset_roots:
        dataset_root = (
            Path(root_value)
            .expanduser()
            .resolve()
        )

        if not dataset_root.is_dir():
            raise FileNotFoundError(
                f"Dataset root does not exist: "
                f"{dataset_root}"
            )

        for split in discover_splits(
            dataset_root
        ):
            for phase in ("A", "B"):
                generate_phase(
                    args=args,
                    dataset_root=dataset_root,
                    split=split,
                    phase=phase,
                    processor=processor,
                    depth_model=depth_model,
                    height_model=height_model,
                    device=device,
                )


if __name__ == "__main__":
    main()