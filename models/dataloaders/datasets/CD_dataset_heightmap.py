"""Dataset for bitemporal change detection with per-image height maps."""

from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _read_rgb(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def _read_label(path):
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., 0]
    # Both 0/1 and 0/255 masks are accepted.
    return Image.fromarray((array > 0).astype(np.uint8), mode="L")


def _read_height(path):
    with Image.open(path) as image:
        array = np.asarray(image, dtype=np.float32)
    if array.ndim == 3:
        array = array[..., 0]
    if not np.isfinite(array).all():
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    return Image.fromarray(array, mode="F")


class CDDataSet(Dataset):
    """Load A/B, binary label, and A/B height maps from one split."""

    def __init__(self, args, split="train"):
        self.root = Path(args.data_root).expanduser().resolve()
        self.split = split
        self.split_root = self.root / split
        self.image_size = int(args.crop_size)
        self.training = split == "train"

        self.a_dir = self.split_root / "A"
        self.b_dir = self.split_root / "B"
        self.label_dir = self._resolve_label_dir(getattr(args, "label_dir", "auto"))
        self.a_height_dir = self.split_root / "A_heightmap"
        self.b_height_dir = self.split_root / "B_heightmap"
        self._validate_directories()
        self.ids = self._collect_ids()

        self.color_jitter = transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3
        )

    def _resolve_label_dir(self, configured):
        if configured != "auto":
            return self.split_root / configured
        for name in ("label", "GT", "gt", "labels"):
            candidate = self.split_root / name
            if candidate.is_dir():
                return candidate
        raise FileNotFoundError(
            f"No label directory found below {self.split_root}; expected label/ or GT/."
        )

    def _validate_directories(self):
        required = (
            self.a_dir,
            self.b_dir,
            self.label_dir,
            self.a_height_dir,
            self.b_height_dir,
        )
        missing = [str(path) for path in required if not path.is_dir()]
        if missing:
            raise FileNotFoundError("Missing dataset directories: " + ", ".join(missing))

    def _collect_ids(self):
        directories = {
            "A": self.a_dir,
            "B": self.b_dir,
            "label": self.label_dir,
            "A_heightmap": self.a_height_dir,
            "B_heightmap": self.b_height_dir,
        }

        name_sets = {
            name: {
                path.name
                for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            }
            for name, directory in directories.items()
        }

        if not name_sets["A"]:
            raise RuntimeError(f"No supported images found in {self.a_dir}")

        reference = name_sets["A"]

        mismatch_messages = []

        for name in ("B", "label", "A_heightmap", "B_heightmap"):
            current = name_sets[name]

            missing = sorted(reference - current)
            extra = sorted(current - reference)

            if missing or extra:
                mismatch_messages.append(
                    f"{name}: "
                    f"count={len(current)}, "
                    f"missing={len(missing)} {missing[:10]}, "
                    f"extra={len(extra)} {extra[:10]}"
                )

        if mismatch_messages:
            counts = ", ".join(
                f"{name}={len(names)}"
                for name, names in name_sets.items()
            )

            raise RuntimeError(
                "Dataset pairing check failed. "
                "A/B/label/A_heightmap/B_heightmap must contain exactly "
                "the same supported-image filenames.\n"
                f"Counts: {counts}\n"
                + "\n".join(mismatch_messages)
            )

        return sorted(reference)

    @staticmethod
    def _geometry(images, label, heights, image_size, training):
        all_items = images + [label] + heights
        if training and random.random() < 0.5:
            all_items = [TF.hflip(item) for item in all_items]
        if training and random.random() < 0.5:
            all_items = [TF.vflip(item) for item in all_items]
        if training and random.random() < 0.5:
            angle = random.choice((90, 180, 270))
            all_items = [TF.rotate(item, angle) for item in all_items]

        if training:
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                images[0], scale=(0.8, 1.0), ratio=(1.0, 1.0)
            )
            rgb_interp = InterpolationMode.BICUBIC
            images = [
                TF.resized_crop(item, i, j, h, w, (image_size, image_size), rgb_interp)
                for item in all_items[:2]
            ]
            label = TF.resized_crop(
                all_items[2], i, j, h, w, (image_size, image_size), InterpolationMode.NEAREST
            )
            heights = [
                TF.resized_crop(item, i, j, h, w, (image_size, image_size), InterpolationMode.NEAREST)
                for item in all_items[3:]
            ]
        else:
            images = [TF.resize(item, (image_size, image_size), InterpolationMode.BICUBIC) for item in all_items[:2]]
            label = TF.resize(all_items[2], (image_size, image_size), InterpolationMode.NEAREST)
            heights = [
                TF.resize(item, (image_size, image_size), InterpolationMode.NEAREST)
                for item in all_items[3:]
            ]
        return images, label, heights

    def __getitem__(self, index):
        name = self.ids[index]
        images = [_read_rgb(self.a_dir / name), _read_rgb(self.b_dir / name)]
        label = _read_label(self.label_dir / name)
        heights = [_read_height(self.a_height_dir / name), _read_height(self.b_height_dir / name)]
        images, label, heights = self._geometry(
            images, label, heights, self.image_size, self.training
        )

        if self.training and random.random() < 0.5:
            radius = random.random()
            images = [image.filter(ImageFilter.GaussianBlur(radius=radius)) for image in images]
        if self.training:
            images = [self.color_jitter(image) for image in images]

        images = [
            TF.normalize(TF.to_tensor(image), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            for image in images
        ]
        label_tensor = torch.from_numpy(np.asarray(label, dtype=np.uint8).copy()).long()
        height_tensors = [TF.to_tensor(height).float() for height in heights]
        return images[0], images[1], label_tensor, height_tensors[0], height_tensors[1], name

    def __len__(self):
        return len(self.ids)
