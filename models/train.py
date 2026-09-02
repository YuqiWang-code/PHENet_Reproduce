"""Train PHENet on a BCD-foggy dataset and keep only the best-F1 checkpoint."""

import argparse
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from dataloaders import make_data_loaders
from modeling.PHENet import PHENet
from modeling.sync_batchnorm.replicate import patch_replication_callback
from utils.loss import build_change_loss
from utils.metrics import Evaluator


class RunLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", encoding="utf-8", buffering=1)

    def write(self, message=""):
        print(message, flush=True)
        self.handle.write(message + "\n")

    def close(self):
        self.handle.close()


class PhysicalLosses(nn.Module):
    @staticmethod
    def tv_loss(transmission):
        diff_x = torch.abs(transmission[:, :, 1:, :] - transmission[:, :, :-1, :]).mean()
        diff_y = torch.abs(transmission[:, :, :, 1:] - transmission[:, :, :, :-1]).mean()
        return diff_x + diff_y

    @staticmethod
    def dark_channel_loss(clear_image):
        return torch.min(clear_image, dim=1).values.square().mean()


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(7, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, value):
        return self.network(value)


class ShallowCNN(nn.Module):
    """Frozen feature extractor used by the released pseudo-label procedure."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

    def forward(self, value):
        value = F.max_pool2d(F.relu(self.conv1(value)), 2)
        value = F.max_pool2d(F.relu(self.conv2(value)), 2)
        return F.relu(self.conv3(value))


@torch.no_grad()
def generate_pseudo_label(image_a, image_b, extractor):
    difference = torch.linalg.vector_norm(extractor(image_a) - extractor(image_b), dim=1)
    masks = []
    for sample in difference:
        array = sample.detach().float().cpu().numpy()
        span = float(array.max() - array.min())
        normalized = np.zeros_like(array, dtype=np.uint8)
        if span > 0:
            normalized = ((array - array.min()) / span * 255.0).astype(np.uint8)
        _, mask = cv2.threshold(normalized, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        masks.append(torch.from_numpy(mask).unsqueeze(0))
    pseudo = torch.stack(masks).to(image_a.device, dtype=image_a.dtype)
    return F.interpolate(pseudo, size=image_a.shape[-2:], mode="nearest")


def set_requires_grad(module, enabled):
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def unwrap(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def model_inputs(batch, device):
    image_a, image_b, target, height_a, height_b, names = batch
    non_blocking = device.type == "cuda"
    return (
        image_a.to(device, non_blocking=non_blocking),
        image_b.to(device, non_blocking=non_blocking),
        target.to(device, non_blocking=non_blocking),
        height_a.to(device, non_blocking=non_blocking),
        height_b.to(device, non_blocking=non_blocking),
        names,
    )


def benchmark_model(loader, network, device, logger, warmup=20, max_iters=50):
    model = unwrap(network)
    network.eval()
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    param_memory_mb = total_params * 4.0 / 1024.0 / 1024.0

    sample = next(iter(loader))
    image_a, image_b, _, height_a, height_b, _ = model_inputs(sample, device)
    dummy = (image_a[:1], image_b[:1], height_a[:1], height_b[:1])
    flops_g = None
    thop_params_m = None
    try:
        from thop import profile

        with torch.no_grad():
            flops, thop_params = profile(model, inputs=dummy, verbose=False)
        flops_g = flops / 1e9
        thop_params_m = thop_params / 1e6
    except Exception as error:
        logger.write(f"THOP profile failed: {error}")

    total_time = 0.0
    total_images = 0
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            for _ in range(warmup):
                network(*dummy)
        torch.cuda.synchronize(device)

    with torch.inference_mode():
        for index, batch in enumerate(tqdm(loader, desc="Benchmark", leave=False)):
            if index >= max_iters:
                break
            batch_a, batch_b, _, batch_ha, batch_hb, _ = model_inputs(batch, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            network(batch_a, batch_b, batch_ha, batch_hb)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            total_time += time.perf_counter() - start
            total_images += batch_a.shape[0]

    fps = total_images / total_time if total_time else 0.0
    allocated = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None
    reserved = torch.cuda.max_memory_reserved(device) / 1024**2 if device.type == "cuda" else None
    logger.write("[Model Statistics]")
    logger.write(f"Params(M): {total_params / 1e6:.3f}")
    logger.write(f"Trainable Params(M): {trainable_params / 1e6:.3f}")
    logger.write(f"Params THOP(M): {thop_params_m:.3f}" if thop_params_m is not None else "Params THOP(M): N/A")
    logger.write(f"FLOPs(G): {flops_g:.3f}" if flops_g is not None else "FLOPs(G): N/A")
    logger.write(f"FPS: {fps:.3f}")
    logger.write(f"Param Memory(MB): {param_memory_mb:.2f}")
    logger.write(f"GPU Mem Allocated(MB): {allocated:.2f}" if allocated is not None else "GPU Mem Allocated(MB): N/A")
    logger.write(f"GPU Mem Reserved(MB): {reserved:.2f}" if reserved is not None else "GPU Mem Reserved(MB): N/A")
    logger.write()


class Trainer:
    def make_pseudo_label(self, image_a, image_b):
        if self.args.pseudo_mode == "frozen":
            return generate_pseudo_label(
                image_a,
                image_b,
                self.pseudo_extractor,
            )

        if self.args.pseudo_mode == "zero":
            return torch.zeros(
                (
                    image_a.shape[0],
                    1,
                    image_a.shape[2],
                    image_a.shape[3],
                ),
                device=image_a.device,
                dtype=image_a.dtype,
            )

        raise ValueError(
            f"Unsupported pseudo mode: "
            f"{self.args.pseudo_mode}"
        )

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.device = torch.device(f"cuda:{args.gpu_ids[0]}" if args.cuda else "cpu")
        if args.cuda:
            torch.cuda.set_device(self.device)
        self.train_loader, self.val_loader, _ = make_data_loaders(args)

        model = PHENet(
            num_classes=2,
            backbone="mobilenet",
            output_stride=args.out_stride,
            sync_bn=args.sync_bn,
            freeze_bn=args.freeze_bn,
        ).to(self.device)
        parameter_groups = [
            {"params": model.get_1x_lr_params(), "lr": args.lr},
            {"params": model.get_10x_lr_params(), "lr": args.lr},
        ]
        self.optimizer = torch.optim.Adam(parameter_groups, lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.9, patience=5
        )
        self.discriminator = Discriminator().to(self.device)
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        self.pseudo_extractor = ShallowCNN().to(self.device).eval()
        set_requires_grad(self.pseudo_extractor, False)

        if args.cuda and len(args.gpu_ids) > 1:
            model = nn.DataParallel(model, device_ids=args.gpu_ids, output_device=args.gpu_ids[0])
            if args.sync_bn:
                patch_replication_callback(model)
        self.model = model

        (
            self.change_loss,
            self.classification_loss_name,
        ) = build_change_loss(
            args.change_loss_mode
        )

        self.physical_loss = PhysicalLosses()
        self.best_f1 = -1.0
        self.start_epoch = 0
        if args.resume:
            self._load_checkpoint(args.resume)

    def _load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        unwrap(self.model).load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        self.discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer"])
        self.best_f1 = float(checkpoint.get("best_f1", -1.0))
        self.start_epoch = int(checkpoint["epoch"])
        self.logger.write(f"Resumed checkpoint: {path} (next epoch {self.start_epoch + 1})")
        if "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

    def train_epoch(self, epoch):
        self.model.train()
        self.discriminator.train()
        totals = {key: 0.0 for key in ("total", "change", "cls", "dice", "tv", "dark", "adv_g", "disc")}
        progress = tqdm(self.train_loader, desc=f"Train {epoch + 1}/{self.args.epochs}", leave=False)
        for batch in progress:
            image_a, image_b, target, height_a, height_b, _ = model_inputs(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(image_a, image_b, height_a, height_b)
            logits, fog_a, clear_a, trans_a, _, fog_b, clear_b, trans_b, _ = outputs
            change, classification, dice = self.change_loss(logits, target)
            tv = self.physical_loss.tv_loss(trans_a) + self.physical_loss.tv_loss(trans_b)
            dark = self.physical_loss.dark_channel_loss(clear_a) + self.physical_loss.dark_channel_loss(clear_b)
            pseudo = self.make_pseudo_label(
                image_a,
                image_b,
            )

            set_requires_grad(self.discriminator, False)
            fake_logits = self.discriminator(torch.cat((fog_a, fog_b, pseudo), dim=1))
            adv_g = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
            total = change + 0.2 * tv + 0.5 * dark + 0.5 * adv_g
            total.backward()
            self.optimizer.step()

            set_requires_grad(self.discriminator, True)
            self.discriminator_optimizer.zero_grad(set_to_none=True)
            real_logits = self.discriminator(torch.cat((image_a, image_b, pseudo), dim=1))
            detached_fake = self.discriminator(torch.cat((fog_a.detach(), fog_b.detach(), pseudo), dim=1))
            disc = 0.5 * (
                F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
                + F.binary_cross_entropy_with_logits(detached_fake, torch.zeros_like(detached_fake))
            )
            disc.backward()
            self.discriminator_optimizer.step()

            values = {
                "total": total,
                "change": change,
                "cls": classification,
                "dice": dice,
                "tv": tv,
                "dark": dark,
                "adv_g": adv_g,
                "disc": disc,
            }
            for key, value in values.items():
                totals[key] += float(value.detach())
            progress.set_postfix(loss=f"{float(total.detach()):.4f}")

        count = len(self.train_loader)
        averages = {key: value / count for key, value in totals.items()}
        return averages

    @torch.inference_mode()
    def validate(self, epoch):
        self.model.eval()
        evaluator = Evaluator()
        total_loss = 0.0
        progress = tqdm(self.val_loader, desc=f"Validate {epoch + 1}/{self.args.epochs}", leave=False)
        for batch in progress:
            image_a, image_b, target, height_a, height_b, _ = model_inputs(batch, self.device)
            logits = self.model(image_a, image_b, height_a, height_b)[0]
            change, _, _ = self.change_loss(logits, target)
            total_loss += float(change)
            prediction = logits.argmax(dim=1)
            evaluator.add_batch(target.cpu().numpy(), prediction.cpu().numpy())
        return total_loss / len(self.val_loader), evaluator.compute()

    def save_best(self, epoch, metrics):
        f1 = metrics["F1"]
        if f1 <= self.best_f1:
            return
        self.best_f1 = f1
        for old_checkpoint in self.args.output_dir.glob("best_F1=*.pth"):
            old_checkpoint.unlink()
        destination = self.args.output_dir / f"best_F1={f1:.4f}.pth"
        torch.save(
            {
                "epoch": epoch + 1,
                "state_dict": unwrap(self.model).state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "discriminator_state_dict": self.discriminator.state_dict(),
                "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
                "best_f1": self.best_f1,
                "metrics": metrics,
                "args": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(self.args).items()
                },
            },
            destination,
        )

    def step_scheduler(self, val_loss):
        self.scheduler.step(val_loss)

        main_lr = self.optimizer.param_groups[0]["lr"]
        for group in self.discriminator_optimizer.param_groups:
            group["lr"] = main_lr


def parse_args():
    parser = argparse.ArgumentParser(description="PHENet training for BCD-foggy datasets")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--save-dir", default="/home/yqwang/project/PHENet/saved_models")
    parser.add_argument("--val-split", default="test", help="Use test to select best as requested")
    parser.add_argument("--label-dir", default="auto", help="auto, label, or GT")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--out-stride", type=int, default=16, choices=(8, 16))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--test-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--sync-bn", action="store_true")
    parser.add_argument("--freeze-bn", action="store_true")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--pseudo-mode",
        default="frozen",
        choices=("frozen", "zero"),
        help=(
            "PGARM discriminator conditioning: "
            "'frozen' uses the current frozen-random ShallowCNN pseudo-label; "
            "'zero' uses an all-zero pseudo channel for controlled ablation."
        ),
    )
    parser.add_argument(
        "--change-loss-mode",
        default="current",
        choices=(
            "current",
            "bce_fg_dice",
        ),
        help=(
            "Change-detection loss ablation: "
            "'current' uses 2-class CE + foreground/background mean Dice; "
            "'bce_fg_dice' uses binary BCEWithLogits + foreground Dice."
        ),
    )
    parser.add_argument("--resume")
    parser.add_argument("--benchmark-warmup", type=int, default=20)
    parser.add_argument("--benchmark-iters", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    args.gpu_ids = [int(value) for value in args.gpu_ids.split(",") if value.strip()]
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if not args.cuda:
        args.gpu_ids = []
        args.sync_bn = False
    args.output_dir = Path(args.save_dir).expanduser().resolve() / args.dataset_name
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        # A fresh run must leave exactly one best checkpoint, never stale
        # latest/epoch checkpoints from an earlier run.
        for stale_checkpoint in args.output_dir.glob("*.pth"):
            stale_checkpoint.unlink()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    logger = RunLogger(args.output_dir / "train.log")
    try:
        trainer = Trainer(args, logger)
        benchmark_model(
            trainer.val_loader,
            trainer.model,
            trainer.device,
            logger,
            warmup=args.benchmark_warmup,
            max_iters=args.benchmark_iters,
        )
        logger.write(
            f"Pseudo Mode: {args.pseudo_mode}"
        )
        logger.write(
            f"Change Loss Mode: {args.change_loss_mode}"
        )
        logger.write(
            "Epoch\tLR\tTotalLoss\tChangeLoss\t"
            f"{trainer.classification_loss_name}"
            "\tDice\tTV\tDark\tAdvG\tDisc\tValLoss"
            "\tRecall\tPrecision\tOA\tF1\tIoU\tKappa"
        )
        for epoch in range(trainer.start_epoch, args.epochs):
            # LR actually used in this epoch.
            lr = trainer.optimizer.param_groups[0]["lr"]

            losses = trainer.train_epoch(epoch)
            val_loss, metrics = trainer.validate(epoch)

            # Update LR for the NEXT epoch.
            trainer.step_scheduler(val_loss)

            trainer.save_best(epoch, metrics)

            logger.write(
                f"{epoch + 1}\t{lr:.8g}\t{losses['total']:.6f}\t{losses['change']:.6f}"
                f"\t{losses['cls']:.6f}\t{losses['dice']:.6f}"
                f"\t{losses['dark']:.6f}\t{losses['adv_g']:.6f}\t{losses['disc']:.6f}"
                f"\t{val_loss:.6f}\t{metrics['Recall'] * 100:.4f}\t{metrics['Precision'] * 100:.4f}"
                f"\t{metrics['OA'] * 100:.4f}\t{metrics['F1'] * 100:.4f}"
                f"\t{metrics['IoU'] * 100:.4f}\t{metrics['Kappa'] * 100:.4f}"
            )
    finally:
        logger.close()


if __name__ == "__main__":
    main()
