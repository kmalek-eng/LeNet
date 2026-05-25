import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from model import UNET_MODEL


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--mask-dir", type=str, required=True)

    parser.add_argument("--image-height", type=int, default=448)
    parser.add_argument("--image-width", type=int, default=448)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)

    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--selected-fold", type=int, default=-1)
    parser.add_argument("--random-seed", type=int, default=42)

    parser.add_argument("--channel-reduction", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--augmentation-prob", type=float, default=0.5)

    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size, augment=False, augmentation_prob=0.5):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = tuple(image_size)
        self.augment = augment
        self.augmentation_prob = augmentation_prob

        image_paths = sorted([p for p in self.image_dir.iterdir() if p.is_file()])
        mask_paths = sorted([p for p in self.mask_dir.iterdir() if p.is_file()])

        if len(image_paths) == 0:
            raise ValueError(f"No image files found in: {self.image_dir}")
        if len(mask_paths) == 0:
            raise ValueError(f"No mask files found in: {self.mask_dir}")

        image_map = {p.stem: p for p in image_paths}
        mask_map = {p.stem: p for p in mask_paths}

        if len(image_map) != len(image_paths):
            raise ValueError("Duplicate image file stems found in image directory")
        if len(mask_map) != len(mask_paths):
            raise ValueError("Duplicate mask file stems found in mask directory")

        if set(image_map.keys()) != set(mask_map.keys()):
            missing_masks = sorted(set(image_map.keys()) - set(mask_map.keys()))
            missing_images = sorted(set(mask_map.keys()) - set(image_map.keys()))

            error_parts = []
            if missing_masks:
                error_parts.append(f"Missing masks for images: {missing_masks[:10]}")
            if missing_images:
                error_parts.append(f"Missing images for masks: {missing_images[:10]}")
            raise ValueError("Image and mask filenames do not match by stem. " + " | ".join(error_parts))

        self.paired_paths = [(image_map[name], mask_map[name]) for name in sorted(image_map.keys())]

        self.image_resize = transforms.Resize(self.image_size)
        self.mask_resize = transforms.Resize(
            self.image_size,
            interpolation=transforms.InterpolationMode.NEAREST,
        )
        self.to_tensor = transforms.ToTensor()
        self.color_jitter = transforms.ColorJitter(
            brightness=0.35,
            contrast=0.22,
            hue=0.02,
        )
        self.random_invert = transforms.RandomInvert(p=0.05)

    def __len__(self):
        return len(self.paired_paths)

    def __getitem__(self, index):
        image_path, mask_path = self.paired_paths[index]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.image_resize(image)
        mask = self.mask_resize(mask)

        if self.augment and np.random.rand() <= self.augmentation_prob:
            image = self.random_invert(image)
            image = self.color_jitter(image)

            if np.random.rand() < 0.5:
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)

            if np.random.rand() < 0.5:
                image = transforms.functional.vflip(image)
                mask = transforms.functional.vflip(mask)

        image = self.to_tensor(image)
        mask = self.to_tensor(mask)
        mask = (mask >= 0.5).float()

        return image, mask


class AugmentedSubset(Dataset):
    def __init__(self, dataset, indices, augment=False):
        self.dataset = dataset
        self.indices = list(indices)
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_augment = self.dataset.augment
        self.dataset.augment = self.augment
        item = self.dataset[self.indices[idx]]
        self.dataset.augment = original_augment
        return item


def build_loaders_for_fold(dataset, train_indices, val_indices, batch_size, num_workers):
    train_dataset = AugmentedSubset(dataset, train_indices, augment=True)
    val_dataset = AugmentedSubset(dataset, val_indices, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def dice_score_from_logits(logits, targets, threshold=0.5, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)

    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean().item()


def f1_score_from_logits(logits, targets, threshold=0.5, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    tp = (preds * targets).sum(dim=1)
    fp = (preds * (1.0 - targets)).sum(dim=1)
    fn = ((1.0 - preds) * targets).sum(dim=1)

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    f1 = (2.0 * precision * recall) / (precision + recall + eps)
    return f1.mean().item()


def run_train_epoch(model, loader, optimizer, criterion, device, threshold):
    model.train()

    total_loss = 0.0
    total_dice = 0.0
    total_f1 = 0.0
    total_batches = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_dice += dice_score_from_logits(logits.detach(), masks, threshold)
        total_f1 += f1_score_from_logits(logits.detach(), masks, threshold)
        total_batches += 1

    return {
        "loss": total_loss / total_batches,
        "dice": total_dice / total_batches,
        "f1": total_f1 / total_batches,
    }


@torch.no_grad()
def run_val_epoch(model, loader, criterion, device, threshold):
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_f1 = 0.0
    total_batches = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)

        total_loss += loss.item()
        total_dice += dice_score_from_logits(logits, masks, threshold)
        total_f1 += f1_score_from_logits(logits, masks, threshold)
        total_batches += 1

    return {
        "loss": total_loss / total_batches,
        "dice": total_dice / total_batches,
        "f1": total_f1 / total_batches,
    }


def save_checkpoint(save_path, model, optimizer, epoch, fold_index, metrics):
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "fold": fold_index,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        save_path,
    )


def train_single_fold(args, dataset, fold_index, train_indices, val_indices, device):
    train_loader, val_loader = build_loaders_for_fold(
        dataset=dataset,
        train_indices=train_indices,
        val_indices=val_indices,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = UNET_MODEL(
        args=None,
        channel_reduction=args.channel_reduction,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_val_f1 = -1.0
    best_metrics = None

    fold_dir = Path(args.save_dir) / f"fold_{fold_index}"
    best_model_path = fold_dir / "best_model.pth"

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            threshold=args.threshold,
        )

        val_metrics = run_val_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            threshold=args.threshold,
        )

        print(
            f"Fold {fold_index} | "
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_metrics['loss']:.6f} | "
            f"Train Dice: {train_metrics['dice']:.6f} | "
            f"Train F1: {train_metrics['f1']:.6f} | "
            f"Val Loss: {val_metrics['loss']:.6f} | "
            f"Val Dice: {val_metrics['dice']:.6f} | "
            f"Val F1: {val_metrics['f1']:.6f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_metrics = val_metrics
            save_checkpoint(
                save_path=best_model_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                fold_index=fold_index,
                metrics=val_metrics,
            )

    return best_metrics


def main():
    args = parse_args()

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )

    dataset = SegmentationDataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        image_size=(args.image_height, args.image_width),
        augment=False,
        augmentation_prob=args.augmentation_prob,
    )

    indices = np.arange(len(dataset))
    kfold = KFold(
        n_splits=args.num_folds,
        shuffle=True,
        random_state=args.random_seed,
    )

    fold_results = []

    for fold_number, (train_indices, val_indices) in enumerate(kfold.split(indices), start=1):
        if args.selected_fold != -1 and fold_number != args.selected_fold:
            continue

        best_metrics = train_single_fold(
            args=args,
            dataset=dataset,
            fold_index=fold_number,
            train_indices=train_indices,
            val_indices=val_indices,
            device=device,
        )

        fold_results.append((fold_number, best_metrics))

    if len(fold_results) == 0:
        raise ValueError("No fold was trained. Check --selected-fold and --num-folds.")

    mean_f1 = np.mean([m["f1"] for _, m in fold_results])
    mean_dice = np.mean([m["dice"] for _, m in fold_results])
    mean_loss = np.mean([m["loss"] for _, m in fold_results])

    print(f"Mean Val Loss: {mean_loss:.6f}")
    print(f"Mean Val Dice: {mean_dice:.6f}")
    print(f"Mean Val F1: {mean_f1:.6f}")


if __name__ == "__main__":
    main()