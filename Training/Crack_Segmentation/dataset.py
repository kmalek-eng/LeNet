import random
from pathlib import Path

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


class SegmentationDataset(Dataset):
    def __init__(
        self,
        image_dir,
        mask_dir,
        image_size,
        augment=False,
        augmentation_prob=0.5,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = tuple(image_size)
        self.augment = augment
        self.augmentation_prob = augmentation_prob

        if not self.image_dir.exists():
            raise ValueError(f"Image directory does not exist: {self.image_dir}")
        if not self.mask_dir.exists():
            raise ValueError(f"Mask directory does not exist: {self.mask_dir}")

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
            raise ValueError(
                "Image and mask filenames do not match by stem. " + " | ".join(error_parts)
            )

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

    def _load_pair(self, index):
        image_path, mask_path = self.paired_paths[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        return image, mask

    def _resize_pair(self, image, mask):
        image = self.image_resize(image)
        mask = self.mask_resize(mask)
        return image, mask

    def _augment_pair(self, image, mask):
        if random.random() > self.augmentation_prob:
            return image, mask

        image = self.random_invert(image)
        image = self.color_jitter(image)

        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if random.random() < 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        return image, mask

    def _to_tensors(self, image, mask):
        image = self.to_tensor(image)
        mask = self.to_tensor(mask)
        mask = (mask >= 0.5).float()
        return image, mask

    def __getitem__(self, index):
        image, mask = self._load_pair(index)
        image, mask = self._resize_pair(image, mask)

        if self.augment:
            image, mask = self._augment_pair(image, mask)

        image, mask = self._to_tensors(image, mask)
        return image, mask


def build_dataset_from_args(args, split):
    if split == "test":
        image_dir = args.test_image_dir
        mask_dir = args.test_mask_dir
        augment = False
    elif split == "train":
        image_dir = args.image_dir
        mask_dir = args.mask_dir
        augment = getattr(args, "augment", False)
    elif split == "val":
        image_dir = args.image_dir
        mask_dir = args.mask_dir
        augment = False
    else:
        raise ValueError(f"Unsupported split: {split}")

    return SegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_size=(args.image_height, args.image_width),
        augment=augment,
        augmentation_prob=getattr(args, "augmentation_prob", 0.5),
    )


def create_dataloader(dataset, batch_size, shuffle, num_workers):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )