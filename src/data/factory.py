from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# Dataset-specific normalization statistics
_DATASET_STATS = {
    "cifar10":  {"mean": (0.4914, 0.4822, 0.4465), "std": (0.2470, 0.2435, 0.2616)},
    "cifar100": {"mean": (0.5071, 0.4867, 0.4408), "std": (0.2675, 0.2565, 0.2761)},
}

_DATASET_CLASSES = {
    "cifar10":  datasets.CIFAR10,
    "cifar100": datasets.CIFAR100,
}


class DatasetFactory:
    """
    Dataset/DataLoader factory for this project.
    Supports CIFAR-10 and CIFAR-100 (both 32×32 images).
    """

    @staticmethod
    def get_dataloaders(
        dataset_name: str = "cifar10",
        batch_size: int = 128,
        data_root: str = "./data",
        val_fraction: float = 0.1,
        num_workers: int = 2,
        pin_memory: bool = True,
        seed: int = 42,
    ) -> Tuple[DataLoader, DataLoader, DataLoader, datasets.VisionDataset, datasets.VisionDataset]:
        name = str(dataset_name).strip().lower()
        if name not in _DATASET_CLASSES:
            supported = ", ".join(sorted(_DATASET_CLASSES.keys()))
            raise ValueError(
                f"Unsupported dataset: {dataset_name!r}. Supported: {supported}"
            )

        stats = _DATASET_STATS[name]
        ds_class = _DATASET_CLASSES[name]

        train_tf = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(stats["mean"], stats["std"]),
            ]
        )
        eval_tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(stats["mean"], stats["std"]),
            ]
        )

        train_full = ds_class(root=data_root, train=True, download=True, transform=train_tf)
        test_ds = ds_class(root=data_root, train=False, download=True, transform=eval_tf)

        n_total = len(train_full)
        n_val = max(1, int(n_total * float(val_fraction)))
        n_train = n_total - n_val
        generator = torch.Generator().manual_seed(int(seed))
        train_ds, val_ds = random_split(train_full, [n_train, n_val], generator=generator)

        # Validation should use deterministic evaluation transforms.
        val_ds.dataset = ds_class(root=data_root, train=True, download=False, transform=eval_tf)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        # Full train set with eval transforms — for K-fold validation subsets.
        train_full_eval = ds_class(root=data_root, train=True, download=False, transform=eval_tf)
        return train_loader, val_loader, test_loader, train_full, train_full_eval

