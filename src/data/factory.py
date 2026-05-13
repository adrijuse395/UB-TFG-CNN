from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class DatasetFactory:
    """
    Dataset/DataLoader factory for this project.
    Currently supports CIFAR-10.
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
    ) -> Tuple[DataLoader, DataLoader, DataLoader, datasets.CIFAR10, datasets.CIFAR10]:
        name = str(dataset_name).strip().lower()
        if name != "cifar10":
            raise ValueError(f"Unsupported dataset: {dataset_name}. Only 'cifar10' is currently implemented.")

        train_tf = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
        eval_tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )

        train_full = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_tf)
        test_ds = datasets.CIFAR10(root=data_root, train=False, download=True, transform=eval_tf)

        n_total = len(train_full)
        n_val = max(1, int(n_total * float(val_fraction)))
        n_train = n_total - n_val
        generator = torch.Generator().manual_seed(int(seed))
        train_ds, val_ds = random_split(train_full, [n_train, n_val], generator=generator)

        # Validation should use deterministic evaluation transforms.
        val_ds.dataset = datasets.CIFAR10(root=data_root, train=True, download=False, transform=eval_tf)

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
        # Full 50k train with eval transforms — same sample order as `train_full` (aug),
        # for K-fold validation subsets aligned by index.
        train_full_eval = datasets.CIFAR10(root=data_root, train=True, download=False, transform=eval_tf)
        return train_loader, val_loader, test_loader, train_full, train_full_eval
