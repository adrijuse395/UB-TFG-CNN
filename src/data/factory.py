from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class DatasetFactory:
    """
    Minimal dataset factory for CIFAR-10 workflows used in this project.
    Keeps dataset download/cache external (e.g. ./data), which can stay gitignored.
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
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
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
        test_tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )

        train_full = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_tf)
        test_ds = datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_tf)

        n_total = len(train_full)
        n_val = max(1, int(n_total * float(val_fraction)))
        n_train = n_total - n_val
        generator = torch.Generator().manual_seed(int(seed))
        train_ds, val_ds = random_split(train_full, [n_train, n_val], generator=generator)
        # Use deterministic eval transform for validation subset.
        val_ds.dataset = datasets.CIFAR10(root=data_root, train=True, download=False, transform=test_tf)

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
        return train_loader, val_loader, test_loader
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from typing import Tuple, Dict, Any

class DatasetFactory:
    """
    Factory class to instantiate datasets and DataLoaders.
    Designed for easy extensibility to new datasets.
    """
    
    @staticmethod
    def get_dataloaders(dataset_name: str, data_dir: str = "./data", batch_size: int = 128, 
                        val_split: float = 0.1, num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Returns train, validation, and test dataloaders for the specified dataset.
        
        Args:
            dataset_name (str): The name of the dataset (e.g., 'cifar10').
            data_dir (str): Directory to store downloaded data.
            batch_size (int): Batch size for the dataloaders.
            val_split (float): Fraction of training data to use for validation.
            num_workers (int): Number of subprocesses to use for data loading.
            
        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: train_loader, val_loader, test_loader
        """
        dataset_name = dataset_name.lower()
        
        if dataset_name == "cifar10":
            # Standard normalization for ImageNet/CIFAR
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            
            full_train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
            test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
            
        else:
            raise ValueError(f"Dataset '{dataset_name}' is not supported. Please add it to the DatasetFactory.")

        # Create validation split
        train_size = int((1 - val_split) * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
        
        # Override transform for validation set to not use data augmentation
        val_dataset.dataset.transform = transform_test

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        return train_loader, val_loader, test_loader
