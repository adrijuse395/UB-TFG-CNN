import time
from typing import Dict, Tuple

import torch
import torch.nn as nn


def fine_tune_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    *,
    device: str,
    epochs: int,
    learning_rate: float,
    early_stopping: bool = True,
    patience: int = 3,
    min_improvement: float = 0.1,
    monitor: str = "val_accuracy",
    max_train_batches_per_epoch: int = 0,
    max_val_batches_per_epoch: int = 0,
) -> Dict[str, float]:
    """
    Lightweight fine-tuning pass after compression.
    Returns timing/loss metadata for logging.
    """
    model.to(device)
    is_cuda = device.startswith("cuda") and torch.cuda.is_available()
    patience = max(1, int(patience))
    monitor = str(monitor).strip().lower()
    if monitor not in {"val_accuracy", "val_loss"}:
        monitor = "val_accuracy"

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_loss = 0.0
    total_batches = 0
    best_score = None
    best_epoch = 0
    best_val_accuracy = 0.0
    best_val_loss = float("inf")
    no_improve_epochs = 0
    stopped_early = False
    t0 = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_batches = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            running_batches += 1
            total_loss += float(loss.item())
            total_batches += 1
            if max_train_batches_per_epoch > 0 and (batch_idx + 1) >= max_train_batches_per_epoch:
                break

        mean_epoch_loss = running_loss / max(1, running_batches)
        print(
            f"    [FineTuning] Epoch {epoch + 1}/{epochs} "
            f"- mean loss: {mean_epoch_loss:.4f}"
        )

        val_loss, val_accuracy = _evaluate_validation(
            model,
            val_loader,
            device=device,
            criterion=criterion,
            max_batches=max_val_batches_per_epoch,
        )
        print(
            f"    [FineTuning]           val_loss={val_loss:.4f} "
            f"val_accuracy={val_accuracy:.2f}%"
        )

        current_score = val_accuracy if monitor == "val_accuracy" else -val_loss
        if best_score is None or current_score > (best_score + min_improvement):
            best_score = current_score
            best_epoch = epoch + 1
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if early_stopping and no_improve_epochs >= patience:
            stopped_early = True
            print(
                f"    [FineTuning] Early stopping triggered at epoch {epoch + 1} "
                f"(patience={patience}, min_improvement={min_improvement})."
            )
            break

    if is_cuda:
        torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - t0

    return {
        "fine_tuning_time_s": round(elapsed_s, 4),
        "fine_tuning_mean_loss": round(total_loss / max(1, total_batches), 6),
        "fine_tuning_best_epoch": best_epoch,
        "fine_tuning_stopped_early": int(stopped_early),
        "fine_tuning_monitor": monitor,
        "fine_tuning_min_improvement": float(min_improvement),
        "fine_tuning_patience": patience,
        # Keep CSV compatibility: these fields now store the BEST validation values reached.
        "fine_tuning_last_val_loss": round(best_val_loss if best_epoch > 0 else val_loss, 6),
        "fine_tuning_last_val_accuracy": round(best_val_accuracy if best_epoch > 0 else val_accuracy, 4),
        "fine_tuning_max_train_batches_per_epoch": max_train_batches_per_epoch,
        "fine_tuning_max_val_batches_per_epoch": max_val_batches_per_epoch,
    }


def _evaluate_validation(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    *,
    device: str,
    criterion: nn.Module,
    max_batches: int = 0,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += float(loss.item()) * targets.size(0)

            _, predicted = outputs.max(1)
            total_correct += int((predicted == targets).sum().item())
            total_samples += int(targets.size(0))
            if max_batches > 0 and (batch_idx + 1) >= max_batches:
                break

    mean_loss = total_loss / max(1, total_samples)
    accuracy = 100.0 * total_correct / max(1, total_samples)
    return mean_loss, accuracy
