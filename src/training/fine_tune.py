import statistics
import time
from typing import Dict, List, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset

CheckpointStrategy = Literal["best_val", "final"]


def _snapshot_state_dict_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _load_state_dict_from_cpu(model: nn.Module, cpu_state: Dict[str, torch.Tensor], device: str) -> None:
    model.load_state_dict({k: v.to(device, non_blocking=True) for k, v in cpu_state.items()})


@torch.no_grad()
def _evaluate_validation(
    model: nn.Module,
    val_loader: DataLoader,
    *,
    device: str,
    criterion: nn.Module,
    max_batches: int = 0,
):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    batches = 0
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        running_loss += float(loss.item())
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += int(predicted.eq(targets).sum().item())
        batches += 1
        if max_batches > 0 and (batch_idx + 1) >= max_batches:
            break

    val_loss = running_loss / max(1, batches)
    val_accuracy = 100.0 * correct / max(1, total)
    return val_loss, val_accuracy


def _fine_tune_one_phase(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
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
    phase_tag: str = "",
    checkpoint_strategy: CheckpointStrategy = "best_val",
    val_overfit_margin: float = 3.0,
    val_overfit_ceiling: float = 96.0,
) -> Dict[str, float]:
    """
    One fine-tuning run with Adam.

    ``checkpoint_strategy``:
      - ``best_val``: keep weights that improve validation vs the pre-FT snapshot
        (never worse than the compressed model before epoch 1).
      - ``final``: keep last-epoch weights; if validation looks overfit vs pre-FT,
        revert to the compressed snapshot (avoids ~99% val / worse test).
    """
    model.to(device)
    is_cuda = device.startswith("cuda") and torch.cuda.is_available()
    patience = max(1, int(patience))
    epochs = max(1, int(epochs))
    monitor = str(monitor).strip().lower()
    if monitor not in {"val_accuracy", "val_loss"}:
        monitor = "val_accuracy"
    if checkpoint_strategy not in {"best_val", "final"}:
        raise ValueError(f"Unknown checkpoint_strategy: {checkpoint_strategy!r}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    prefix = f"    [FineTuning{f' · {phase_tag}' if phase_tag else ''}] "

    compressed_state_cpu = _snapshot_state_dict_cpu(model)
    initial_val_loss, initial_val_accuracy = _evaluate_validation(
        model,
        val_loader,
        device=device,
        criterion=criterion,
        max_batches=max_val_batches_per_epoch,
    )
    print(
        f"{prefix}Pre-FT val: loss={initial_val_loss:.4f} "
        f"accuracy={initial_val_accuracy:.2f}% "
        f"(checkpoint={checkpoint_strategy})"
    )

    best_state_cpu = compressed_state_cpu
    best_epoch = 0
    best_val_accuracy = initial_val_accuracy
    best_val_loss = initial_val_loss
    best_score = initial_val_accuracy if monitor == "val_accuracy" else -initial_val_loss

    total_loss = 0.0
    total_batches = 0
    no_improve_epochs = 0
    stopped_early = False
    reverted_to_compressed = 0
    last_epoch = 0

    t0 = time.perf_counter()

    for epoch in range(epochs):
        last_epoch = epoch + 1
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
        print(f"{prefix}Epoch {epoch + 1}/{epochs} - mean train loss: {mean_epoch_loss:.4f}")

        val_loss, val_accuracy = _evaluate_validation(
            model,
            val_loader,
            device=device,
            criterion=criterion,
            max_batches=max_val_batches_per_epoch,
        )
        print(f"{prefix}          val_loss={val_loss:.4f} val_accuracy={val_accuracy:.2f}%")

        if checkpoint_strategy == "best_val":
            current_score = val_accuracy if monitor == "val_accuracy" else -val_loss
            if current_score > (best_score + min_improvement):
                best_score = current_score
                best_epoch = epoch + 1
                best_val_accuracy = val_accuracy
                best_val_loss = val_loss
                no_improve_epochs = 0
                best_state_cpu = _snapshot_state_dict_cpu(model)
            else:
                no_improve_epochs += 1

            if early_stopping and no_improve_epochs >= patience:
                stopped_early = True
                print(
                    f"{prefix}Early stopping at epoch {epoch + 1} "
                    f"(patience={patience}, min_improvement={min_improvement})."
                )
                break

    if checkpoint_strategy == "best_val":
        _load_state_dict_from_cpu(model, best_state_cpu, device)
        if best_epoch == 0:
            print(f"{prefix}Restored compressed weights (no val improvement over pre-FT).")
        else:
            print(
                f"{prefix}Restored best checkpoint (epoch {best_epoch}, monitor={monitor}, "
                f"val_accuracy={best_val_accuracy:.2f}%)."
            )
    else:
        final_val_loss, final_val_accuracy = _evaluate_validation(
            model,
            val_loader,
            device=device,
            criterion=criterion,
            max_batches=max_val_batches_per_epoch,
        )
        best_val_loss = final_val_loss
        best_val_accuracy = final_val_accuracy
        best_epoch = last_epoch

        overfit = final_val_accuracy > (initial_val_accuracy + val_overfit_margin) or (
            final_val_accuracy > val_overfit_ceiling
        )
        if overfit:
            _load_state_dict_from_cpu(model, compressed_state_cpu, device)
            reverted_to_compressed = 1
            best_epoch = 0
            best_val_accuracy = initial_val_accuracy
            best_val_loss = initial_val_loss
            print(
                f"{prefix}Validation overfit detected "
                f"(final={final_val_accuracy:.2f}% vs pre-FT={initial_val_accuracy:.2f}%) "
                f"→ reverted to compressed weights."
            )
        else:
            print(
                f"{prefix}Keeping last-epoch weights "
                f"(final val_accuracy={final_val_accuracy:.2f}%)."
            )

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
        "fine_tuning_last_val_loss": round(best_val_loss, 6),
        "fine_tuning_last_val_accuracy": round(best_val_accuracy, 4),
        "fine_tuning_initial_val_accuracy": round(initial_val_accuracy, 4),
        "fine_tuning_checkpoint_strategy": checkpoint_strategy,
        "fine_tuning_reverted_to_compressed": reverted_to_compressed,
        "fine_tuning_learning_rate": float(learning_rate),
        "fine_tuning_max_train_batches_per_epoch": max_train_batches_per_epoch,
        "fine_tuning_max_val_batches_per_epoch": max_val_batches_per_epoch,
    }


def fine_tune_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
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
    kfold_splits: int = 1,
    train_full_aug=None,
    train_full_eval=None,
    kfold_seed: int = 42,
    batch_size: int = 128,
    dataloader_num_workers: int = 2,
    pin_memory: bool = True,
    checkpoint_strategy: CheckpointStrategy = "best_val",
    val_overfit_margin: float = 3.0,
    val_overfit_ceiling: float = 96.0,
) -> Dict[str, float]:
    """
    Fine-tuning after compression.

    - Default (kfold_splits <= 1): single run on the provided train/val loaders.

    - K-fold (kfold_splits >= 2): runs K folds on the full 50k CIFAR train split
      (requires train_full_aug / train_full_eval from DatasetFactory), each time
      resetting to the same compressed weights, then a short **final** phase on the
      usual 90/10 train/val loaders. Final epoch budget is min(config epochs, round(mean best epoch per fold)).
    """
    kfold_splits = max(1, int(kfold_splits))
    phase_kw = {
        "checkpoint_strategy": checkpoint_strategy,
        "val_overfit_margin": val_overfit_margin,
        "val_overfit_ceiling": val_overfit_ceiling,
    }

    if kfold_splits <= 1:
        out = _fine_tune_one_phase(
            model,
            train_loader,
            val_loader,
            device=device,
            epochs=max(1, int(epochs)),
            learning_rate=learning_rate,
            early_stopping=early_stopping,
            patience=patience,
            min_improvement=min_improvement,
            monitor=monitor,
            max_train_batches_per_epoch=max_train_batches_per_epoch,
            max_val_batches_per_epoch=max_val_batches_per_epoch,
            phase_tag="single",
            **phase_kw,
        )
        out["fine_tuning_kfold_k"] = 0
        out["fine_tuning_kfold_mean_val_accuracy"] = ""
        out["fine_tuning_kfold_std_val_accuracy"] = ""
        out["fine_tuning_cv_time_s"] = 0.0
        out["fine_tuning_final_epochs_used"] = ""
        out["fine_tuning_mean_best_epoch_cv"] = ""
        return out

    if train_full_aug is None or train_full_eval is None:
        raise ValueError("kfold_splits>1 requires train_full_aug and train_full_eval datasets.")

    n = len(train_full_aug)
    if n != len(train_full_eval):
        raise ValueError("train_full_aug and train_full_eval must have the same length.")

    if kfold_splits > n:
        kfold_splits = n

    initial_cpu = _snapshot_state_dict_cpu(model)
    fold_val_accs: List[float] = []
    fold_best_epochs: List[int] = []
    cv_time = 0.0

    kf = KFold(n_splits=kfold_splits, shuffle=True, random_state=int(kfold_seed))

    for fold_id, (train_idx, val_idx) in enumerate(kf.split(np.zeros((n, 1)))):
        tag = f"CV fold {fold_id + 1}/{kfold_splits}"
        _load_state_dict_from_cpu(model, initial_cpu, device)

        tr_ds = Subset(train_full_aug, train_idx.tolist())
        va_ds = Subset(train_full_eval, val_idx.tolist())
        fold_train_loader = DataLoader(
            tr_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=dataloader_num_workers,
            pin_memory=pin_memory,
        )
        fold_val_loader = DataLoader(
            va_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=dataloader_num_workers,
            pin_memory=pin_memory,
        )

        st = _fine_tune_one_phase(
            model,
            fold_train_loader,
            fold_val_loader,
            device=device,
            epochs=max(1, int(epochs)),
            learning_rate=learning_rate,
            early_stopping=early_stopping,
            patience=patience,
            min_improvement=min_improvement,
            monitor=monitor,
            max_train_batches_per_epoch=max_train_batches_per_epoch,
            max_val_batches_per_epoch=max_val_batches_per_epoch,
            phase_tag=tag,
            **phase_kw,
        )
        fold_val_accs.append(float(st["fine_tuning_last_val_accuracy"]))
        fold_best_epochs.append(int(st["fine_tuning_best_epoch"]))
        cv_time += float(st["fine_tuning_time_s"])

    mean_fold_val = statistics.mean(fold_val_accs) if fold_val_accs else 0.0
    std_fold_val = statistics.pstdev(fold_val_accs) if len(fold_val_accs) > 1 else 0.0
    mean_best_ep = statistics.mean(fold_best_epochs) if fold_best_epochs else 1.0
    final_epochs = min(max(1, int(epochs)), max(1, int(round(mean_best_ep))))

    print(
        f"    [FineTuning · CV summary] k={kfold_splits}, "
        f"mean best val acc={mean_fold_val:.2f}%, std={std_fold_val:.3f}, "
        f"mean best epoch={mean_best_ep:.2f} → final phase epochs={final_epochs}"
    )

    _load_state_dict_from_cpu(model, initial_cpu, device)
    final_stats = _fine_tune_one_phase(
        model,
        train_loader,
        val_loader,
        device=device,
        epochs=final_epochs,
        learning_rate=learning_rate,
        early_stopping=early_stopping,
        patience=patience,
        min_improvement=min_improvement,
        monitor=monitor,
        max_train_batches_per_epoch=max_train_batches_per_epoch,
        max_val_batches_per_epoch=max_val_batches_per_epoch,
        phase_tag="final (90/10 split)",
        **phase_kw,
    )

    total_time = round(cv_time + float(final_stats["fine_tuning_time_s"]), 4)

    out = {
        **final_stats,
        "fine_tuning_time_s": total_time,
        "fine_tuning_cv_time_s": round(cv_time, 4),
        "fine_tuning_final_epochs_used": final_epochs,
        "fine_tuning_kfold_k": kfold_splits,
        "fine_tuning_kfold_mean_val_accuracy": round(mean_fold_val, 4),
        "fine_tuning_kfold_std_val_accuracy": round(std_fold_val, 4),
        "fine_tuning_mean_best_epoch_cv": round(mean_best_ep, 4),
    }
    return out
