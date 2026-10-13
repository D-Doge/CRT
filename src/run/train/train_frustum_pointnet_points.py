from __future__ import annotations

# ============================================================
# Paths / config
# ============================================================

TRAIN_ROOT = r"E:\path\to\train_dataset"
TEST_ROOT = r"E:\path\to\test_dataset"
POINTCLOUD_SENSOR = "FOV-Grid_PointCloud"

SAVE_DIR = r"E:\domain_gap_metric\checkpoints\frustum_pointnet"
BEST_CKPT_NAME = "best_frustum_pointnet.pt"

NUM_EPOCHS = 100
BATCH_SIZE = 8
NUM_POINTS = 1024
NUM_OBJECT_POINTS = 512
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0

USE_TENSORBOARD = True
LR_SCHEDULER = "cosine"  # "cosine", "step", or "none"
STEP_LR_STEP_SIZE = 30
STEP_LR_GAMMA = 0.5

USE_GT_MASK_FOR_BOX_TRAIN = True
USE_GT_MASK_FOR_BOX_EVAL = False

DEVICE = "cuda"  # "cuda" or "cpu"


# ============================================================
# Imports
# ============================================================

from pathlib import Path
import random
import numpy as np

import torch
from torch.utils.data import DataLoader

import time
from torch.utils.tensorboard import SummaryWriter

from task_network.data.task_unity_sim_dataloader import UnitySimDetectionDataset
from task_network.frustrum_pointnet.unity.unity_frustum_seg_dataset import (
    UnityFrustumSegDataset,
    frustum_seg_collate,
)
from task_network.frustrum_pointnet.frustum_pointnet_full_points import (
    FrustumPointNet,
    FrustumPointNetLoss,
)


# ============================================================
# Helpers
# ============================================================

def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def make_loader(root: str, shuffle: bool) -> DataLoader:
    base_ds = UnitySimDetectionDataset(
        root_dir=root,
        pointcloud_sensor=POINTCLOUD_SENSOR,
    )

    frustum_ds = UnityFrustumSegDataset(
        base_dataset=base_ds,
        num_points=NUM_POINTS,
        box_margin_px=5.0,
    )

    return DataLoader(
        frustum_ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        collate_fn=frustum_seg_collate,
        drop_last=False,
    )


@torch.no_grad()
def eval_one_epoch(model, loss_fn, loader, device):
    model.eval()

    total_loss = 0.0
    total_seg_loss = 0.0
    total_center_loss = 0.0
    total_box_loss = 0.0
    total_batches = 0

    center_errors = []
    confidences = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        pred = model(
            batch["points_lidar"],
            gt_seg_mask=batch["seg_label"],
            use_gt_mask_for_box=USE_GT_MASK_FOR_BOX_EVAL,
        )

        losses = loss_fn(pred, batch)

        total_loss += float(losses["loss"].item())
        total_seg_loss += float(losses["seg_loss"].item())
        total_center_loss += float(losses["center_loss"].item())
        total_box_loss += float(losses["box_loss"].item())
        total_batches += 1

        center_err = torch.linalg.norm(
            pred["center_lidar"] - batch["gt_center_lidar"],
            dim=1,
        )
        center_errors.append(center_err.detach().cpu())
        confidences.append(pred["confidence"].detach().cpu())

    if total_batches == 0:
        raise RuntimeError("Evaluation loader produced zero batches.")

    center_errors = torch.cat(center_errors)
    confidences = torch.cat(confidences)

    return {
        "loss": total_loss / total_batches,
        "seg_loss": total_seg_loss / total_batches,
        "center_loss": total_center_loss / total_batches,
        "box_loss": total_box_loss / total_batches,
        "center_error_mean": float(center_errors.mean().item()),
        "center_error_median": float(center_errors.median().item()),
        "confidence_mean": float(confidences.mean().item()),
    }


def train_one_epoch(model, loss_fn, loader, optimizer, device):
    model.train()

    total_loss = 0.0
    total_seg_loss = 0.0
    total_center_loss = 0.0
    total_box_loss = 0.0
    total_batches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        pred = model(
            batch["points_lidar"],
            gt_seg_mask=batch["seg_label"],
            use_gt_mask_for_box=USE_GT_MASK_FOR_BOX_TRAIN,
        )

        losses = loss_fn(pred, batch)
        loss = losses["loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += float(loss.item())
        total_seg_loss += float(losses["seg_loss"].item())
        total_center_loss += float(losses["center_loss"].item())
        total_box_loss += float(losses["box_loss"].item())
        total_batches += 1

    if total_batches == 0:
        raise RuntimeError("Training loader produced zero batches.")

    return {
        "loss": total_loss / total_batches,
        "seg_loss": total_seg_loss / total_batches,
        "center_loss": total_center_loss / total_batches,
        "box_loss": total_box_loss / total_batches,
    }


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    best_eval_center_err,
    train_metrics,
    eval_metrics,
):
    ckpt = {
        "epoch": epoch,
        "best_eval_center_err": best_eval_center_err,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "config": {
            "TRAIN_ROOT": TRAIN_ROOT,
            "TEST_ROOT": TEST_ROOT,
            "POINTCLOUD_SENSOR": POINTCLOUD_SENSOR,
            "NUM_EPOCHS": NUM_EPOCHS,
            "NUM_POINTS": NUM_POINTS,
            "NUM_OBJECT_POINTS": NUM_OBJECT_POINTS,
            "BATCH_SIZE": BATCH_SIZE,
            "LEARNING_RATE": LEARNING_RATE,
            "WEIGHT_DECAY": WEIGHT_DECAY,
            "LR_SCHEDULER": LR_SCHEDULER,
        },
    }
    torch.save(ckpt, path)

def log_metrics(writer, split: str, metrics: dict, epoch: int):
    if writer is None:
        return

    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            writer.add_scalar(f"{split}/{k}", v, epoch)
# ============================================================
# Main
# ============================================================

def main():
    set_seed(0)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt_path = save_dir / BEST_CKPT_NAME
    last_ckpt_path = save_dir / "last_frustum_pointnet.pt"

    print(f"[INFO] Device: {device}")
    print("[INFO] Building datasets...")

    train_loader = make_loader(TRAIN_ROOT, shuffle=True)
    test_loader = make_loader(TEST_ROOT, shuffle=False)

    print(f"[INFO] Train batches: {len(train_loader)}")
    print(f"[INFO] Test batches : {len(test_loader)}")

    model = FrustumPointNet(
        input_channels=3,
        num_seg_classes=2,
        num_object_points=NUM_OBJECT_POINTS,
        use_bn=False,
    ).to(device)

    loss_fn = FrustumPointNetLoss().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    if LR_SCHEDULER == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=NUM_EPOCHS,
            eta_min=LEARNING_RATE * 0.01,
        )
    elif LR_SCHEDULER == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=STEP_LR_STEP_SIZE,
            gamma=STEP_LR_GAMMA,
        )
    elif LR_SCHEDULER == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown LR_SCHEDULER: {LR_SCHEDULER}")

    writer = None
    if USE_TENSORBOARD:
        log_dir = save_dir / "tensorboard"
        writer = SummaryWriter(log_dir=str(log_dir))
        print(f"[INFO] TensorBoard log dir: {log_dir}")

    best_eval_center_err = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.perf_counter()

        train_metrics = train_one_epoch(
            model=model,
            loss_fn=loss_fn,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        eval_metrics = eval_one_epoch(
            model=model,
            loss_fn=loss_fn,
            loader=test_loader,
            device=device,
        )

        if scheduler is not None:
            scheduler.step()

        epoch_time_s = time.perf_counter() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        train_metrics["lr"] = float(current_lr)
        train_metrics["epoch_time_s"] = float(epoch_time_s)

        log_metrics(writer, "train", train_metrics, epoch)
        log_metrics(writer, "eval", eval_metrics, epoch)
        if writer is not None:
            writer.add_scalar("time/epoch_time_s", epoch_time_s, epoch)
            writer.add_scalar("optim/lr", current_lr, epoch)

        print(
            f"[Epoch {epoch:03d}/{NUM_EPOCHS}] "
            f"time={epoch_time_s:.1f}s "
            f"lr={current_lr:.2e} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_seg={train_metrics['seg_loss']:.4f} "
            f"train_center={train_metrics['center_loss']:.4f} "
            f"train_box={train_metrics['box_loss']:.4f} | "
            f"eval_loss={eval_metrics['loss']:.4f} "
            f"eval_seg={eval_metrics['seg_loss']:.4f} "
            f"eval_center_loss={eval_metrics['center_loss']:.4f} "
            f"eval_box={eval_metrics['box_loss']:.4f} "
            f"eval_center_err_mean={eval_metrics['center_error_mean']:.4f} "
            f"eval_center_err_med={eval_metrics['center_error_median']:.4f} "
            f"eval_conf={eval_metrics['confidence_mean']:.4f}"
        )

        save_checkpoint(
            path=last_ckpt_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_eval_center_err=best_eval_center_err,
            train_metrics=train_metrics,
            eval_metrics=eval_metrics,
        )

        if eval_metrics["center_error_mean"] < best_eval_center_err:
            best_eval_center_err = eval_metrics["center_error_mean"]

            save_checkpoint(
                path=best_ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_eval_center_err=best_eval_center_err,
                train_metrics=train_metrics,
                eval_metrics=eval_metrics,
            )

            print(
                f"[INFO] Saved new best checkpoint by center error: "
                f"{best_ckpt_path} "
                f"(best_eval_center_err={best_eval_center_err:.6f})"
            )

    if writer is not None:
        writer.close()

    print("[DONE]")
    print(f"Best eval center error: {best_eval_center_err:.6f}")
    print(f"Best checkpoint: {best_ckpt_path}")


if __name__ == "__main__":
    main()