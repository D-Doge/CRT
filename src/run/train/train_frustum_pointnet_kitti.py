from __future__ import annotations

# ============================================================
# Paths / config
# ============================================================

# Accepts either:
#   KITTI_ROOT/training/{image_2,velodyne,calib,label_2}
# or:
#   KITTI_ROOT/{image_2,velodyne,calib,label_2}
KITTI_ROOT = r"E:\path\to\KITTI\object"

# Optional standard KITTI train/val split files (one six-digit frame ID per line).
# If either file is missing, this script creates a deterministic random split
# under SAVE_DIR/generated_splits.
TRAIN_SPLIT_FILE = r"E:\path\to\KITTI\splits\train.txt"
VAL_SPLIT_FILE = r"E:\path\to\KITTI\splits\val.txt"
VAL_FRACTION_IF_GENERATED = 0.20
SPLIT_SEED = 0

VEHICLE_CLASSES = ("Car", "Van", "Truck")

SAVE_DIR = r"E:\domain_gap_metric\checkpoints\frustum_pointnet_kitti"
BEST_CKPT_NAME = "best_frustum_pointnet_kitti.pt"

NUM_EPOCHS = 100
BATCH_SIZE = 8
NUM_POINTS = 1024
NUM_OBJECT_POINTS = 512
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0

BOX_MARGIN_PX = 5.0
MIN_BOX_HEIGHT_PX = 25.0
MAX_TRUNCATION = 1.0
MAX_OCCLUSION = 3
MIN_FRUSTUM_POINTS = 5
MIN_OBJECT_POINTS = 1
VALIDATE_AND_CACHE_FRUSTUM_INDEX = True

USE_TENSORBOARD = True
LR_SCHEDULER = "cosine"  # "cosine", "step", or "none"
STEP_LR_STEP_SIZE = 30
STEP_LR_GAMMA = 0.5

USE_GT_MASK_FOR_BOX_TRAIN = True
USE_GT_MASK_FOR_BOX_EVAL = False

# Set this to a compatible checkpoint to fine-tune; leave None to train fresh.
INIT_MODEL_CKPT = None
DEVICE = "cuda"  # "cuda" or "cpu"


# ============================================================
# Imports
# ============================================================

from pathlib import Path
import random
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from task_network.data.kitti_detection_dataset import KittiDetectionDataset
from task_network.frustrum_pointnet.kitti.kitti_frustum_dataset import (
    KittiFrustumDataset,
    kitti_frustum_collate,
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


def move_batch_to_device(batch: Dict[str, object], device: torch.device):
    out: Dict[str, object] = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def _write_split(path: Path, frame_ids: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(frame_ids) + "\n", encoding="utf-8")


def resolve_train_val_ids(save_dir: Path) -> Tuple[List[str], List[str]]:
    train_path = Path(TRAIN_SPLIT_FILE)
    val_path = Path(VAL_SPLIT_FILE)

    if train_path.is_file() and val_path.is_file():
        train_ids = KittiDetectionDataset.read_split_file(train_path)
        val_ids = KittiDetectionDataset.read_split_file(val_path)
        print(f"[INFO] Using supplied train split: {train_path} ({len(train_ids)} frames)")
        print(f"[INFO] Using supplied val split  : {val_path} ({len(val_ids)} frames)")
    else:
        all_ids = KittiDetectionDataset.discover_frame_ids(KITTI_ROOT)
        rng = random.Random(SPLIT_SEED)
        rng.shuffle(all_ids)

        n_val = max(1, int(round(len(all_ids) * VAL_FRACTION_IF_GENERATED)))
        val_ids = sorted(all_ids[:n_val])
        train_ids = sorted(all_ids[n_val:])

        generated_dir = save_dir / "generated_splits"
        train_path = generated_dir / "train.txt"
        val_path = generated_dir / "val.txt"
        _write_split(train_path, train_ids)
        _write_split(val_path, val_ids)

        print(
            "[WARN] Supplied split files were not both available. "
            "Created a deterministic random split instead."
        )
        print(f"[INFO] Generated train split: {train_path} ({len(train_ids)} frames)")
        print(f"[INFO] Generated val split  : {val_path} ({len(val_ids)} frames)")

    overlap = set(train_ids).intersection(val_ids)
    if overlap:
        raise ValueError(f"Train/validation splits overlap in {len(overlap)} frame IDs")
    if not train_ids or not val_ids:
        raise ValueError("Train and validation splits must both be non-empty")
    return train_ids, val_ids


def make_loader(
    frame_ids: Sequence[str],
    *,
    split_name: str,
    shuffle: bool,
    save_dir: Path,
) -> DataLoader:
    base_ds = KittiDetectionDataset(
        root_dir=KITTI_ROOT,
        frame_ids=frame_ids,
        class_whitelist=VEHICLE_CLASSES,
        return_lidar_reflectance=False,
    )

    index_cache = (
        save_dir / "frustum_index_cache" / f"{split_name}.json"
        if VALIDATE_AND_CACHE_FRUSTUM_INDEX
        else None
    )

    frustum_ds = KittiFrustumDataset(
        base_dataset=base_ds,
        num_points=NUM_POINTS,
        box_margin_px=BOX_MARGIN_PX,
        min_box_height_px=MIN_BOX_HEIGHT_PX,
        max_truncation=MAX_TRUNCATION,
        max_occlusion=MAX_OCCLUSION,
        min_frustum_points=MIN_FRUSTUM_POINTS,
        min_object_points=MIN_OBJECT_POINTS,
        validate_items=VALIDATE_AND_CACHE_FRUSTUM_INDEX,
        valid_index_cache=index_cache,
        deterministic_sampling=not shuffle,
        sampling_seed=SPLIT_SEED,
    )

    print(
        f"[INFO] {split_name}: {len(base_ds)} frames, "
        f"{len(frustum_ds)} valid vehicle frustums"
    )

    return DataLoader(
        frustum_ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        collate_fn=kitti_frustum_collate,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0,
    )


@torch.no_grad()
def eval_one_epoch(model, loss_fn, loader, device):
    model.eval()

    totals = {"loss": 0.0, "seg_loss": 0.0, "center_loss": 0.0, "box_loss": 0.0}
    total_batches = 0
    center_errors = []
    confidences = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        points = batch["points_lidar"]
        seg_label = batch["seg_label"]
        assert torch.is_tensor(points)
        assert torch.is_tensor(seg_label)

        pred = model(
            points,
            gt_seg_mask=seg_label,
            use_gt_mask_for_box=USE_GT_MASK_FOR_BOX_EVAL,
        )
        losses = loss_fn(pred, batch)

        for key in totals:
            totals[key] += float(losses[key].item())
        total_batches += 1

        gt_center = batch["gt_center_lidar"]
        assert torch.is_tensor(gt_center)
        center_errors.append(
            torch.linalg.norm(pred["center_lidar"] - gt_center, dim=1).cpu()
        )
        confidences.append(pred["confidence"].cpu())

    if total_batches == 0:
        raise RuntimeError("Evaluation loader produced zero batches")

    center_errors_t = torch.cat(center_errors)
    confidences_t = torch.cat(confidences)
    return {
        **{key: value / total_batches for key, value in totals.items()},
        "center_error_mean": float(center_errors_t.mean().item()),
        "center_error_median": float(center_errors_t.median().item()),
        "confidence_mean": float(confidences_t.mean().item()),
    }


def train_one_epoch(model, loss_fn, loader, optimizer, device):
    model.train()

    totals = {"loss": 0.0, "seg_loss": 0.0, "center_loss": 0.0, "box_loss": 0.0}
    total_batches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        points = batch["points_lidar"]
        seg_label = batch["seg_label"]
        assert torch.is_tensor(points)
        assert torch.is_tensor(seg_label)

        pred = model(
            points,
            gt_seg_mask=seg_label,
            use_gt_mask_for_box=USE_GT_MASK_FOR_BOX_TRAIN,
        )
        losses = loss_fn(pred, batch)
        loss = losses["loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        for key in totals:
            totals[key] += float(losses[key].item())
        total_batches += 1

    if total_batches == 0:
        raise RuntimeError("Training loader produced zero batches")
    return {key: value / total_batches for key, value in totals.items()}


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    epoch: int,
    best_eval_center_err: float,
    train_metrics: Dict[str, float],
    eval_metrics: Dict[str, float],
) -> None:
    checkpoint = {
        "epoch": epoch,
        "best_eval_center_err": best_eval_center_err,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "config": {
            "KITTI_ROOT": KITTI_ROOT,
            "VEHICLE_CLASSES": VEHICLE_CLASSES,
            "NUM_EPOCHS": NUM_EPOCHS,
            "NUM_POINTS": NUM_POINTS,
            "NUM_OBJECT_POINTS": NUM_OBJECT_POINTS,
            "BATCH_SIZE": BATCH_SIZE,
            "LEARNING_RATE": LEARNING_RATE,
            "WEIGHT_DECAY": WEIGHT_DECAY,
            "LR_SCHEDULER": LR_SCHEDULER,
            "BOX_MARGIN_PX": BOX_MARGIN_PX,
            "MIN_BOX_HEIGHT_PX": MIN_BOX_HEIGHT_PX,
        },
    }
    torch.save(checkpoint, path)


def log_metrics(writer, split: str, metrics: Dict[str, float], epoch: int) -> None:
    if writer is None:
        return
    for key, value in metrics.items():
        writer.add_scalar(f"{split}/{key}", value, epoch)


# ============================================================
# Main
# ============================================================


def main() -> None:
    set_seed(SPLIT_SEED)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = save_dir / BEST_CKPT_NAME
    last_ckpt_path = save_dir / "last_frustum_pointnet_kitti.pt"

    print(f"[INFO] Device: {device}")
    train_ids, val_ids = resolve_train_val_ids(save_dir)

    print("[INFO] Building KITTI datasets...")
    train_loader = make_loader(
        train_ids,
        split_name="train",
        shuffle=True,
        save_dir=save_dir,
    )
    val_loader = make_loader(
        val_ids,
        split_name="val",
        shuffle=False,
        save_dir=save_dir,
    )
    print(f"[INFO] Train batches: {len(train_loader)}")
    print(f"[INFO] Val batches  : {len(val_loader)}")

    model = FrustumPointNet(
        input_channels=3,
        num_seg_classes=2,
        num_object_points=NUM_OBJECT_POINTS,
        use_bn=False,
    ).to(device)

    if INIT_MODEL_CKPT is not None:
        init_ckpt = torch.load(INIT_MODEL_CKPT, map_location=device)
        state_dict = init_ckpt.get("model_state_dict", init_ckpt)
        model.load_state_dict(state_dict)
        print(f"[INFO] Loaded initial model weights: {INIT_MODEL_CKPT}")

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
            model, loss_fn, train_loader, optimizer, device
        )
        eval_metrics = eval_one_epoch(model, loss_fn, val_loader, device)

        if scheduler is not None:
            scheduler.step()

        epoch_time_s = time.perf_counter() - epoch_start
        current_lr = float(optimizer.param_groups[0]["lr"])
        train_metrics["lr"] = current_lr
        train_metrics["epoch_time_s"] = float(epoch_time_s)

        log_metrics(writer, "train", train_metrics, epoch)
        log_metrics(writer, "eval", eval_metrics, epoch)

        print(
            f"[Epoch {epoch:03d}/{NUM_EPOCHS}] "
            f"time={epoch_time_s:.1f}s lr={current_lr:.2e} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_seg={train_metrics['seg_loss']:.4f} "
            f"train_center={train_metrics['center_loss']:.4f} "
            f"train_box={train_metrics['box_loss']:.4f} | "
            f"val_loss={eval_metrics['loss']:.4f} "
            f"val_seg={eval_metrics['seg_loss']:.4f} "
            f"val_center={eval_metrics['center_loss']:.4f} "
            f"val_box={eval_metrics['box_loss']:.4f} "
            f"val_center_err_mean={eval_metrics['center_error_mean']:.4f} "
            f"val_center_err_med={eval_metrics['center_error_median']:.4f} "
            f"val_conf={eval_metrics['confidence_mean']:.4f}"
        )

        save_checkpoint(
            last_ckpt_path,
            model,
            optimizer,
            scheduler,
            epoch,
            best_eval_center_err,
            train_metrics,
            eval_metrics,
        )

        if eval_metrics["center_error_mean"] < best_eval_center_err:
            best_eval_center_err = eval_metrics["center_error_mean"]
            save_checkpoint(
                best_ckpt_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_eval_center_err,
                train_metrics,
                eval_metrics,
            )
            print(
                f"[INFO] Saved new best checkpoint: {best_ckpt_path} "
                f"(center error={best_eval_center_err:.6f})"
            )

    if writer is not None:
        writer.close()

    print("[DONE]")
    print(f"Best validation center error: {best_eval_center_err:.6f}")
    print(f"Best checkpoint: {best_ckpt_path}")


if __name__ == "__main__":
    main()
