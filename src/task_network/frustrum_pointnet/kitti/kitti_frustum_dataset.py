from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from task_network.data.kitti_detection_dataset import KittiDetectionDataset
from task_network.frustrum_pointnet.kitti.kitti_frustum_extractor import (
    KittiFrustumExtractor,
    points_in_2d_box_mask,
    points_in_oriented_box_lidar,
    project_lidar_to_image,
)


_INDEX_CACHE_VERSION = 1


class KittiFrustumDataset(Dataset):
    """
    Convert KITTI frames into one class-agnostic vehicle frustum per item.

    Returned tensors match FrustumPointNetLoss's existing contract:
        points_lidar:       (N,3)
        seg_label:          (N,), 0 background / 1 vehicle
        gt_center_lidar:    (3,)
        gt_size_lidar:      (3,), local [length,height,width]
        gt_quat_xyzw:       (4,), local-box -> LiDAR rotation
    """

    def __init__(
        self,
        base_dataset: KittiDetectionDataset,
        *,
        num_points: int = 1024,
        box_margin_px: float = 5.0,
        min_depth: float = 1e-4,
        min_box_height_px: float = 25.0,
        max_truncation: float = 1.0,
        max_occlusion: int = 3,
        min_frustum_points: int = 5,
        min_object_points: int = 1,
        validate_items: bool = True,
        valid_index_cache: Optional[str | Path] = None,
        deterministic_sampling: bool = False,
        sampling_seed: int = 0,
    ) -> None:
        if num_points <= 0:
            raise ValueError("num_points must be positive")
        if min_box_height_px < 0:
            raise ValueError("min_box_height_px must be non-negative")
        if min_frustum_points < 1 or min_object_points < 1:
            raise ValueError("Minimum point counts must be >= 1")

        self.base_dataset = base_dataset
        self.num_points = int(num_points)
        self.box_margin_px = float(box_margin_px)
        self.min_depth = float(min_depth)
        self.min_box_height_px = float(min_box_height_px)
        self.max_truncation = float(max_truncation)
        self.max_occlusion = int(max_occlusion)
        self.min_frustum_points = int(min_frustum_points)
        self.min_object_points = int(min_object_points)
        self.validate_items = bool(validate_items)
        self.valid_index_cache = (
            Path(valid_index_cache) if valid_index_cache is not None else None
        )
        self.deterministic_sampling = bool(deterministic_sampling)
        self.sampling_seed = int(sampling_seed)

        self.extractor = KittiFrustumExtractor(
            box_margin_px=self.box_margin_px,
            min_depth=self.min_depth,
            num_sampled_points=self.num_points,
            pad_repeat=True,
        )

        cached = self._load_cached_index()
        self.items = cached if cached is not None else self._build_index()
        if not self.items:
            raise RuntimeError("No valid KITTI vehicle frustums were found")
        if cached is None:
            self._save_cached_index()

    def _cache_config(self) -> Dict[str, object]:
        return {
            "version": _INDEX_CACHE_VERSION,
            "frame_ids": list(self.base_dataset.frame_ids),
            "classes": list(self.base_dataset.class_whitelist),
            "box_margin_px": self.box_margin_px,
            "min_depth": self.min_depth,
            "min_box_height_px": self.min_box_height_px,
            "max_truncation": self.max_truncation,
            "max_occlusion": self.max_occlusion,
            "min_frustum_points": self.min_frustum_points,
            "min_object_points": self.min_object_points,
            "validate_items": self.validate_items,
        }

    def _load_cached_index(self) -> Optional[List[Tuple[int, int]]]:
        path = self.valid_index_cache
        if path is None or not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[WARN] Could not read KITTI frustum cache {path}: {exc}")
            return None

        if payload.get("config") != self._cache_config():
            print(f"[INFO] Ignoring stale KITTI frustum cache: {path}")
            return None

        items = payload.get("items")
        if not isinstance(items, list):
            return None
        print(f"[INFO] Loaded {len(items)} valid KITTI frustums from {path}")
        return [(int(a), int(b)) for a, b in items]

    def _save_cached_index(self) -> None:
        path = self.valid_index_cache
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self._cache_config(),
            "items": [[a, b] for a, b in self.items],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[INFO] Wrote KITTI frustum cache: {path}")

    def _candidate_object_indices(self, target: Dict[str, object]) -> List[int]:
        boxes = target["boxes_2d_xyxy"]
        truncated = target["truncated"]
        occluded = target["occluded"]
        assert torch.is_tensor(boxes)
        assert torch.is_tensor(truncated)
        assert torch.is_tensor(occluded)

        out: List[int] = []
        for obj_idx in range(boxes.shape[0]):
            box_height = float((boxes[obj_idx, 3] - boxes[obj_idx, 1]).item())
            if box_height < self.min_box_height_px:
                continue
            if float(truncated[obj_idx].item()) > self.max_truncation:
                continue
            if int(occluded[obj_idx].item()) > self.max_occlusion:
                continue
            out.append(obj_idx)
        return out

    def _build_index(self) -> List[Tuple[int, int]]:
        items: List[Tuple[int, int]] = []
        total_candidates = 0
        skipped_low_frustum = 0
        skipped_low_object = 0

        for frame_idx in range(len(self.base_dataset)):
            if frame_idx % 250 == 0 or frame_idx + 1 == len(self.base_dataset):
                print(
                    f"[INFO] Building KITTI frustum index: "
                    f"{frame_idx + 1}/{len(self.base_dataset)}"
                )

            if not self.validate_items:
                target = self.base_dataset.get_target(frame_idx)
                candidates = self._candidate_object_indices(target)
                total_candidates += len(candidates)
                items.extend((frame_idx, obj_idx) for obj_idx in candidates)
                continue

            lidar_xyz, target = self.base_dataset.get_lidar_and_target(frame_idx)
            candidates = self._candidate_object_indices(target)
            total_candidates += len(candidates)
            if not candidates:
                continue

            P2 = target["P2"]
            T_rect_lidar = target["T_rect_lidar"]
            image_size_t = target["image_size"]
            assert torch.is_tensor(P2)
            assert torch.is_tensor(T_rect_lidar)
            assert torch.is_tensor(image_size_t)
            image_size = tuple(int(v) for v in image_size_t.tolist())

            _, uv, depth = project_lidar_to_image(lidar_xyz, P2, T_rect_lidar)
            boxes = target["boxes_2d_xyxy"]
            centers = target["boxes_3d_center_lidar"]
            sizes = target["boxes_3d_size_local_xyz"]
            quats = target["boxes_3d_quat_lidar_xyzw"]
            assert torch.is_tensor(boxes)
            assert torch.is_tensor(centers)
            assert torch.is_tensor(sizes)
            assert torch.is_tensor(quats)

            for obj_idx in candidates:
                frustum_mask = points_in_2d_box_mask(
                    uv,
                    depth,
                    boxes[obj_idx],
                    image_size=image_size,
                    min_depth=self.min_depth,
                    box_margin_px=self.box_margin_px,
                )
                frustum_points = lidar_xyz[frustum_mask]
                if frustum_points.shape[0] < self.min_frustum_points:
                    skipped_low_frustum += 1
                    continue

                object_mask = points_in_oriented_box_lidar(
                    frustum_points,
                    centers[obj_idx],
                    sizes[obj_idx],
                    quats[obj_idx],
                )
                if int(object_mask.sum().item()) < self.min_object_points:
                    skipped_low_object += 1
                    continue

                items.append((frame_idx, obj_idx))

        print(
            f"[INFO] KITTI frustum index: {len(items)}/{total_candidates} kept; "
            f"low-frustum={skipped_low_frustum}, low-object={skipped_low_object}"
        )
        return items

    def __len__(self) -> int:
        return len(self.items)

    def _sampling_generator(self, frame_idx: int, obj_idx: int) -> Optional[torch.Generator]:
        if not self.deterministic_sampling:
            return None
        seed = (
            self.sampling_seed
            + frame_idx * 1_000_003
            + obj_idx * 10_007
        ) & 0x7FFF_FFFF_FFFF_FFFF
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        return generator

    def __getitem__(self, idx: int) -> Dict[str, object]:
        frame_idx, obj_idx = self.items[idx]
        lidar_xyz, target = self.base_dataset.get_lidar_and_target(frame_idx)

        boxes = target["boxes_2d_xyxy"]
        centers = target["boxes_3d_center_lidar"]
        sizes = target["boxes_3d_size_local_xyz"]
        quats = target["boxes_3d_quat_lidar_xyzw"]
        P2 = target["P2"]
        T_rect_lidar = target["T_rect_lidar"]
        image_size_t = target["image_size"]
        assert torch.is_tensor(boxes)
        assert torch.is_tensor(centers)
        assert torch.is_tensor(sizes)
        assert torch.is_tensor(quats)
        assert torch.is_tensor(P2)
        assert torch.is_tensor(T_rect_lidar)
        assert torch.is_tensor(image_size_t)
        image_size = tuple(int(v) for v in image_size_t.tolist())

        result = self.extractor.extract(
            lidar_xyz=lidar_xyz,
            box_2d_xyxy=boxes[obj_idx],
            P2=P2,
            T_rect_lidar=T_rect_lidar,
            image_size=image_size,
            gt_box_center_lidar=centers[obj_idx],
            gt_box_size_local_xyz=sizes[obj_idx],
            gt_box_quat_xyzw_lidar=quats[obj_idx],
            generator=self._sampling_generator(frame_idx, obj_idx),
        )

        if result.sampled_points_lidar is None:
            raise RuntimeError("KittiFrustumExtractor did not produce sampled points")
        if result.gt_point_mask_in_frustum is None:
            raise RuntimeError("KittiFrustumExtractor did not produce GT point labels")

        class_names = target["label_names"]
        assert isinstance(class_names, list)

        return {
            "points_lidar": result.sampled_points_lidar.float(),
            "seg_label": result.gt_point_mask_in_frustum.long(),
            "gt_center_lidar": centers[obj_idx].float(),
            "gt_size_lidar": sizes[obj_idx].float(),
            "gt_quat_xyzw": quats[obj_idx].float(),
            "frame_idx": torch.tensor(frame_idx, dtype=torch.long),
            "obj_idx": torch.tensor(obj_idx, dtype=torch.long),
            "frame_id": str(target["frame_id"]),
            "class_name": str(class_names[obj_idx]),
            "num_frustum_points": torch.tensor(
                result.frustum_points_lidar.shape[0], dtype=torch.long
            ),
            "num_object_points_before_sampling": torch.tensor(
                int(
                    points_in_oriented_box_lidar(
                        result.frustum_points_lidar,
                        centers[obj_idx],
                        sizes[obj_idx],
                        quats[obj_idx],
                    ).sum().item()
                ),
                dtype=torch.long,
            ),
        }


def kitti_frustum_collate(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    tensor_keys = (
        "points_lidar",
        "seg_label",
        "gt_center_lidar",
        "gt_size_lidar",
        "gt_quat_xyzw",
        "frame_idx",
        "obj_idx",
        "num_frustum_points",
        "num_object_points_before_sampling",
    )
    out: Dict[str, object] = {
        key: torch.stack([item[key] for item in batch], dim=0)  # type: ignore[arg-type]
        for key in tensor_keys
    }
    out["frame_id"] = [str(item["frame_id"]) for item in batch]
    out["class_name"] = [str(item["class_name"]) for item in batch]
    return out
