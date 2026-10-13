from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from task_network.data.task_unity_sim_dataloader import UnitySimDetectionDataset
from task_network.frustrum_pointnet.unity.unity_frustum_extractor import FrustumExtractor

def transform_camera_center_to_lidar(center_cam: torch.Tensor, T_cam_lidar: torch.Tensor) -> torch.Tensor:
    T_lidar_cam = torch.linalg.inv(T_cam_lidar.float())
    center_h = torch.cat([center_cam.float(), torch.ones(1)])
    center_lidar_h = T_lidar_cam @ center_h
    return center_lidar_h[:3]


class UnityFrustumSegDataset(Dataset):
    """
    Converts UnitySimDetectionDataset frames into object-frustum segmentation samples.

    Output per item:
        points_lidar: (N,3)
        seg_label:    (N,)  0 background, 1 object
        meta: dict
    """

    def __init__(
        self,
        base_dataset: UnitySimDetectionDataset,
        *,
        num_points: int = 1024,
        box_margin_px: float = 5.0,
        skip_empty_frustums: bool = True,
    ):
        self.base_dataset = base_dataset
        self.frustum_extractor = FrustumExtractor(
            box_margin_px=box_margin_px,
            num_sampled_points=num_points,
            sampling="random",
            pad_repeat=True,
        )
        self.skip_empty_frustums = skip_empty_frustums

        self.items: List[Tuple[int, int]] = self._build_index()

    def _build_index(self) -> List[Tuple[int, int]]:
        items = []

        for frame_idx in range(len(self.base_dataset)):
            _, _, target = self.base_dataset[frame_idx]

            boxes_2d = target["boxes_2d_xyxy"]
            has_3d = target["has_3d_box"]

            for obj_idx in range(boxes_2d.shape[0]):
                if bool(has_3d[obj_idx]):
                    items.append((frame_idx, obj_idx))

        return items

    def __len__(self) -> int:
        return len(self.items)
    

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        frame_idx, obj_idx = self.items[idx]

        image, lidar_xyz, target = self.base_dataset[frame_idx]

        image_size = target["image_size"]
        if torch.is_tensor(image_size):
            image_size = tuple(int(v) for v in image_size.tolist())

        result = self.frustum_extractor.extract(
            lidar_xyz=lidar_xyz,
            box_2d_xyxy=target["boxes_2d_xyxy"][obj_idx],
            intrinsics=target["intrinsics"],
            T_cam_lidar=target["T_cam_lidar"],
            image_size=image_size,
            gt_box_center_cam=target["boxes_3d_center"][obj_idx],
            gt_box_size=target["boxes_3d_size"][obj_idx],
            gt_box_quat_xyzw=target["boxes_3d_quat"][obj_idx],
        )

        if result.sampled_points_lidar is None:
            points = result.frustum_points_lidar
        else:
            points = result.sampled_points_lidar

        if result.gt_point_mask_in_frustum is None:
            raise ValueError("No GT point mask was produced. Check 3D box annotations.")

        seg_label = result.gt_point_mask_in_frustum.long()


        gt_center_cam = target["boxes_3d_center"][obj_idx]
        gt_center_lidar = transform_camera_center_to_lidar(
            gt_center_cam,
            target["T_cam_lidar"],
            )
        
        gt_size_lidar = target["boxes_3d_size"][obj_idx].float()
        gt_quat_xyzw = target["boxes_3d_quat"][obj_idx].float()
        
        return {
            "points_lidar": points.float(),       # (N,3), LiDAR frame
            "seg_label": seg_label.long(),        # (N,)
            "frame_idx": torch.tensor(frame_idx),
            "obj_idx": torch.tensor(obj_idx),
            "gt_center_lidar": gt_center_lidar.float(),
            "gt_size_lidar": gt_size_lidar,
            "gt_quat_xyzw": gt_quat_xyzw,
        }


def frustum_seg_collate(batch):
    return {
        "points_lidar": torch.stack([b["points_lidar"] for b in batch], dim=0),
        "seg_label": torch.stack([b["seg_label"] for b in batch], dim=0),
        "gt_center_lidar": torch.stack([b["gt_center_lidar"] for b in batch], dim=0),
        "gt_size_lidar": torch.stack([b["gt_size_lidar"] for b in batch], dim=0),
        "gt_quat_xyzw": torch.stack([b["gt_quat_xyzw"] for b in batch], dim=0),
        "frame_idx": torch.stack([b["frame_idx"] for b in batch], dim=0),
        "obj_idx": torch.stack([b["obj_idx"] for b in batch], dim=0),
    }