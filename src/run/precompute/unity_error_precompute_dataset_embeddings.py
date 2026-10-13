from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path

import torch

from data.dataset.unity_sim_dataset_loader import UnitySimDatasetLoader
from data.cache.cached_dataset_loader import CachedDatasetLoader

from data.extractor.rgb.dinov2_rgb_extractor import DINOv2RGBExtractor
from data.extractor.pointcloud.pointnet2_lidar_extractor import PointNet2LiDARExtractor
from data.extractor.joint.concat_joint_embedding_extractor import ConcatJointEmbeddingExtractor
from data.extractor.residual.unity_projection_residual_extractor import ProjectionResidualExtractor


ERROR_SPECS = [
    # name, tx_m, ty_m, tz_m, rx_deg, ry_deg, rz_deg
    ("etx+1", +0.01, 0, 0, 0, 0, 0),
    ("ety+1", 0, +0.01, 0, 0, 0, 0),
    ("etz+1", 0, 0, +0.01, 0, 0, 0),

    ("etx+2", +0.02, 0, 0, 0, 0, 0),
    ("ety+2", 0, +0.02, 0, 0, 0, 0),
    ("etz+2", 0, 0, +0.02, 0, 0, 0),

    ("etx+3", +0.03, 0, 0, 0, 0, 0),
    ("ety+3", 0, +0.03, 0, 0, 0, 0),
    ("etz+3", 0, 0, +0.03, 0, 0, 0),

    ("etx+4", +0.04, 0, 0, 0, 0, 0),
    ("ety+4", 0, +0.04, 0, 0, 0, 0),
    ("etz+4", 0, 0, +0.04, 0, 0, 0),

    ("etx+5", +0.05, 0, 0, 0, 0, 0),
    ("ety+5", 0, +0.05, 0, 0, 0, 0),
    ("etz+5", 0, 0, +0.05, 0, 0, 0),

]


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def rot_x(a: float) -> torch.Tensor:
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=torch.float32)


def rot_y(a: float) -> torch.Tensor:
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float32)


def rot_z(a: float) -> torch.Tensor:
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float32)


def make_error_transform(
    tx_m: float,
    ty_m: float,
    tz_m: float,
    rx_deg: float,
    ry_deg: float,
    rz_deg: float,
) -> torch.Tensor:
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)

    R = rot_z(rz) @ rot_y(ry) @ rot_x(rx)

    E = torch.eye(4, dtype=torch.float32)
    E[:3, :3] = R
    E[:3, 3] = torch.tensor([tx_m, ty_m, tz_m], dtype=torch.float32)
    return E


def inject_calibration_error(T_cam_lidar: torch.Tensor, E_cam: torch.Tensor) -> torch.Tensor:
    """
    T_cam_lidar maps LiDAR -> camera.
    E_cam is an artificial calibration error in the camera coordinate frame.

    Bad calibration:
        T_bad = E_cam @ T_true
    """
    return E_cam @ T_cam_lidar.detach().float()


class CalibrationErrorLoader:
    def __init__(self, base_loader, *, setup_id: str, E_cam: torch.Tensor):
        self.base_loader = base_loader
        self.setup_id = setup_id
        self.E_cam = E_cam

    def get_frame_ids(self):
        return self.base_loader.get_frame_ids()

    def load_sample(self, frame_id: str):
        sample = self.base_loader.load_sample(frame_id)

        if sample.T_cam_lidar is None:
            raise ValueError(f"Sample {frame_id} has no T_cam_lidar.")

        T_bad = inject_calibration_error(sample.T_cam_lidar, self.E_cam)

        meta = dict(sample.meta)
        meta["calibration_error_setup"] = self.setup_id
        meta["calibration_error_T_left_multiplied"] = True

        return replace(
            sample,
            setup_id=self.setup_id,
            T_cam_lidar=T_bad,
            meta=meta,
        )

    def __getattr__(self, name):
        return getattr(self.base_loader, name)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--pointcloud-sensor", default="FOV-Grid_PointCloud")
    parser.add_argument("--rgb-capture-id", default="rgb_camera")
    parser.add_argument("--lidar-capture-id", default="lidar")

    parser.add_argument("--cache-dir-name", default=".embedding_cache_calib_errors")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--device", default="auto")

    parser.add_argument("--lidar-num-points", type=int, default=2048)
    parser.add_argument("--lidar-sampling-mode", choices=["random", "fps", "none"], default="fps")

    parser.add_argument("--residual-edge-threshold", type=float, default=0.2)
    parser.add_argument("--residual-max-points", type=int, default=100)

    args = parser.parse_args()
    device = resolve_device(args.device)
    root_dir = Path(args.root_dir)

    base_loader = UnitySimDatasetLoader(
        root_dir=root_dir,
        pointcloud_sensor=args.pointcloud_sensor,
        setup_id="calib_error_base",
        rgb_capture_id=args.rgb_capture_id,
        lidar_capture_id=args.lidar_capture_id,
        use_identity_extrinsic_if_missing=False,
    )

    rgb_extractor = DINOv2RGBExtractor(device=device)

    lidar_extractor = PointNet2LiDARExtractor(
        num_points=None if args.lidar_sampling_mode == "none" else args.lidar_num_points,
        sampling_mode=args.lidar_sampling_mode,
        device=device,
    )

    joint_extractor = ConcatJointEmbeddingExtractor(l2_normalize_each=False)

    residual_extractor = ProjectionResidualExtractor(
        edge_threshold=args.residual_edge_threshold,
        max_points=args.residual_max_points,
    )

    extractors = [
        rgb_extractor,
        lidar_extractor,
        joint_extractor,
        residual_extractor,
    ]

    for name, tx, ty, tz, rx, ry, rz in ERROR_SPECS:
        print("\n" + "=" * 100)
        print(f"[INFO] Precomputing calibration-error setup: {name}")
        print(f"[INFO] tx={tx}m ty={ty}m tz={tz}m rx={rx}deg ry={ry}deg rz={rz}deg")
        print("=" * 100)

        E_cam = make_error_transform(tx, ty, tz, rx, ry, rz)

        error_loader = CalibrationErrorLoader(
            base_loader,
            setup_id=f"{args.lidar_capture_id}_{name}",
            E_cam=E_cam,
        )

        cached_loader = CachedDatasetLoader(
            error_loader,
            cache_root=root_dir / args.cache_dir_name,
            extractors=extractors,
            enabled=True,
            write_cache=True,
            refresh_cache=args.refresh_cache,
            drop_raw_after_populate=True,
        )

        frame_ids = cached_loader.get_frame_ids()
        ok = 0
        failed = 0

        print(f"[INFO] Frames: {len(frame_ids)}")
        print(f"[INFO] Cache root: {root_dir / args.cache_dir_name}")
        print(f"[INFO] Cache id: {cached_loader.cache_id}")

        for i, frame_id in enumerate(frame_ids, start=1):
            try:
                sample = cached_loader.load_sample(frame_id)
                sample.require("rgb_emb", "lidar_emb", "joint_emb", "residuals")
                ok += 1
            except Exception as e:
                failed += 1
                print(f"[ERROR] {name} frame={frame_id}: {type(e).__name__}: {e}")

            if i == 1 or i % 25 == 0 or i == len(frame_ids):
                print(f"[INFO] {name}: {i}/{len(frame_ids)} ok={ok} failed={failed}")

        print(f"[DONE] {name}: ok={ok}, failed={failed}")


if __name__ == "__main__":
    main()