from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data.dataset.unity_sim_dataset_loader import UnitySimDatasetLoader
from data.cache.cached_dataset_loader import CachedDatasetLoader

from data.extractor.rgb.dinov2_rgb_extractor import DINOv2RGBExtractor
from data.extractor.pointcloud.pointnet2_lidar_extractor import PointNet2LiDARExtractor
from data.extractor.joint.concat_joint_embedding_extractor import ConcatJointEmbeddingExtractor
from data.extractor.residual.unity_projection_residual_extractor import ProjectionResidualExtractor

"""
python -m run.precompute.unity_precompute_dataset_embeddings \
  --root-dir /data_private/data/eval_multi_sensor \
  --setup-id baseline \
  --pointcloud-sensor FOV-Grid_PointCloud \
  --lidar-capture-id lidar \
  --device cuda \
  --lidar-num-points 2048 \
  --lidar-sampling-mode fps \
  --compute-residuals \
  --residual-max-points 100

  python -m run.precompute.unity_precompute_dataset_embeddings \
  --root-dir /data_private/data/eval_multi_sensor \
  --setup-id z10 \
  --pointcloud-sensor FOV-Grid_PointCloud \
  --lidar-capture-id lidar_z10 \
  --device cuda \
  --lidar-num-points 2048 \
  --lidar-sampling-mode fps \
  --compute-residuals \
  --residual-max-points 100
"""


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute and cache RGB, LiDAR, joint embeddings, and optionally "
            "projection residuals for one Unity RGB+LiDAR capture setup."
        )
    )

    parser.add_argument(
        "--root-dir",
        required=True,
        help="Unity dataset root, e.g. /data_private/data/eval_multi_sensor.",
    )

    parser.add_argument(
        "--pointcloud-sensor",
        default="FOV-Grid_PointCloud",
        help="Point cloud annotation/sensor filename identifier used by Unity, "
            "e.g. FOV-Grid_PointCloud.",
    )

    parser.add_argument(
        "--setup-id",
        default=None,
        help="Logical setup name stored in DataSample.setup_id. "
            "For multi-LiDAR sweeps, use names like baseline, z10, z15. "
            "If omitted, UnitySimDatasetLoader defaults to the pointcloud sensor.",
    )

    parser.add_argument(
        "--rgb-capture-id",
        default="rgb_camera",
        help="Unity capture id for the RGB camera.",
    )

    parser.add_argument(
        "--lidar-capture-id",
        default="lidar",
        help="Unity capture id for the LiDAR to precompute. "
            "For your sweep examples: lidar, lidar_z10, lidar_z15.",
    )

    parser.add_argument(
        "--cache-dir-name",
        default=".embedding_cache",
        help="Cache folder created inside --root-dir.",
    )

    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Recompute and overwrite existing cache files.",
    )

    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cuda, or cpu.",
    )

    parser.add_argument(
        "--lidar-num-points",
        type=int,
        default=2048,
        help="Number of LiDAR points used by PointNet++ embedding extractor. "
            "Should match eval script default unless intentionally changed.",
    )

    parser.add_argument(
        "--lidar-sampling-mode",
        choices=["random", "fps", "none"],
        default="fps",
        help="Point sampling for LiDAR embeddings. Must match eval script.",
    )

    parser.add_argument(
        "--compute-residuals",
        action="store_true",
        help="Also compute E3 projection residuals and cache them.",
    )

    parser.add_argument(
        "--residual-edge-threshold",
        type=float,
        default=0.2,
        help="Edge threshold for ProjectionResidualExtractor. Must match eval.",
    )

    parser.add_argument(
        "--residual-max-points",
        type=int,
        default=100,
        help="Maximum projected LiDAR points used for residual extraction. "
            "For your shown eval command this should be 100.",
    )

    args = parser.parse_args()
    device = resolve_device(args.device)

    root_dir = Path(args.root_dir)

    raw_loader = UnitySimDatasetLoader(
        root_dir=root_dir,
        pointcloud_sensor=args.pointcloud_sensor,
        setup_id=args.setup_id,
        rgb_capture_id=args.rgb_capture_id,
        lidar_capture_id=args.lidar_capture_id,
        use_identity_extrinsic_if_missing=args.compute_residuals,
    )

    rgb_extractor = DINOv2RGBExtractor(device=device)

    lidar_extractor = PointNet2LiDARExtractor(
        num_points=None if args.lidar_sampling_mode == "none" else args.lidar_num_points,
        sampling_mode=args.lidar_sampling_mode,
        device=device,
    )

    joint_extractor = ConcatJointEmbeddingExtractor(
        l2_normalize_each=False,
    )

    extractors = [
        rgb_extractor,
        lidar_extractor,
        joint_extractor,
    ]

    if args.compute_residuals:
        residual_extractor = ProjectionResidualExtractor(
           edge_threshold=args.residual_edge_threshold,
            max_points=args.residual_max_points,
        )
        extractors.append(residual_extractor)

    cached_loader = CachedDatasetLoader(
        raw_loader,
        cache_root=root_dir / args.cache_dir_name,
        extractors=extractors,
        enabled=True,
        write_cache=True,
        refresh_cache=args.refresh_cache,
        drop_raw_after_populate=True,
    )

    frame_ids = cached_loader.get_frame_ids()

    print(f"[INFO] Dataset: {root_dir}")
    print(f"[INFO] Frames: {len(frame_ids)}")
    print(f"[INFO] Cache root: {root_dir / args.cache_dir_name}")
    print(f"[INFO] Cache id: {cached_loader.cache_id}")
    print(f"[INFO] Device: {device}")

    ok = 0
    failed = 0

    for i, frame_id in enumerate(frame_ids, start=1):
        try:
            sample = cached_loader.load_sample(frame_id)
            sample.require("rgb_emb", "lidar_emb", "joint_emb")

            if args.compute_residuals and sample.residuals is None:
                print(f"[WARN] No residuals for frame {frame_id}")

            ok += 1

        except Exception as e:
            failed += 1
            print(f"[ERROR] Failed frame {frame_id}: {type(e).__name__}: {e}")

        if i == 1 or i % 25 == 0 or i == len(frame_ids):
            print(f"[INFO] Progress {i}/{len(frame_ids)} | ok={ok} failed={failed}")

    print("[DONE]")
    print(f"[DONE] ok={ok}, failed={failed}")


if __name__ == "__main__":
    main()