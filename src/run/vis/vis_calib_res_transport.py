# visualize_e3_projection_residuals.py

from __future__ import annotations

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from task_network.data.task_unity_sim_dataloader import UnitySimDetectionDataset


# ============================================================
# Config
# ============================================================

DATA_ROOT = r"E:\data\unity_syn_data\expirments\baseline"
POINTCLOUD_SENSOR = "FOV-Grid_PointCloud"

SAMPLE_INDEX = 0
MAX_POINTS = 500
EDGE_THRESHOLD = 0.2


# ============================================================
# Helpers
# ============================================================


def prepare_rgb(rgb: torch.Tensor) -> np.ndarray:
    """
    CHW torch tensor -> uint8 HWC numpy
    """
    x = rgb.detach().cpu()

    if x.shape[0] == 3:
        x = x.permute(1, 2, 0)

    x = x.float().numpy()

    if x.max() <= 1.5:
        x = x * 255.0

    x = np.clip(x, 0, 255).astype(np.uint8)
    return x



def compute_edges(rgb_uint8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    mag = np.sqrt(gx * gx + gy * gy)

    if mag.max() > 0:
        mag = mag / mag.max()

    edge_mask = mag >= EDGE_THRESHOLD

    # thicken edges for visualization
    kernel = np.ones((3, 3), np.uint8)
    edge_mask = cv2.dilate(edge_mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    # distance to nearest edge
    dist = cv2.distanceTransform((~edge_mask).astype(np.uint8), cv2.DIST_L2, 5)

    return edge_mask, dist



def project_points(
    lidar_xyz: torch.Tensor,
    K: torch.Tensor,
    T_cam_lidar: torch.Tensor,
):
    """
    Unity camera convention:
        +X right
        +Y up
        +Z forward

    OpenCV pixels:
        +u right
        +v down

    therefore v uses flipped y.
    """

    pts = lidar_xyz.detach().cpu().float()

    if pts.shape[0] > MAX_POINTS:
        idx = torch.linspace(
            0,
            pts.shape[0] - 1,
            steps=MAX_POINTS,
        ).long()
        pts = pts[idx]

    n = pts.shape[0]

    ones = torch.ones((n, 1), dtype=torch.float32)
    pts_h = torch.cat([pts, ones], dim=1)

    cam_h = (T_cam_lidar.float() @ pts_h.T).T
    cam_xyz = cam_h[:, :3]

    x = cam_xyz[:, 0]
    y = -cam_xyz[:, 1]
    z = cam_xyz[:, 2]

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    z_safe = torch.clamp(z, min=1e-8)

    u = fx * x / z_safe + cx
    v = fy * y / z_safe + cy

    uv = torch.stack([u, v], dim=1)

    return uv, z


# ============================================================
# Load sample
# ============================================================

print("[INFO] Loading dataset...")

dataset = UnitySimDetectionDataset(
    root_dir=DATA_ROOT,
    pointcloud_sensor=POINTCLOUD_SENSOR,
)

image, lidar_xyz, target = dataset[SAMPLE_INDEX]

K = target["intrinsics"]
T_cam_lidar = target["T_cam_lidar"]

rgb_uint8 = prepare_rgb(image)
H, W = rgb_uint8.shape[:2]


# ============================================================
# Edge extraction
# ============================================================

print("[INFO] Computing edges...")

edge_mask, dist_map = compute_edges(rgb_uint8)


# ============================================================
# Project LiDAR points
# ============================================================

print("[INFO] Projecting LiDAR points...")

uv, depth = project_points(
    lidar_xyz=lidar_xyz,
    K=K,
    T_cam_lidar=T_cam_lidar,
)

u = uv[:, 0]
v = uv[:, 1]

valid = depth > 0
valid = valid & torch.isfinite(u) & torch.isfinite(v)
valid = valid & (u >= 0) & (u < W)
valid = valid & (v >= 0) & (v < H)

uv = uv[valid]

u_idx = uv[:, 0].round().long().numpy()
v_idx = uv[:, 1].round().long().numpy()

residuals = dist_map[v_idx, u_idx]


# ============================================================
# Visualization
# ============================================================

print("[INFO] Visualizing...")

fig, ax = plt.subplots(figsize=(12, 8))

ax.imshow(rgb_uint8)

# overlay edges
edge_overlay = np.zeros((H, W, 4), dtype=np.float32)
edge_overlay[..., 1] = edge_mask.astype(np.float32)  # green
edge_overlay[..., 3] = edge_mask.astype(np.float32) * 0.5

ax.imshow(edge_overlay)

# projected points
# more sensitive coloring using percentile clipping
vmax = np.percentile(residuals, 95)
vmin = np.percentile(residuals, 5)

sc = ax.scatter(
    uv[:, 0].numpy(),
    uv[:, 1].numpy(),
    c=residuals,
    cmap="jet",
    s=18,
    vmin=vmin,
    vmax=vmax,
)

cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Distance to nearest RGB edge [px]")

ax.set_title(
    f"E3 Projection Residual Visualization\n"
    f"Blue/red = larger residuals | green overlay = RGB edges"
)

ax.set_xlim([0, W])
ax.set_ylim([H, 0])
ax.set_aspect("equal")

plt.tight_layout()
plt.show()

