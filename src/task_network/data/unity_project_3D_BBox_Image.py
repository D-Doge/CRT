from __future__ import annotations

import matplotlib.pyplot as plt
import torch

from task_unity_sim_dataloader import UnitySimDetectionDataset


# ============================================================
# Config
# ============================================================

DATA_ROOT = r"E:\data\unity_syn_data\solo_6"
POINTCLOUD_SENSOR = "FOV-Grid_PointCloud"
SAMPLE_INDEX = 0
BOX_INDEX = 0


# ============================================================
# Helpers
# ============================================================

def quat_xyzw_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    q = quat.detach().float().flatten()
    qx, qy, qz, qw = q.tolist()

    n = max((qx*qx + qy*qy + qz*qz + qw*qw) ** 0.5, 1e-12)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n

    return torch.tensor(
        [
            [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw,     1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw,     1 - 2*qx*qx - 2*qy*qy],
        ],
        dtype=torch.float32,
    )


def compute_box_corners_cam(center_cam, size_xyz, quat_xyzw) -> torch.Tensor:
    center_cam = center_cam.detach().float().flatten()
    size_xyz = size_xyz.detach().float().flatten()
    quat_xyzw = quat_xyzw.detach().float().flatten()

    sx, sy, sz = size_xyz.tolist()
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0

    corners_local = torch.tensor(
        [
            [-hx, -hy, -hz],
            [ hx, -hy, -hz],
            [ hx,  hy, -hz],
            [-hx,  hy, -hz],
            [-hx, -hy,  hz],
            [ hx, -hy,  hz],
            [ hx,  hy,  hz],
            [-hx,  hy,  hz],
        ],
        dtype=torch.float32,
    )

    R = quat_xyzw_to_rotmat(quat_xyzw)
    return corners_local @ R.T + center_cam[None, :]


def project_unity_style(points_cam: torch.Tensor, projection_matrix: torch.Tensor, image_size: tuple[int, int]):
    points_cam = points_cam.detach().float()
    P = projection_matrix.detach().float()
    H, W = image_size

    projected = (P @ points_cam.T).T  # (N,3)

    denom = projected[:, 2]
    valid = torch.abs(denom) > 1e-8

    safe_denom = torch.where(
        torch.abs(denom) > 1e-8,
        denom,
        torch.full_like(denom, 1e-8),
    )

    ndc_x = projected[:, 0] / safe_denom
    ndc_y = projected[:, 1] / safe_denom

    u = -(ndc_x * W) / 2.0 + W / 2.0
    v =  (ndc_y * H) / 2.0 + H / 2.0

    uv = torch.stack([u, v], dim=1)
    return uv, valid


def draw_projected_3d_box(ax, corners_uv: torch.Tensor, color="lime", linewidth=2.0):
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    uv = corners_uv.detach().cpu().numpy()

    for i, j in edges:
        ax.plot(
            [uv[i, 0], uv[j, 0]],
            [uv[i, 1], uv[j, 1]],
            color=color,
            linewidth=linewidth,
        )


# ============================================================
# Load sample
# ============================================================

dataset = UnitySimDetectionDataset(
    root_dir=DATA_ROOT,
    pointcloud_sensor=POINTCLOUD_SENSOR,
)

image, lidar_xyz, target = dataset[SAMPLE_INDEX]

box_2d = target["boxes_2d_xyxy"][BOX_INDEX]
center_3d = target["boxes_3d_center"][BOX_INDEX]
size_3d = target["boxes_3d_size"][BOX_INDEX]
quat_3d = target["boxes_3d_quat"][BOX_INDEX]
P_unity = target["unity_projection_matrix"]
image_size = tuple(target["image_size"].tolist())

print("2D box:", box_2d.tolist())
print("3D center:", center_3d.tolist())
print("3D size:", size_3d.tolist())
print("3D quat:", quat_3d.tolist())
print("Unity projection matrix:\n", P_unity)

corners_cam = compute_box_corners_cam(center_3d, size_3d, quat_3d)
corners_uv, corners_valid = project_unity_style(corners_cam, P_unity, image_size)

center_uv, center_valid = project_unity_style(center_3d[None, :], P_unity, image_size)

print("Projected center uv:", center_uv[0].tolist())
print("All corners valid:", bool(corners_valid.all().item()))


# ============================================================
# Visualization
# ============================================================

img_np = image.permute(1, 2, 0).cpu().numpy()

fig, ax = plt.subplots(figsize=(12, 9))
ax.imshow(img_np)

# 2D GT box
x1, y1, x2, y2 = box_2d.tolist()
ax.plot(
    [x1, x2, x2, x1, x1],
    [y1, y1, y2, y2, y1],
    color="red",
    linewidth=2,
    label="2D GT box",
)

# 3D projected box
if bool(corners_valid.all().item()):
    draw_projected_3d_box(ax, corners_uv, color="lime", linewidth=2.0)
else:
    valid_uv = corners_uv[corners_valid]
    if valid_uv.numel() > 0:
        ax.scatter(
            valid_uv[:, 0].cpu().numpy(),
            valid_uv[:, 1].cpu().numpy(),
            s=50,
            c="lime",
            label="Valid 3D corners",
        )

# projected 3D center
if bool(center_valid[0].item()):
    ax.scatter(
        [center_uv[0, 0].item()],
        [center_uv[0, 1].item()],
        s=90,
        c="magenta",
        marker="x",
        label="Projected 3D center",
    )

ax.set_title("Unity-style projection matrix: 3D bbox projected into RGB image")
ax.legend(loc="upper right")
ax.set_xlim(0, image_size[1])
ax.set_ylim(image_size[0], 0)
plt.tight_layout()
plt.show()