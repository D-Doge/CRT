from __future__ import annotations

import matplotlib.pyplot as plt
import torch

from task_network.frustrum_pointnet.unity_frustum_extractor import FrustumExtractor, project_camera_points
from task_unity_sim_dataloader import UnitySimDetectionDataset


DATA_ROOT = r"E:\data\unity_syn_data\solo_6"
POINTCLOUD_SENSOR = "FOV-Grid_PointCloud"
SAMPLE_INDEX = 0
BOX_INDEX = 0

SIZE_INFLATION = 1


def quat_xyzw_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    q = q.detach().float().flatten()
    qx, qy, qz, qw = q.tolist()

    n = max((qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5, 1e-12)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n

    return torch.tensor(
        [
            [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw,     1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw,     1 - 2*qx*qx - 2*qy*qy],
        ],
        dtype=torch.float32,
    )


def compute_box_corners_cam(center_cam, size_xyz, quat_xyzw):
    center_cam = center_cam.detach().float().flatten()
    size_xyz = size_xyz.detach().float().flatten()
    quat_xyzw = quat_xyzw.detach().float().flatten()

    hx, hy, hz = (size_xyz / 2.0).tolist()

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


def draw_box(ax, uv):
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    for i, j in edges:
        ax.plot(
            [uv[i, 0].item(), uv[j, 0].item()],
            [uv[i, 1].item(), uv[j, 1].item()],
            color="lime",
            linewidth=2,
        )


dataset = UnitySimDetectionDataset(
    root_dir=DATA_ROOT,
    pointcloud_sensor=POINTCLOUD_SENSOR,
)

image, lidar_xyz, target = dataset[SAMPLE_INDEX]

box_2d = target["boxes_2d_xyxy"][BOX_INDEX]
center = target["boxes_3d_center"][BOX_INDEX]
size = target["boxes_3d_size"][BOX_INDEX] * SIZE_INFLATION
quat = target["boxes_3d_quat"][BOX_INDEX]

K = target["intrinsics"]
T_cam_lidar = target["T_cam_lidar"]
image_size = tuple(target["image_size"].tolist())

extractor = FrustumExtractor(
    box_margin_px=2.0,
    min_depth=1e-4,
    num_sampled_points=None,
    sampling="none",
)

result = extractor.extract(
    lidar_xyz=lidar_xyz,
    box_2d_xyxy=box_2d,
    intrinsics=K,
    T_cam_lidar=T_cam_lidar,
    image_size=image_size,
    gt_box_center_cam=center,
    gt_box_size=size,
    gt_box_quat_xyzw=quat,
)

inside = result.gt_point_mask_in_frustum

print("2D box:", box_2d.tolist())
print("3D center cam:", center.tolist())
print("3D size used:", size.tolist())
print("3D quat xyzw:", quat.tolist())
print("Frustum points:", result.frustum_points_cam.shape[0])
print("Inside 3D bbox:", int(inside.sum().item()))
print("Outside 3D bbox:", int((~inside).sum().item()))


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

# Frustum points
uv_out = result.frustum_uv[~inside]
if uv_out.numel() > 0:
    ax.scatter(
        uv_out[:, 0],
        uv_out[:, 1],
        s=20,
        c="cyan",
        label="Frustum outside 3D bbox",
    )

uv_in = result.frustum_uv[inside]
if uv_in.numel() > 0:
    ax.scatter(
        uv_in[:, 0],
        uv_in[:, 1],
        s=35,
        c="lime",
        label="Frustum inside 3D bbox",
    )

# 3D bbox overlay using the same corrected projection as the extractor
corners = compute_box_corners_cam(center, size, quat)
corners_uv, corners_valid = project_camera_points(corners, K)

if bool(corners_valid.all().item()):
    draw_box(ax, corners_uv)

ax.set_title(f"Frustum points inside 3D bbox: {int(inside.sum())}/{len(inside)}")
ax.legend(loc="upper right")
ax.set_xlim(0, image_size[1])
ax.set_ylim(image_size[0], 0)
plt.tight_layout()
plt.show()