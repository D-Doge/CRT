from __future__ import annotations

import torch
import plotly.graph_objects as go

from task_network.frustrum_pointnet.unity_frustum_extractor import FrustumExtractor, transform_lidar_to_camera
from task_unity_sim_dataloader import UnitySimDetectionDataset


# ============================================================
# Config
# ============================================================

DATA_ROOT = r"E:\data\unity_syn_data\solo_6"
POINTCLOUD_SENSOR = "FOV-Grid_PointCloud"
SAMPLE_INDEX = 0
BOX_INDEX = 0

MAX_POINTS_TO_DRAW = 12000
FRUSTUM_DEPTH = 4.0          # meters, only for visualization
SIZE_INFLATION = 1.2         # only for visualized/label box if desired


# ============================================================
# Geometry helpers
# ============================================================
def remove_far_outliers(points: torch.Tensor, percentile: float = 99.5):
    """
    Remove very distant points based on Z/depth percentile.
    """
    z = points[:, 2]

    valid = torch.isfinite(z) & (z > 0)

    if valid.sum() == 0:
        return points

    z_valid = z[valid]

    threshold = torch.quantile(
        z_valid,
        percentile / 100.0,
    )

    keep = z <= threshold
    return points[keep]

def quat_xyzw_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    q = q.detach().float().flatten()
    qx, qy, qz, qw = q.tolist()

    n = max((qx*qx + qy*qy + qz*qz + qw*qw) ** 0.5, 1e-12)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n

    return torch.tensor([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,     1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw,     1 - 2*qx*qx - 2*qy*qy],
    ], dtype=torch.float32)


def compute_box_corners_cam(center_cam, size_xyz, quat_xyzw):
    center_cam = center_cam.detach().float().flatten()
    size_xyz = size_xyz.detach().float().flatten()
    quat_xyzw = quat_xyzw.detach().float().flatten()

    hx, hy, hz = (size_xyz / 2.0).tolist()

    corners_local = torch.tensor([
        [-hx, -hy, -hz],
        [ hx, -hy, -hz],
        [ hx,  hy, -hz],
        [-hx,  hy, -hz],
        [-hx, -hy,  hz],
        [ hx, -hy,  hz],
        [ hx,  hy,  hz],
        [-hx,  hy,  hz],
    ], dtype=torch.float32)

    R = quat_xyzw_to_rotmat(quat_xyzw)
    return corners_local @ R.T + center_cam[None, :]


def image_box_to_frustum_corners_cam(
    box_xyxy: torch.Tensor,
    K: torch.Tensor,
    depth: float,
) -> torch.Tensor:
    """
    Build a camera-frame frustum wireframe from a 2D box.

    Important:
    We use Unity camera coordinates:
        +X right, +Y up, +Z forward

    OpenCV pixel coordinates have +v down, so:
        x =  (u - cx) * z / fx
        y = -(v - cy) * z / fy
        z = depth
    """
    x1, y1, x2, y2 = box_xyxy.detach().float().tolist()

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    pixels = torch.tensor([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2],
    ], dtype=torch.float32)

    u = pixels[:, 0]
    v = pixels[:, 1]
    z = torch.full_like(u, float(depth))

    x = (u - cx) * z / fx
    y = -(v - cy) * z / fy

    far = torch.stack([x, y, z], dim=1)

    origin = torch.zeros((1, 3), dtype=torch.float32)
    return torch.cat([origin, far], dim=0)  # 0=origin, 1..4=far corners


def make_line_trace(points: torch.Tensor, edges, name: str, color: str, width: int = 5):
    p = points.detach().cpu().numpy()

    xs, ys, zs = [], [], []
    for i, j in edges:
        xs.extend([p[i, 0], p[j, 0], None])
        ys.extend([p[i, 1], p[j, 1], None])
        zs.extend([p[i, 2], p[j, 2], None])

    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line=dict(color=color, width=width),
        name=name,
    )


def make_box_trace(corners_cam: torch.Tensor, name: str, color: str):
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    return make_line_trace(corners_cam, edges, name=name, color=color, width=6)


def make_frustum_trace(frustum_corners_cam: torch.Tensor):
    edges = [
        (1, 2), (2, 3), (3, 4), (4, 1),  # far plane rectangle
        (0, 1), (0, 2), (0, 3), (0, 4),  # rays from camera origin
    ]
    return make_line_trace(
        frustum_corners_cam,
        edges,
        name="2D bbox frustum",
        color="orange",
        width=5,
    )


def make_axes_traces(origin, R=None, scale=0.25, name_prefix=""):
    """
    Draw local coordinate axes.
    """
    origin = torch.as_tensor(origin, dtype=torch.float32).flatten()

    if R is None:
        R = torch.eye(3, dtype=torch.float32)
    else:
        R = torch.as_tensor(R, dtype=torch.float32)

    axes = [
        ("X", "red",   R[:, 0]),
        ("Y", "green", R[:, 1]),
        ("Z", "blue",  R[:, 2]),
    ]

    traces = []
    for label, color, direction in axes:
        end = origin + scale * direction
        pts = torch.stack([origin, end], dim=0)
        traces.append(
            make_line_trace(
                pts,
                [(0, 1)],
                name=f"{name_prefix}{label}",
                color=color,
                width=8,
            )
        )
    return traces


def subsample_points(points: torch.Tensor, max_points: int) -> torch.Tensor:
    if points.shape[0] <= max_points:
        return points
    idx = torch.randperm(points.shape[0])[:max_points]
    return points[idx]


# ============================================================
# Load sample
# ============================================================

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

# Full point cloud in camera frame
pc_cam = transform_lidar_to_camera(lidar_xyz, T_cam_lidar)
pc_cam_filtered = remove_far_outliers(
    pc_cam,
    percentile=98,
)

pc_vis = subsample_points(
    pc_cam_filtered,
    MAX_POINTS_TO_DRAW,
)

# Frustum points from extractor
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

# Camera origin in camera coordinates
camera_origin_cam = torch.zeros(3)

# LiDAR origin in camera coordinates = T_cam_lidar @ [0,0,0,1]
lidar_origin_h = T_cam_lidar.float() @ torch.tensor([0.0, 0.0, 0.0, 1.0])
lidar_origin_cam = lidar_origin_h[:3]

# LiDAR axes in camera coordinates
R_cam_lidar = T_cam_lidar[:3, :3].float()

# 3D bbox corners
box_corners_cam = compute_box_corners_cam(center, size, quat)

# Frustum wireframe
frustum_corners_cam = image_box_to_frustum_corners_cam(
    box_xyxy=box_2d,
    K=K,
    depth=FRUSTUM_DEPTH,
)

print("2D box:", box_2d.tolist())
print("3D center cam:", center.tolist())
print("3D size used:", size.tolist())
print("3D quat xyzw:", quat.tolist())
print("Point cloud points:", pc_cam.shape[0])
print("Frustum points:", result.frustum_points_cam.shape[0])
print("Inside 3D bbox:", int(inside.sum().item()))
print("Camera origin cam:", camera_origin_cam.tolist())
print("LiDAR origin cam:", lidar_origin_cam.tolist())


# ============================================================
# Plotly interactive visualization
# ============================================================

fig = go.Figure()

# Full point cloud
pts = pc_vis.detach().cpu().numpy()
fig.add_trace(
    go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode="markers",
        marker=dict(size=2, color="gray", opacity=0.25),
        name="Full point cloud",
    )
)

# Frustum outside points
if (~inside).any():
    out = result.frustum_points_cam[~inside].detach().cpu().numpy()
    fig.add_trace(
        go.Scatter3d(
            x=out[:, 0],
            y=out[:, 1],
            z=out[:, 2],
            mode="markers",
            marker=dict(size=4, color="cyan", opacity=0.9),
            name="Frustum outside 3D bbox",
        )
    )

# Frustum inside points
if inside.any():
    inn = result.frustum_points_cam[inside].detach().cpu().numpy()
    fig.add_trace(
        go.Scatter3d(
            x=inn[:, 0],
            y=inn[:, 1],
            z=inn[:, 2],
            mode="markers",
            marker=dict(size=6, color="lime", opacity=1.0),
            name="Frustum inside 3D bbox",
        )
    )

# Camera origin
fig.add_trace(
    go.Scatter3d(
        x=[0.0], y=[0.0], z=[0.0],
        mode="markers+text",
        marker=dict(size=7, color="red"),
        text=["Camera"],
        textposition="top center",
        name="Camera origin",
    )
)

# LiDAR origin
lo = lidar_origin_cam.detach().cpu().numpy()
fig.add_trace(
    go.Scatter3d(
        x=[lo[0]], y=[lo[1]], z=[lo[2]],
        mode="markers+text",
        marker=dict(size=7, color="blue"),
        text=["LiDAR"],
        textposition="top center",
        name="LiDAR origin",
    )
)

# Camera axes
for tr in make_axes_traces(camera_origin_cam, R=torch.eye(3), scale=0.35, name_prefix="Camera "):
    fig.add_trace(tr)

# LiDAR axes
for tr in make_axes_traces(lidar_origin_cam, R=R_cam_lidar, scale=0.35, name_prefix="LiDAR "):
    fig.add_trace(tr)

# Frustum wireframe
fig.add_trace(make_frustum_trace(frustum_corners_cam))

# 3D bbox
fig.add_trace(make_box_trace(box_corners_cam, name="3D bounding box", color="lime"))

# 3D bbox center
c = center.detach().cpu().numpy()
fig.add_trace(
    go.Scatter3d(
        x=[c[0]],
        y=[c[1]],
        z=[c[2]],
        mode="markers+text",
        marker=dict(size=6, color="magenta"),
        text=["3D bbox center"],
        textposition="top center",
        name="3D bbox center",
    )
)

fig.update_layout(
    title=(
        f"Camera-space geometry | "
        f"frustum points={result.frustum_points_cam.shape[0]}, "
        f"inside bbox={int(inside.sum().item())}"
    ),
    scene=dict(
        xaxis_title="X_cam",
        yaxis_title="Y_cam",
        zaxis_title="Z_cam",
        aspectmode="data",
    ),
    margin=dict(l=0, r=0, b=0, t=45),
)

fig.show()