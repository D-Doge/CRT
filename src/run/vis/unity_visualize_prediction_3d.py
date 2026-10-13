from __future__ import annotations

import torch
import plotly.graph_objects as go

from task_network.data.task_unity_sim_dataloader import UnitySimDetectionDataset
from task_network.frustrum_pointnet.unity.unity_frustum_extractor import (
    FrustumExtractor,
    transform_lidar_to_camera,
)
from task_network.frustrum_pointnet.frustum_pointnet_full_points import FrustumPointNet


# ============================================================
# Config
# ============================================================

DATA_ROOT = r"E:\data\unity_syn_data\eval_dgm"
POINTCLOUD_SENSOR = "FOV-Grid_PointCloud"

CKPT_PATH = r"E:\domain_gap_metric\results\train\baseline_point_only\best_frustum_pointnet.pt"

SAMPLE_INDEX = 10
BOX_INDEX = 0

NUM_POINTS = 1024
NUM_OBJECT_POINTS = 512

MAX_POINTS_TO_DRAW = 12000
FRUSTUM_DEPTH = 4.0
SIZE_INFLATION = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Geometry helpers
# ============================================================

def remove_far_outliers(points: torch.Tensor, percentile: float = 99.0):
    z = points[:, 2]
    valid = torch.isfinite(z) & (z > 0)
    if valid.sum() == 0:
        return points

    threshold = torch.quantile(z[valid], percentile / 100.0)
    return points[z <= threshold]


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


def compute_box_corners(center, size_xyz, quat_xyzw):
    center = center.detach().float().flatten()
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
    return corners_local @ R.T + center[None, :]


def transform_box_lidar_to_cam(center_lidar, quat_lidar, T_cam_lidar):
    center_h = torch.cat([center_lidar.detach().float(), torch.ones(1)])
    center_cam = (T_cam_lidar.float() @ center_h)[:3]

    R_cam_lidar = T_cam_lidar[:3, :3].float()
    R_box_lidar = quat_xyzw_to_rotmat(quat_lidar)
    R_box_cam = R_cam_lidar @ R_box_lidar

    return center_cam, rotmat_to_quat_xyzw(R_box_cam)


def rotmat_to_quat_xyzw(R: torch.Tensor) -> torch.Tensor:
    R = R.detach().float()
    trace = R.trace()

    if trace > 0:
        s = torch.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s

    q = torch.tensor([qx, qy, qz, qw], dtype=torch.float32)
    return q / q.norm().clamp_min(1e-12)


def image_box_to_frustum_corners_cam(box_xyxy, K, depth):
    x1, y1, x2, y2 = box_xyxy.detach().float().tolist()

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

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
    return torch.cat([origin, far], dim=0)


def make_line_trace(points, edges, name, color, width=5):
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


def make_box_trace(corners, name, color):
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    return make_line_trace(corners, edges, name=name, color=color, width=6)


def make_frustum_trace(frustum_corners_cam):
    edges = [
        (1, 2), (2, 3), (3, 4), (4, 1),
        (0, 1), (0, 2), (0, 3), (0, 4),
    ]
    return make_line_trace(
        frustum_corners_cam,
        edges,
        name="2D bbox frustum",
        color="orange",
        width=5,
    )


def subsample_points(points, max_points):
    if points.shape[0] <= max_points:
        return points
    idx = torch.randperm(points.shape[0])[:max_points]
    return points[idx]


# ============================================================
# Load model
# ============================================================

model = FrustumPointNet(
    input_channels=3,
    num_seg_classes=2,
    num_object_points=NUM_OBJECT_POINTS,
    use_bn=False,
).to(DEVICE)

ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

print(f"[INFO] Loaded checkpoint: {CKPT_PATH}")
print(f"[INFO] Checkpoint epoch: {ckpt.get('epoch')}")
print(f"[INFO] Best eval center err: {ckpt.get('best_eval_center_err')}")


# ============================================================
# Load sample
# ============================================================

dataset = UnitySimDetectionDataset(
    root_dir=DATA_ROOT,
    pointcloud_sensor=POINTCLOUD_SENSOR,
)

image, lidar_xyz, target = dataset[SAMPLE_INDEX]

box_2d = target["boxes_2d_xyxy"][BOX_INDEX]
gt_center_cam = target["boxes_3d_center"][BOX_INDEX]
gt_size = target["boxes_3d_size"][BOX_INDEX] * SIZE_INFLATION
gt_quat_cam = target["boxes_3d_quat"][BOX_INDEX]

K = target["intrinsics"]
T_cam_lidar = target["T_cam_lidar"]
T_lidar_cam = torch.linalg.inv(T_cam_lidar.float())

image_size = tuple(target["image_size"].tolist())


# ============================================================
# Extract frustum
# ============================================================

extractor = FrustumExtractor(
    box_margin_px=2.0,
    min_depth=1e-4,
    num_sampled_points=NUM_POINTS,
    sampling="random",
    pad_repeat=True,
)

result = extractor.extract(
    lidar_xyz=lidar_xyz,
    box_2d_xyxy=box_2d,
    intrinsics=K,
    T_cam_lidar=T_cam_lidar,
    image_size=image_size,
    gt_box_center_cam=gt_center_cam,
    gt_box_size=gt_size,
    gt_box_quat_xyzw=gt_quat_cam,
)

points_lidar = result.sampled_points_lidar.float()
points_batch = points_lidar.unsqueeze(0).to(DEVICE)


# ============================================================
# Predict 3D box
# ============================================================

with torch.no_grad():
    pred = model(points_batch)

pred_center_lidar = pred["center_lidar"][0].detach().cpu()
pred_size = pred["size_lidar"][0].detach().cpu()
pred_quat_lidar = pred["quat_xyzw"][0].detach().cpu()
pred_conf = float(pred["confidence"][0].detach().cpu().item())

pred_center_cam, pred_quat_cam = transform_box_lidar_to_cam(
    pred_center_lidar,
    pred_quat_lidar,
    T_cam_lidar,
)

# GT center converted to LiDAR only for error check
gt_center_h = torch.cat([gt_center_cam.float(), torch.ones(1)])
gt_center_lidar = (T_lidar_cam @ gt_center_h)[:3]

center_err_lidar = torch.linalg.norm(pred_center_lidar - gt_center_lidar).item()
center_err_cam = torch.linalg.norm(pred_center_cam - gt_center_cam).item()

print("GT center cam:", gt_center_cam.tolist())
print("Pred center cam:", pred_center_cam.tolist())
print("Pred size:", pred_size.tolist())
print("Pred quat cam:", pred_quat_cam.tolist())
print("Pred confidence:", pred_conf)
print("Center error lidar:", center_err_lidar)
print("Center error cam:", center_err_cam)


# ============================================================
# Prepare visualization
# ============================================================

pc_cam = transform_lidar_to_camera(lidar_xyz, T_cam_lidar)
pc_cam_filtered = remove_far_outliers(pc_cam, percentile=98)
pc_vis = subsample_points(pc_cam_filtered, MAX_POINTS_TO_DRAW)

inside = result.gt_point_mask_in_frustum

gt_box_corners_cam = compute_box_corners(gt_center_cam, gt_size, gt_quat_cam)
pred_box_corners_cam = compute_box_corners(pred_center_cam, pred_size, pred_quat_cam)

frustum_corners_cam = image_box_to_frustum_corners_cam(
    box_xyxy=box_2d,
    K=K,
    depth=FRUSTUM_DEPTH,
)

camera_origin_cam = torch.zeros(3)
lidar_origin_h = T_cam_lidar.float() @ torch.tensor([0.0, 0.0, 0.0, 1.0])
lidar_origin_cam = lidar_origin_h[:3]


# ============================================================
# Plotly visualization
# ============================================================

fig = go.Figure()

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

# Use sampled points because inside/seg mask has shape NUM_POINTS
frustum_points_cam_for_vis = result.sampled_points_cam
inside_for_vis = result.gt_point_mask_in_frustum

if frustum_points_cam_for_vis is None:
    frustum_points_cam_for_vis = result.frustum_points_cam

if inside_for_vis is not None:
    inside_for_vis = inside_for_vis.bool()

    if (~inside_for_vis).any():
        out = frustum_points_cam_for_vis[~inside_for_vis].detach().cpu().numpy()
        fig.add_trace(
            go.Scatter3d(
                x=out[:, 0],
                y=out[:, 1],
                z=out[:, 2],
                mode="markers",
                marker=dict(size=4, color="cyan", opacity=0.9),
                name="Frustum outside GT box",
            )
        )

    if inside_for_vis.any():
        inn = frustum_points_cam_for_vis[inside_for_vis].detach().cpu().numpy()
        fig.add_trace(
            go.Scatter3d(
                x=inn[:, 0],
                y=inn[:, 1],
                z=inn[:, 2],
                mode="markers",
                marker=dict(size=6, color="lime", opacity=1.0),
                name="Frustum inside GT box",
            )
        )

fig.add_trace(make_frustum_trace(frustum_corners_cam))

fig.add_trace(make_box_trace(gt_box_corners_cam, name="GT 3D box", color="lime"))
fig.add_trace(make_box_trace(pred_box_corners_cam, name="Predicted 3D box", color="red"))

gc = gt_center_cam.detach().cpu().numpy()
fig.add_trace(
    go.Scatter3d(
        x=[gc[0]], y=[gc[1]], z=[gc[2]],
        mode="markers+text",
        marker=dict(size=6, color="lime"),
        text=["GT center"],
        textposition="top center",
        name="GT center",
    )
)

pc = pred_center_cam.detach().cpu().numpy()
fig.add_trace(
    go.Scatter3d(
        x=[pc[0]], y=[pc[1]], z=[pc[2]],
        mode="markers+text",
        marker=dict(size=6, color="red"),
        text=[f"Pred center conf={pred_conf:.2f}"],
        textposition="top center",
        name="Pred center",
    )
)

fig.update_layout(
    title=(
        f"GT vs predicted 3D box | "
        f"conf={pred_conf:.3f}, "
        f"center_err_cam={center_err_cam:.3f} m"
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