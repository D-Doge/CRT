from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import torch


@dataclass(frozen=True)
class DataSample:
    """
    One synchronized RGB + LiDAR frame (or time window) from a sensor setup.

    Design rules (from spec):
    - This is the ONLY object metrics receive.
    - May include raw tensors, calibration tensors, and cached representations.
    - Metrics must only touch fields they declare in `required_fields`.
    - Any heavy computation (embeddings/residuals/logits) happens upstream
      in the DataLoader / precompute stage. :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}
    """

    # --- Identity / bookkeeping ---
    setup_id: str                 # e.g. "setup_A", "vlp32", "mid360", ...
    frame_id: str                 # unique within setup
    timestamp_s: Optional[float] = None
    sequence_id: Optional[str] = None

    # --- Raw sensor tensors (OPTIONAL) ---
    # Convention: keep tensors on CPU unless your pipeline decides otherwise.
    # RGB recommended layout: (3, H, W) float32 in [0,1] OR uint8 in [0,255] (choose one globally).
    rgb: Optional[torch.Tensor] = None              # (3,H,W) or (H,W,3)
    lidar_xyz: Optional[torch.Tensor] = None        # (N,3)
    lidar_feat: Optional[torch.Tensor] = None       # (N,F) e.g., intensity, ring, time

    # --- Calibration / geometry (OPTIONAL) ---
    intrinsics: Optional[torch.Tensor] = None       # (3,3) camera K
    T_cam_lidar: Optional[torch.Tensor] = None      # (4,4) lidar -> cam (or document your direction)
    image_size: Optional[Tuple[int, int]] = None    # (H,W) to avoid reading from rgb

    # --- Cached representations (OPTIONAL, produced upstream) ---
    # Embeddings
    rgb_emb: Optional[torch.Tensor] = None          # (D_r,)
    lidar_emb: Optional[torch.Tensor] = None        # (D_l,)
    joint_emb: Optional[torch.Tensor] = None        # (D_j,) e.g. concat(rgb_emb, lidar_emb)

    # Model outputs (for DoC)
    logits: Optional[torch.Tensor] = None           # (C,) or (C,...) depending on task
    conf: Optional[torch.Tensor] = None             # scalar or (C,) probability/confidence vector

    # Residual signals (for calibration residual transport)
    # e.g. distance-to-edge residuals, reprojection residuals, temporal offset residuals, ...
    residuals: Optional[torch.Tensor] = None        # (M,) or (M,K)

    # --- Extensible metadata ---
    # Use for things like weather, exposure, sensor settings, perturbation strength, etc.
    meta: Dict[str, Any] = field(default_factory=dict)

    # ---------------- Convenience helpers ----------------

    def require(self, *fields: str) -> None:
        """
        Light-weight, metric-runner-side validation helper.
        Raises ValueError if any required field is None.
        """
        missing = [f for f in fields if getattr(self, f, None) is None]
        if missing:
            raise ValueError(
                f"DataSample(setup_id={self.setup_id}, frame_id={self.frame_id}) "
                f"is missing required fields: {missing}"
            )

    def get_emb(self, kind: str) -> torch.Tensor:
        """
        Convenience accessor for common embedding choices.
        kind ∈ {"rgb", "lidar", "joint"}
        """
        if kind == "rgb":
            if self.rgb_emb is None:
                raise ValueError("rgb_emb is None")
            return self.rgb_emb
        if kind == "lidar":
            if self.lidar_emb is None:
                raise ValueError("lidar_emb is None")
            return self.lidar_emb
        if kind == "joint":
            if self.joint_emb is None:
                raise ValueError("joint_emb is None")
            return self.joint_emb
        raise ValueError(f"Unknown embedding kind: {kind}")