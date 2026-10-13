from __future__ import annotations

from typing import Union

import torch
import torchvision.transforms as T
from PIL import Image

from data.data_sample_interface import DataSample
from data.extractor.ExtractorInterface import DataSampleFieldExtractor


class DINOv2RGBExtractor(DataSampleFieldExtractor):
    """
    Extracts a DINOv2 RGB embedding from DataSample.rgb.

    Expected input:
    - sample.rgb is present
    - shape either (3, H, W) or (H, W, 3)
    - dtype can be uint8 in [0,255] or float tensor

    Output:
    - 1D torch.Tensor of shape (D,) stored as rgb_emb upstream
    """

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        hub_repo: str = "facebookresearch/dinov2",
        device: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._hub_repo = hub_repo
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = torch.hub.load(hub_repo, model_name)
        self.model.to(self._device)
        self.model.eval()

        self.transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

    @property
    def output_field(self) -> str:
        return "rgb_emb"

    @property
    def extractor_id(self) -> str:
        return self._model_name

    @torch.no_grad()
    def extract(self, sample: DataSample) -> torch.Tensor:
        if sample.rgb is None:
            raise ValueError(
                f"DINOv2RGBExtractor: sample.rgb is None for "
                f"setup_id={sample.setup_id}, frame_id={sample.frame_id}"
            )

        x = self._prepare_rgb_tensor(sample.rgb)
        x = self.transform(x).unsqueeze(0).to(self._device)  # [1, 3, 224, 224]

        feat = self.model(x)

        if not torch.is_tensor(feat):
            raise TypeError(
                f"DINOv2RGBExtractor: model output must be torch.Tensor, got {type(feat)}"
            )

        feat = feat.squeeze(0).detach().cpu().flatten()  # -> (D,)
        return feat

    def _prepare_rgb_tensor(self, rgb: Union[torch.Tensor, Image.Image]) -> torch.Tensor:
        """
        Convert input RGB to float tensor in shape (3, H, W), values in [0,1].
        """
        if isinstance(rgb, Image.Image):
            rgb = T.ToTensor()(rgb.convert("RGB"))
            return rgb

        if not torch.is_tensor(rgb):
            raise TypeError(
                f"DINOv2RGBExtractor: expected sample.rgb to be torch.Tensor or PIL.Image, got {type(rgb)}"
            )

        x = rgb.detach().cpu()

        # Accept HWC or CHW
        if x.ndim != 3:
            raise ValueError(
                f"DINOv2RGBExtractor: expected rgb with 3 dims, got shape {tuple(x.shape)}"
            )

        if x.shape[0] == 3:
            # already CHW
            pass
        elif x.shape[-1] == 3:
            # HWC -> CHW
            x = x.permute(2, 0, 1)
        else:
            raise ValueError(
                f"DINOv2RGBExtractor: expected RGB tensor with shape (3,H,W) or (H,W,3), got {tuple(x.shape)}"
            )

        # Convert to float in [0,1]
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        else:
            x = x.float()
            # Heuristic: if values look like [0,255], scale down
            if x.max() > 1.5:
                x = x / 255.0

        if x.shape[0] != 3:
            raise ValueError(
                f"DINOv2RGBExtractor: expected 3 channels after conversion, got shape {tuple(x.shape)}"
            )

        return x