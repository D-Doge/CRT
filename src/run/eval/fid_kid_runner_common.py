"""Shared support for the metric-only runners."""
from __future__ import annotations
import csv
import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from metric.joint_distribution_metrics import JointFrechetFloat64, JointPolynomialMMD
from metric.rgb_reference_metrics import RGBReferenceConfig, compute_reference_rgb_metrics, rgb_uint8_chw


def sample_ids(ids, count, seed):
    ids = list(ids)  # preserve loader ordering, as in original runners
    if count <= 0 or count >= len(ids):
        return ids
    random.Random(seed).shuffle(ids)
    return ids[:count]


def stable_seed(*parts, base=0):
    value = '|'.join(str(x) for x in (base, *parts)).encode()
    return int.from_bytes(hashlib.sha1(value).digest()[:8], 'little') & 0x7fffffff


def resolve_device(device):
    if device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; set DEVICE="cpu" or "auto".')
    return device


class Extractors:
    """Lazy initialization: cached comparisons need no encoder downloads."""
    def __init__(self, device, num_points, sampling_mode):
        self.device, self.num_points, self.sampling_mode = device, num_points, sampling_mode
        self._rgb = self._lidar = None
        self.rgb_id = 'dinov2_vits14'
        self.lidar_id = f'pointnet2_ssg_{sampling_mode}_np{num_points}'

    def rgb(self, sample, seed):
        if self._rgb is None:
            from data.extractor.rgb.dinov2_rgb_extractor import DINOv2RGBExtractor
            self._rgb = DINOv2RGBExtractor(device=self.device)
        torch.manual_seed(seed)
        return self._rgb.extract(sample)

    def lidar(self, sample, seed):
        if self._lidar is None:
            from data.extractor.pointcloud.pointnet2_lidar_extractor import PointNet2LiDARExtractor
            self._lidar = PointNet2LiDARExtractor(num_points=self.num_points,
                sampling_mode=self.sampling_mode, device=self.device)
        torch.manual_seed(seed)
        return self._lidar.extract(sample)


def cached_fields(payload):
    """Accept direct DataSample or dictionaries containing embedding fields."""
    if hasattr(payload, 'rgb_emb') and hasattr(payload, 'lidar_emb'):
        return payload.rgb_emb, payload.lidar_emb
    if isinstance(payload, dict):
        if torch.is_tensor(payload.get('rgb_emb')) and torch.is_tensor(payload.get('lidar_emb')):
            return payload['rgb_emb'], payload['lidar_emb']
        for key in ('sample', 'data_sample', 'fields', 'data', 'payload', 'embeddings'):
            if key in payload:
                result = cached_fields(payload[key])
                if result is not None:
                    return result
    return None


def load_fields(path):
    if not path.is_file():
        return None
    # Only load trusted local project caches; old caches may contain DataSample.
    return cached_fields(torch.load(path, map_location='cpu', weights_only=False))


def attach_embeddings(sample, rgb_emb, lidar_emb):
    if not torch.is_tensor(rgb_emb) or not torch.is_tensor(lidar_emb):
        raise ValueError(f'Missing embeddings: {sample.setup_id}/{sample.frame_id}')
    rgb_emb = rgb_emb.detach().cpu().flatten().float()
    lidar_emb = lidar_emb.detach().cpu().flatten().float()
    joint = torch.cat((rgb_emb, lidar_emb))
    if not torch.isfinite(joint).all():
        raise ValueError(f'Nonfinite embedding: {sample.setup_id}/{sample.frame_id}')
    return replace(sample, rgb_emb=rgb_emb, lidar_emb=lidar_emb, joint_emb=joint,
                   lidar_xyz=None, lidar_feat=None, residuals=None)


def save_fields(path, sample):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'rgb_emb':sample.rgb_emb, 'lidar_emb':sample.lidar_emb}, path)


def rgb_digest(samples):
    h = hashlib.sha256()
    for s in samples:
        x = rgb_uint8_chw(s.rgb, '0_1')
        h.update(str(tuple(x.shape)).encode())
        h.update(x.numpy().tobytes())
    return h.hexdigest()


class Comparisons:
    def __init__(self, *, device, seed, subset_size, subsets):
        self.config = RGBReferenceConfig(input_range='0_1', cuda=device=='cuda',
            batch_size=1, seed=seed, kid_subset_size=subset_size, kid_subsets=subsets)
        self.joint = JointPolynomialMMD(subset_size=subset_size, subsets=subsets, seed=seed)
        self.rgb_cache = {}

    def evaluate(self, baseline, target, *, dataset, setup, baseline_name, capture_id=''):
        key = (rgb_digest(baseline), rgb_digest(target))
        if key not in self.rgb_cache:
            self.rgb_cache[key] = compute_reference_rgb_metrics(baseline, target, self.config)
        rgb = self.rgb_cache[key]
        fd = JointFrechetFloat64()(baseline, target)
        mmd = self.joint(baseline, target)
        return dict(dataset=dataset, setup=setup, baseline=baseline_name,
            lidar_capture_id=capture_id, n_baseline=len(baseline), n_target=len(target),
            RGB_FID=rgb['RGB-FID'].score, RGB_KID_mean=rgb['RGB-KID'].score,
            RGB_KID_std=rgb['RGB-KID'].details['std'], Joint_Frechet=fd.score,
            Joint_polynomial_MMD_mean=mmd.score,
            Joint_polynomial_MMD_std=mmd.details['std'],
            KID_subset_size=rgb['RGB-KID'].details['subset_size_used']), dict(
                baseline_frame_ids=[s.frame_id for s in baseline],
                target_frame_ids=[s.frame_id for s in target])


def write_results(rows, selections, config, output, name):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = output/f'{name}.csv', output/f'{name}.json'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    from importlib.metadata import version
    config = dict(config, torch_fidelity_version=version('torch-fidelity'))
    with json_path.open('w', encoding='utf-8') as f:
        json.dump(dict(config=config, results=rows, frames=selections),f,indent=2,allow_nan=False)
    print(f'[SAVED] {csv_path}\n[SAVED] {json_path}')


def discover_lidar_captures(root):
    """Read Unity Perception captures and annotation-level PCD filenames.

    Capture IDs are returned in sorted order. Frame sampling separately retains
    the original loader's ordering. Explicit ID lists override discovery.
    """
    found = set()
    files = sorted(Path(root).rglob('*.frame_data.json'))
    if not files:
        raise FileNotFoundError(f'No *.frame_data.json under {root}')
    for path in files:
        with path.open(encoding='utf-8-sig') as f:
            payload = json.load(f)
        for capture in payload.get('captures', []):
            def has_pcd(value):
                if isinstance(value,str): return value.lower().endswith('.pcd')
                if isinstance(value,dict): return any(has_pcd(v) for v in value.values())
                if isinstance(value,list): return any(has_pcd(v) for v in value)
                return False
            if has_pcd(capture):
                capture_id = capture.get('id')
                if isinstance(capture_id,str): found.add(capture_id)
    if not found:
        raise RuntimeError('No PCD capture IDs discovered. Set CAPTURE_IDS explicitly from metadata.')
    return sorted(found)
