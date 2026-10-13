# Calibration Residual Transport (CRT)

This repository contains the implementation and experimental code for **Calibration Residual Transport (CRT)**, an unlabeled, geometry-aware domain-gap metric for RGB–LiDAR sensor configurations.

CRT characterizes changes in the geometric relationship between RGB images and LiDAR measurements. LiDAR points are projected into the camera image, their distances to image-edge structure are computed, and the resulting projection-residual distributions are compared between sensor configurations using Sinkhorn divergence.

The repository also contains the data loading, experiment runners, visualization utilities, and Frustum PointNet-style task network used for the experiments accompanying the paper.

## Overview

Given two RGB–LiDAR sensor configurations \(A\) and \(B\), CRT follows the pipeline

```text
RGB image + LiDAR point cloud + calibration
                    |
                    v
          Project LiDAR into image
                    |
                    v
             Extract image edges
                    |
                    v
       Compute distance-to-edge residuals
                    |
                    v
      Aggregate residual distributions
                    |
                    v
            Sinkhorn divergence
                    |
                    v
                    CRT
```

CRT does not require target-domain task annotations or predictions from the downstream perception network.

The downstream task network included in this repository is used only to quantify the performance degradation associated with sensor-configuration changes.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
└── src
    ├── data
    │   ├── cache
    │   ├── dataset
    │   ├── extractor
    │   │   ├── joint
    │   │   ├── pointcloud
    │   │   ├── residual
    │   │   └── rgb
    │   └── data_sample_interface.py
    │
    ├── metric
    │   ├── calibration_residual_transport.py
    │   └── metric_interface.py
    │
    ├── run
    │   ├── eval
    │   ├── precompute
    │   ├── train
    │   └── vis
    │
    ├── task_network
    │   ├── data
    │   └── frustrum_pointnet
    │
    └── tests
        ├── extractor
        └── metric
```

The main CRT components are:

* `src/data/extractor/residual/unity_projection_residual_extractor.py`
  Computes RGB–LiDAR projection residuals.

* `src/metric/calibration_residual_transport.py`
  Computes the domain-gap score from the residual distributions.

* `src/run/eval/`
  Contains the evaluation scripts used for controlled sensor perturbations and real-world experiments.

The task network under `src/task_network/` is independent of the CRT computation and is included to reproduce the downstream perception evaluation.

## Installation

Clone the repository and create a Python environment:

```bash
git clone <repository-url>
cd domain_gap_metric

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Add the `src` directory to `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

A CUDA-capable PyTorch installation is recommended for feature extraction and task-network evaluation, but parts of the pipeline can also be executed on CPU.

## PointNet++ Dependency

The LiDAR feature extraction code uses parts of the following PointNet/PointNet++ PyTorch implementation:

[yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)

The required source files are included under

```text
src/data/extractor/pointcloud/Pointnet_Pointnet2_pytorch/
```

The pretrained PointNet++ SSG classification checkpoint is **not included** in this repository.

Place the corresponding checkpoint as

```text
src/data/extractor/pointcloud/
└── Pointnet_Pointnet2_pytorch/
    └── log/
        └── classification/
            └── pointnet2_ssg_wo_normals/
                └── checkpoints/
                    └── best_model.pth
```

The PointNet++ implementation is distributed under the MIT License. The original copyright and license notice must be retained with the redistributed source files.

PointNet++ embeddings are used by auxiliary preprocessing components and are not part of the CRT definition itself.

## DINOv2

RGB feature extraction utilities use DINOv2 through PyTorch Hub:

```python
torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
```

The model is downloaded automatically on first use.

These embeddings are used by auxiliary preprocessing components and are not required for the core CRT computation.

## Data

Datasets are not included directly in this repository.

The experiment code supports the Unity-based synthetic RGB–LiDAR datasets used for the controlled sensor-configuration experiments as well as the real-world RGB–LiDAR evaluation pipeline.

Dataset paths are provided to the corresponding experiment scripts at runtime.

The Unity loaders expect synchronized RGB images, LiDAR point clouds, calibration information, and the associated Unity frame metadata.

## Precomputation

Precomputation utilities are located under

```text
src/run/precompute/
```

The main scripts are

```text
unity_precompute_dataset_embeddings.py
unity_error_precompute_dataset_embeddings.py
```

They prepare reusable representations for repeated evaluations.

Run scripts from the repository root, for example:

```bash
python -m run.precompute.unity_precompute_dataset_embeddings --help
```

where supported, to inspect the available dataset and preprocessing arguments.

## CRT Evaluation

Experiment runners are located under

```text
src/run/eval/
```

and cover the main experiment families:

```text
run_calibration_error_task_and_metrics.py
run_dataset_sweep_task_and_metrics_multi_lidar.py
run_scan_pattern_dataset_sweep_task_and_metrics_multi_lidar.py
run_mobima_task_and_metrics.py
```

These scripts combine CRT computation with downstream task-network evaluation for the corresponding experiments.

For example:

```bash
python -m run.eval.run_calibration_error_task_and_metrics --help
```

The exact dataset paths, sensor names, checkpoints, sample counts, and output locations are configured through the command-line arguments of the respective experiment script.

## Task Network

The repository contains the Frustum PointNet-style perception model used to quantify downstream transfer degradation.

Its main components are located under

```text
src/task_network/frustrum_pointnet/
```

and include:

```text
frustum_pointnet_seg.py
center_refinement.py
box_head.py
frustum_pointnet_full_points.py
```

Dataset-specific frustum extraction is provided for both KITTI and the Unity datasets.

Training scripts are available under

```text
src/run/train/
```

including:

```text
train_frustum_pointnet_kitti.py
train_frustum_pointnet_points.py
```

The task network is **not part of CRT**. It is used only to measure the downstream performance changes against which CRT is evaluated.

## Visualization

Visualization utilities are located under

```text
src/run/vis/
```

including:

```text
vis_calib_res_transport.py
unity_visualize_prediction_3d.py
```

These scripts can be used to inspect RGB–LiDAR projection residuals and task-network predictions.

## Tests

Tests for the residual extractor, metric implementation, and supporting feature extractors are located under

```text
src/tests/
```

Run the test suite from the repository root with

```bash
pytest src/tests
```

after adding `src` to `PYTHONPATH`.

## Reproducibility

The experiment runners expose the relevant dataset, sampling, sensor, and task-network settings through their command-line interfaces.

For reproducing results from the paper, we recommend:

1. using the same dataset configuration and sensor perturbation;
2. using the corresponding trained task-network checkpoint;
3. using a fixed random seed;
4. keeping the CRT residual extraction and Sinkhorn parameters identical to those reported in the paper; and
5. using the same number of sampled frames and residuals.

The task-network results are only used for evaluating the relationship between CRT and downstream performance degradation. They are not inputs to CRT.

## Citation

If you use CRT or this implementation in your research, please cite the accompanying paper.

Citation information will be added after publication.

## License

Unless otherwise noted, the original code in this repository is released
under the MIT License. See the [`LICENSE`](LICENSE) file for details.

### Third-Party Software

This repository uses or contains components from external open-source
projects. These components remain subject to their respective original
licenses and are not relicensed by the MIT License of this repository.

#### PointNet++

Parts of the PointNet++ implementation are derived from:

> Yan et al., `yanx27/Pointnet_Pointnet2_pytorch`
> https://github.com/yanx27/Pointnet_Pointnet2_pytorch

The upstream project is distributed under the MIT License.

The corresponding source files are located under

```text
src/data/extractor/pointcloud/Pointnet_Pointnet2_pytorch/
```

The original PointNet++ copyright and license notice are retained with these
files. The pretrained PointNet++ checkpoint is not redistributed by this
repository and must be obtained separately.

#### DINOv2

RGB feature extraction utilities use DINOv2 from Meta AI:

> `facebookresearch/dinov2`
> https://github.com/facebookresearch/dinov2

DINOv2 is distributed under the Apache License 2.0. DINOv2 source code and
model weights are not redistributed in this repository; the model is loaded
from the upstream project through PyTorch Hub.

Other Python dependencies listed in `requirements.txt` remain subject to
their respective licenses.

