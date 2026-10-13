"""Run only RGB FID/KID and joint distribution metrics on Unity datasets.
Run from project root: PYTHONPATH=src python -m run.eval.run_unity_fid_kid
"""
from pathlib import Path

# ========================== EDIT CONFIGURATION HERE ==========================
BIG_SWEEP_ROOT = Path(r'E:\data\unity_syn_data\expirments\multi_sensors\big_sweep')
SCAN_PATTERN_ROOT = Path(r'E:\data\unity_syn_data\expirments\multi_sensors\scan_patterns')
OUTPUT_DIR = Path(r'E:\domain_gap_metric\results\KID_FID_Results')
# True splits existing result JSON files without loading data or running metrics.
SPLIT_EXISTING_RESULTS_ONLY = False
DEVICE = 'cuda'
SEED = 0
N_METRIC_FRAMES = 100
POINTCLOUD_SENSOR = 'FOV-Grid_PointCloud'
BASELINE_CAPTURE_ID = 'lidar'
LIDAR_NUM_POINTS = 2048
LIDAR_SAMPLING_MODE = 'fps'
KID_SUBSET_SIZE = 1000
KID_SUBSETS = 100
RUN_BIG_SWEEP = False
RUN_SCAN_PATTERNS = True
RUN_CALIBRATION_ERRORS = False
# None discovers all PCD capture IDs. Set explicit tuples to restrict the run.
BIG_SWEEP_CAPTURE_IDS = None
SCAN_PATTERN_CAPTURE_IDS = None
# Exact annotation IDs from scan_patterns/step0.frame_data.json.
SCAN_PATTERN_SENSORS = (
    ('baseline', 'FOV-Grid_PointCloud'),
    ('fov_random', 'FOV-Radom_PointCloud'),  # Dataset spelling is "Radom".
    ('scanline', 'ScanLine_PointCloud'),
    ('sinus', 'Sinus_PointCloud'),
    ('mid360', 'MID360_PointCloud'),
    ('mid70', 'MID70_PointCloud'),
)
CALIBRATION_AXES = ('x', 'y', 'z')
CALIBRATION_TRANSLATIONS_CM = (1, 2, 3, 4, 5)
USE_EXISTING_EMBEDDING_CACHE = True
EXISTING_CACHE_DIRS = ('.embedding_cache',)
# New cache only: old project caches and result files are never overwritten.
NEW_CACHE_DIR = '.embedding_cache_fid_kid_v1'
# =============================================================================

import torch
import re
import json
from run.eval.fid_kid_runner_common import (Extractors, Comparisons,
    resolve_device, sample_ids, stable_seed, attach_embeddings, load_fields,
    save_fields, write_results, discover_lidar_captures)


def setup_name(capture_id):
    if capture_id == BASELINE_CAPTURE_ID:
        return 'baseline'
    return capture_id[len('lidar_'):] if capture_id.startswith('lidar_') else capture_id


def existing_cache_index(root, capture_id, setup, pointcloud_sensor=POINTCLOUD_SENSOR):
    paths = {}
    if not USE_EXISTING_EMBEDDING_CACHE:
        return paths
    marker = f'__lidar_{capture_id}__'
    for dirname in EXISTING_CACHE_DIRS:
        parent = root / dirname
        if not parent.exists(): continue
        for namespace in parent.iterdir():
            if not namespace.is_dir(): continue
            if marker not in namespace.name: continue
            if f'pc_{pointcloud_sensor}__' not in namespace.name: continue
            if 'rgb_emb_dinov2_vits14' not in namespace.name: continue
            if f'lidar_emb_pointnet2_ssg_{LIDAR_SAMPLING_MODE}_np{LIDAR_NUM_POINTS}' not in namespace.name: continue
            # Existing baseline may be called lidar rather than baseline.
            setups = (setup, capture_id, pointcloud_sensor) if setup == 'baseline' else (setup, pointcloud_sensor)
            for setup_id in setups:
                directory = namespace/setup_id
                if directory.is_dir():
                    for p in directory.rglob('*.pt'):
                        paths.setdefault(p.stem, []).append(p)
    return paths


def load_samples(root, capture_id, extractors, *, pointcloud_sensor=POINTCLOUD_SENSOR, setup=None):
    from data.dataset.unity_sim_dataset_loader import UnitySimDatasetLoader
    setup = setup or setup_name(capture_id)
    loader = UnitySimDatasetLoader(root_dir=root, pointcloud_sensor=pointcloud_sensor,
        lidar_capture_id=capture_id, setup_id=setup,
        use_identity_extrinsic_if_missing=True)
    ids = sample_ids(loader.get_frame_ids(), N_METRIC_FRAMES, SEED)
    index = existing_cache_index(root,capture_id,setup,pointcloud_sensor)
    samples = []
    counts = dict(existing_cache=0, new_cache=0, recomputed=0)
    for i,frame_id in enumerate(ids,1):
        raw = loader.load_sample(frame_id)
        cache_capture = capture_id if pointcloud_sensor == POINTCLOUD_SENSOR else f'{capture_id}__pc_{pointcloud_sensor}'
        new_path = root/NEW_CACHE_DIR/f'{extractors.rgb_id}__{extractors.lidar_id}__seed{SEED}'/cache_capture/(str(frame_id)+'.pt')
        fields = None
        matches = list(dict.fromkeys(index.get(str(frame_id), [])))
        if len(matches) == 1:
            fields = load_fields(matches[0])
            if fields is not None: counts['existing_cache'] += 1
        if fields is None:
            fields = load_fields(new_path)
            if fields is not None: counts['new_cache'] += 1
        if fields is None:
            # Recomputed features are reproducible; original FPS RNG histories
            # cannot be reconstructed if the corresponding old cache is absent.
            rgb = extractors.rgb(raw, stable_seed('rgb',frame_id,base=SEED))
            lidar = extractors.lidar(raw, stable_seed('lidar_emb',capture_id,pointcloud_sensor,frame_id,base=SEED))
            fields = (rgb,lidar)
            counts['recomputed'] += 1
        try:
            sample = attach_embeddings(raw,*fields)
        except ValueError as exc:
            if not str(exc).startswith('Nonfinite embedding:'):
                raise
            counts['skipped_nonfinite'] = counts.get('skipped_nonfinite', 0) + 1
            print(f'[WARN] Skipping {pointcloud_sensor}: {exc}')
            continue
        if not new_path.is_file(): save_fields(new_path,sample)
        samples.append(sample)
        if i==1 or i%25==0 or i==len(ids):
            print(f'[UNITY] {setup}: {i}/{len(ids)}')
    if len(samples)<2: raise RuntimeError(f'Fewer than 2 samples for {capture_id}')
    print(f'[FEATURES] {setup}: {counts}')
    return samples, counts


def config_summary():
    return dict(big_sweep_root=str(BIG_SWEEP_ROOT),scan_pattern_root=str(SCAN_PATTERN_ROOT),
        seed=SEED,n_metric_frames=N_METRIC_FRAMES,pointcloud_sensor=POINTCLOUD_SENSOR,
        baseline_capture_id=BASELINE_CAPTURE_ID,lidar_num_points=LIDAR_NUM_POINTS,
        lidar_sampling_mode=LIDAR_SAMPLING_MODE,joint_normalization=False,
        kid_subsets=KID_SUBSETS,kid_subset_size_requested=KID_SUBSET_SIZE,
        calibration_axes=list(CALIBRATION_AXES),
        calibration_translations_cm=list(CALIBRATION_TRANSLATIONS_CM),
        use_existing_embedding_cache=USE_EXISTING_EMBEDDING_CACHE,
        existing_cache_dirs=list(EXISTING_CACHE_DIRS),new_cache_dir=NEW_CACHE_DIR,
        rgb_input_range='0_1',device=DEVICE)


def change_group(setup):
    """Group pure changes; mixed perturbations remain a separate series."""
    match = re.fullmatch(r'([tr])([xyz])([+-]?)(?:_)?(\d+(?:p\d+|\.\d+)?)', setup)
    if match:
        kind, axis, sign, amount = match.groups()
        direction = 'minus' if sign == '-' else 'plus'
        kind = 'translation' if kind == 't' else 'rotation'
        return f'{kind}_{axis}_{direction}', float(amount.replace('p', '.'))
    match = re.fullmatch(r'et([xyz])_(\d+(?:p\d+)?)', setup)
    if match:
        axis, amount = match.groups()
        return f'calibration_translation_{axis}_plus', float(amount.replace('p', '.'))
    if setup.startswith('mix_'):
        match = re.match(r'mix_(\d+)', setup)
        return 'mixed', float(match.group(1)) if match else 0.0
    return 'other', 0.0


def write_grouped_results(rows, selections, config, dataset):
    if dataset not in ('big_sweep', 'calibration_errors'):
        return
    baseline = next(((row,sel) for row,sel in zip(rows,selections)
                     if row['setup']=='baseline'),None)
    if baseline is None:
        raise ValueError('Grouped outputs require a baseline row.')
    groups = {}
    for row,sel in zip(rows,selections):
        if row['setup']=='baseline': continue
        group,level = change_group(row['setup'])
        groups.setdefault(group,[]).append((level,row,sel))
    for group,items in sorted(groups.items()):
        items.sort(key=lambda item:(item[0],item[1]['setup']))
        group_rows = [baseline[0]]+[row for _,row,_ in items]
        group_selections = [baseline[1]]+[sel for _,_,sel in items]
        write_results(group_rows,group_selections,dict(config,change_group=group),
                      OUTPUT_DIR,f'unity_{group}_fid_kid')


def split_existing_results():
    found = False
    for dataset in ('big_sweep','calibration_errors'):
        path = OUTPUT_DIR/f'unity_{dataset}_fid_kid.json'
        if not path.is_file(): continue
        payload = json.loads(path.read_text(encoding='utf-8'))
        write_grouped_results(payload['results'],payload['frames'],payload['config'],dataset)
        found = True
    if not found:
        raise FileNotFoundError(f'No existing Unity result JSON files under {OUTPUT_DIR}')


def match_frames(baseline, target):
    """Compare the same surviving scenes after invalid embeddings are removed."""
    target_by_id = {sample.frame_id: sample for sample in target}
    matched_baseline = [sample for sample in baseline if sample.frame_id in target_by_id]
    matched_target = [target_by_id[sample.frame_id] for sample in matched_baseline]
    if len(matched_baseline) < 2:
        raise RuntimeError('Fewer than two common valid frames for comparison.')
    return matched_baseline, matched_target


def run_scan_patterns(extractors, comparisons):
    root = SCAN_PATTERN_ROOT
    if not root.is_dir():
        raise FileNotFoundError(root)
    sensors = list(SCAN_PATTERN_SENSORS)
    if not sensors or sensors[0] != ('baseline', POINTCLOUD_SENSOR):
        raise ValueError('SCAN_PATTERN_SENSORS must begin with the Grid baseline.')
    print(f'[SCAN PATTERNS] {sensors}')
    baseline, base_counts = load_samples(root, BASELINE_CAPTURE_ID, extractors,
        pointcloud_sensor=POINTCLOUD_SENSOR, setup='baseline')
    rows, selections = [], []
    for setup, sensor in sensors:
        target, counts = (baseline, base_counts) if setup == 'baseline' else load_samples(
            root, BASELINE_CAPTURE_ID, extractors, pointcloud_sensor=sensor, setup=setup)
        paired_baseline, paired_target = match_frames(baseline, target)
        row, selection = comparisons.evaluate(paired_baseline, paired_target,
            dataset='scan_patterns', setup=setup, baseline_name='baseline',
            capture_id=BASELINE_CAPTURE_ID)
        row['pointcloud_sensor'] = sensor
        selection.update(setup=setup, pointcloud_sensor=sensor, feature_sources=counts)
        rows.append(row)
        selections.append(selection)
        print(f'[RESULT] {row}')
        config = dict(config_summary(), scan_pattern_sensors=sensors,
            frame_policy='same common valid frame IDs on both sides')
        write_results(rows, selections, config, OUTPUT_DIR, 'unity_scan_patterns_fid_kid')


def main():
    if SPLIT_EXISTING_RESULTS_ONLY:
        split_existing_results()
        return
    device=resolve_device(DEVICE)
    extractors=Extractors(device,LIDAR_NUM_POINTS,LIDAR_SAMPLING_MODE)
    comparisons=Comparisons(device=device,seed=SEED,subset_size=KID_SUBSET_SIZE,subsets=KID_SUBSETS)
    jobs=[]
    if RUN_BIG_SWEEP: jobs.append(('big_sweep',BIG_SWEEP_ROOT,BIG_SWEEP_CAPTURE_IDS))
    if RUN_SCAN_PATTERNS:
        run_scan_patterns(extractors, comparisons)
    if RUN_CALIBRATION_ERRORS and not RUN_BIG_SWEEP:
        jobs.append(('big_sweep',BIG_SWEEP_ROOT,(BASELINE_CAPTURE_ID,)))
    if not jobs:
        if RUN_SCAN_PATTERNS: return
        raise ValueError('No experiments enabled.')
    for dataset,root,configured in jobs:
        if not root.is_dir(): raise FileNotFoundError(root)
        captures=list(configured) if configured is not None else discover_lidar_captures(root)
        if BASELINE_CAPTURE_ID not in captures:
            raise ValueError(f'{root}: baseline {BASELINE_CAPTURE_ID!r} not found among {captures}')
        captures=[BASELINE_CAPTURE_ID]+[x for x in captures if x!=BASELINE_CAPTURE_ID]
        print(f'[CAPTURES] {dataset}: {captures}')
        baseline,base_counts=load_samples(root,BASELINE_CAPTURE_ID,extractors)
        rows,selections=[],[]
        for capture in captures:
            target,counts=(baseline,base_counts) if capture==BASELINE_CAPTURE_ID else load_samples(root,capture,extractors)
            row,selection=comparisons.evaluate(baseline,target,dataset=dataset,
                setup=setup_name(capture),baseline_name='baseline',capture_id=capture)
            selection.update(setup=row['setup'],feature_sources=counts)
            rows.append(row);selections.append(selection)
            print(f'[RESULT] {row}')
            config = dict(config_summary(),capture_ids=captures)
            write_results(rows,selections,config,OUTPUT_DIR,f'unity_{dataset}_fid_kid')
            write_grouped_results(rows,selections,config,dataset)
        if dataset=='big_sweep' and RUN_CALIBRATION_ERRORS:
            error_rows,error_selections=[],[]
            for label in ['baseline']+[f'et{axis}_{cm}' for axis in CALIBRATION_AXES for cm in CALIBRATION_TRANSLATIONS_CM]:
                # T_cam_lidar alone changes; these encoders receive unchanged
                # RGB/XYZ. Reuse identical features to exclude FPS noise.
                row,selection=comparisons.evaluate(baseline,baseline,dataset='calibration_errors',
                    setup=label,baseline_name='baseline',capture_id=BASELINE_CAPTURE_ID)
                selection.update(setup=label,features='unchanged baseline RGB/XYZ; calibration not an encoder input')
                error_rows.append(row);error_selections.append(selection)
            write_results(error_rows,error_selections,dict(config_summary(),
                calibration_feature_policy='baseline reuse; T_cam_lidar is not an encoder input'),
                OUTPUT_DIR,'unity_calibration_errors_fid_kid')
            write_grouped_results(error_rows,error_selections,dict(config_summary(),
                calibration_feature_policy='baseline reuse; T_cam_lidar is not an encoder input'),
                'calibration_errors')

if __name__=='__main__': main()
