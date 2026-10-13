"""Run only reference RGB and joint FID/KID-style metrics on MOBIMA.
Run from project root: PYTHONPATH=src python -m run.eval.run_mobima_fid_kid
"""
from pathlib import Path
# ========================== EDIT CONFIGURATION HERE ==========================
ROOT_DIR = Path(r'E:\data\final_MOBIMA\final_MOBIMA')
OUTPUT_DIR = Path(r'E:\domain_gap_metric\results\KID_FID_Results')
SENSORS = ('mid360', 'mid70', 'velodyne')
BASELINE_SENSOR = 'mid360'
N_METRIC_FRAMES = 70
SEED = 0
DEVICE = 'cuda'
LIDAR_NUM_POINTS = 2048
LIDAR_SAMPLING_MODE = 'fps'
KID_SUBSET_SIZE = 1000
KID_SUBSETS = 100
USE_EXISTING_EMBEDDING_CACHE = True
EXISTING_CACHE_DIR = '.embedding_cache_mobima'
NEW_CACHE_DIR = '.embedding_cache_fid_kid_v1'
# These identify the old cache namespace only; no residuals are computed.
OLD_CACHE_RESIDUAL_EDGE_THRESHOLD = 0.2
OLD_CACHE_RESIDUAL_MAX_POINTS = 100
OLD_CACHE_RESIDUAL_MIN_DEPTH = 0.05
# =============================================================================

from types import SimpleNamespace
from run.eval.fid_kid_mobima_support import MobimaContext, MetricCache
from run.eval.fid_kid_runner_common import (Extractors, Comparisons, resolve_device,
    sample_ids, stable_seed, attach_embeddings, load_fields, save_fields, write_results)


def main():
    if BASELINE_SENSOR not in SENSORS: raise ValueError('Baseline must be in SENSORS.')
    device=resolve_device(DEVICE)
    ctx=MobimaContext(ROOT_DIR)
    extractors=Extractors(device,LIDAR_NUM_POINTS,LIDAR_SAMPLING_MODE)
    # Reproduce the original namespace for read-only cache reuse.
    old=MetricCache(ROOT_DIR/EXISTING_CACHE_DIR,ctx=ctx,
        rgb_extractor=SimpleNamespace(extractor_id=extractors.rgb_id),
        lidar_extractor=SimpleNamespace(extractor_id=extractors.lidar_id),
        residual_edge_threshold=OLD_CACHE_RESIDUAL_EDGE_THRESHOLD,
        residual_max_points=OLD_CACHE_RESIDUAL_MAX_POINTS,
        residual_min_depth=OLD_CACHE_RESIDUAL_MIN_DEPTH,
        residuals_enabled=True,refresh=False,enabled=USE_EXISTING_EMBEDDING_CACHE)
    ids=sample_ids(ctx.discover_common_sensor_frames(SENSORS),N_METRIC_FRAMES,SEED)
    if len(ids)<2: raise RuntimeError('Fewer than 2 common frames.')
    print(f'[MOBIMA] {len(ids)} common frames; baseline={BASELINE_SENSOR}')
    samples={s:[] for s in SENSORS}
    counts={s:dict(existing_cache=0,new_cache=0,recomputed=0) for s in SENSORS}
    for i,frame_id in enumerate(ids,1):
        rgb,K,size=ctx.load_image_and_intrinsics(frame_id)
        rgb_emb=old.load_rgb(frame_id)
        for sensor in SENSORS:
            new_path=(ROOT_DIR/NEW_CACHE_DIR/
                f'{extractors.rgb_id}__{extractors.lidar_id}__seed{SEED}'/sensor/(frame_id+'.pt'))
            fields=load_fields(new_path)
            if fields is not None:
                counts[sensor]['new_cache']+=1
            else:
                cached=old.load_sensor(sensor,frame_id)
                lidar_emb=cached.get('lidar_emb') if cached is not None else None
                if rgb_emb is not None and lidar_emb is not None:
                    fields=(rgb_emb,lidar_emb)
                    counts[sensor]['existing_cache']+=1
                else:
                    raw=ctx.raw_sample(frame_id,sensor,rgb=rgb,K=K,image_size=size)
                    if rgb_emb is None:
                        rgb_emb=extractors.rgb(raw,stable_seed('rgb',frame_id,base=SEED))
                    if lidar_emb is None:
                        lidar_emb=extractors.lidar(raw,stable_seed('lidar_emb',sensor,frame_id,base=SEED))
                    fields=(rgb_emb,lidar_emb)
                    counts[sensor]['recomputed']+=1
            # No point cloud loading is needed when cached embeddings exist.
            from data.data_sample_interface import DataSample
            raw_rgb=DataSample(setup_id=sensor,frame_id=frame_id,rgb=rgb,
                intrinsics=K,image_size=size)
            sample=attach_embeddings(raw_rgb,*fields)
            if not new_path.is_file(): save_fields(new_path,sample)
            samples[sensor].append(sample)
        if i==1 or i%25==0 or i==len(ids): print(f'[MOBIMA] {i}/{len(ids)}')
    comparisons=Comparisons(device=device,seed=SEED,subset_size=KID_SUBSET_SIZE,subsets=KID_SUBSETS)
    rows,selections=[],[]
    config=dict(root_dir=str(ROOT_DIR),sensors=list(SENSORS),baseline_sensor=BASELINE_SENSOR,
        seed=SEED,n_metric_frames_requested=N_METRIC_FRAMES,
        lidar_num_points=LIDAR_NUM_POINTS,lidar_sampling_mode=LIDAR_SAMPLING_MODE,
        joint_normalization=False,kid_subsets=KID_SUBSETS,kid_subset_size_requested=KID_SUBSET_SIZE,
        old_cache_dir=EXISTING_CACHE_DIR,new_cache_dir=NEW_CACHE_DIR,
        feature_sources=counts,device=DEVICE)
    for sensor in SENSORS:
        row,selection=comparisons.evaluate(samples[BASELINE_SENSOR],samples[sensor],
            dataset='mobima',setup=sensor,baseline_name=BASELINE_SENSOR,capture_id=sensor)
        selection.update(setup=sensor)
        rows.append(row);selections.append(selection)
        print(f'[RESULT] {row}')
        write_results(rows,selections,config,OUTPUT_DIR,'mobima_fid_kid')

if __name__=='__main__': main()
