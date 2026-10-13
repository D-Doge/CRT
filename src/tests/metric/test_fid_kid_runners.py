"""Runner wiring tests use tiny fixtures and fake encoders, no model downloads."""
import csv
import json
import random
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
from data.data_sample_interface import DataSample
from metric.metric_interface import MetricResult
from run.eval import fid_kid_runner_common as common
from run.eval import run_unity_fid_kid as unity
from run.eval import run_mobima_fid_kid as mobima
from run.eval.fid_kid_mobima_support import load_pcd_xyz


def fake_rgb_metrics(a,b,config):
    assert all(s.rgb is not None for s in a+b)
    return {'RGB-FID':MetricResult(0.0),
            'RGB-KID':MetricResult(-0.1,dict(std=0.0,subset_size_used=min(len(a),len(b))))}


class FakeExtractors:
    def __init__(self,*args):
        self.rgb_id='dinov2_vits14'; self.lidar_id='pointnet2_ssg_fps_np2048'
    def rgb(self,sample,seed): return torch.tensor([sample.rgb.mean().item(),1.])
    def lidar(self,sample,seed): return sample.lidar_xyz.mean(0)


class RunnerTests(unittest.TestCase):
    def test_sampling_matches_original(self):
        ids=[str(i) for i in range(20)]
        expected=list(ids);random.Random(0).shuffle(expected)
        self.assertEqual(common.sample_ids(ids,5,0),expected[:5])
        self.assertEqual(common.sample_ids(ids,100,0),ids)

    def test_unity_full_wiring(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); big=root/'big'; scan=root/'scan'; out=root/'out'
            for folder in (big,scan):
                folder.mkdir()
                (folder/'step0.frame_data.json').write_text(json.dumps({'captures':[
                    {'id':'lidar','annotations':[{'filename':'a.pcd'}]},
                    {'id':'lidar_tx_5','annotations':[{'filename':'b.pcd'}]},
                    {'id':'rgb_camera','filename':'image.png'}]}))
            class Loader:
                def __init__(self,**kw): self.kw=kw
                def get_frame_ids(self): return ['0','1','2','3']
                def load_sample(self,f):
                    shift=0 if self.kw['lidar_capture_id']=='lidar' else 1
                    return DataSample(self.kw['setup_id'],f,
                        rgb=torch.ones(3,5,6)*0.25,
                        lidar_xyz=torch.ones(4,3)*(int(f)+shift))
            module=types.ModuleType('data.dataset.unity_sim_dataset_loader')
            module.UnitySimDatasetLoader=Loader
            with patch.dict('sys.modules',{'data.dataset.unity_sim_dataset_loader':module}), \
                 patch.multiple(unity,BIG_SWEEP_ROOT=big,SCAN_PATTERN_ROOT=scan,
                    OUTPUT_DIR=out,DEVICE='cpu',N_METRIC_FRAMES=3,
                    USE_EXISTING_EMBEDDING_CACHE=False,Extractors=FakeExtractors), \
                 patch.object(common,'compute_reference_rgb_metrics',side_effect=fake_rgb_metrics):
                unity.main()
                # Second run reads new cache: no extraction allowed.
                with patch.object(FakeExtractors,'rgb',side_effect=AssertionError('cache miss')), \
                     patch.object(FakeExtractors,'lidar',side_effect=AssertionError('cache miss')):
                    unity.main()
            physical=json.loads((out/'unity_big_sweep_fid_kid.json').read_text())
            self.assertEqual(len(physical['results']),2)
            self.assertGreater(physical['results'][1]['Joint_Frechet'],0)
            calibration=json.loads((out/'unity_calibration_errors_fid_kid.json').read_text())
            self.assertEqual(len(calibration['results']),16)
            self.assertTrue(all(abs(r['Joint_Frechet'])<1e-9 for r in calibration['results']))
            self.assertEqual(physical['frames'][0]['target_frame_ids'],common.sample_ids(['0','1','2','3'],3,0))

    def test_mobima_full_wiring(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'samples').mkdir();out=root/'out'
            (root/'dataset_meta.json').write_text(json.dumps({'camera_info':{
                'K':[1,0,2,0,1,2,0,0,1],'width':6,'height':5}}))
            eye=np.eye(4).tolist()
            (root/'calibration.json').write_text(json.dumps({'reference_frame':'mid360',
                'transforms':{name:{'matrix':eye} for name in
                ('mid360_to_camera','mid70_to_mid360','velodyne_to_mid360')}}))
            for i in range(3):
                folder=root/'samples'/str(i);folder.mkdir()
                Image.new('RGB',(6,5),(60,60,60)).save(folder/'image_rgb.jpg')
                for j,sensor in enumerate(mobima.SENSORS):
                    (folder/f'{sensor}.pcd').write_text('FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH 2\nHEIGHT 1\nPOINTS 2\nDATA ascii\n'+f'{i+j} 0 0\n{i+j} 1 1\n')
            with patch.multiple(mobima,ROOT_DIR=root,OUTPUT_DIR=out,DEVICE='cpu',
                    USE_EXISTING_EMBEDDING_CACHE=False,Extractors=FakeExtractors), \
                 patch.object(common,'compute_reference_rgb_metrics',side_effect=fake_rgb_metrics):
                mobima.main()
                with patch.object(FakeExtractors,'rgb',side_effect=AssertionError('cache miss')), \
                     patch.object(FakeExtractors,'lidar',side_effect=AssertionError('cache miss')):
                    mobima.main()
            result=json.loads((out/'mobima_fid_kid.json').read_text())
            self.assertEqual(len(result['results']),3)
            self.assertEqual(result['results'][0]['n_target'],3)
            self.assertAlmostEqual(result['results'][1]['Joint_Frechet'],1)
            self.assertFalse((root/'.embedding_cache_mobima').exists())

    def test_unity_old_cache_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            directory=root/'.embedding_cache'/(
                'pc_FOV-Grid_PointCloud__rgb_rgb_camera__lidar_lidar_rx_45__'
                'rgb_emb_dinov2_vits14__lidar_emb_pointnet2_ssg_fps_np2048__jo__hash')/'rx_45'
            directory.mkdir(parents=True)
            torch.save({'rgb_emb':torch.ones(2),'lidar_emb':torch.ones(3)},directory/'frame1.pt')
            paths=unity.existing_cache_index(root,'lidar_rx_45','rx_45')
            self.assertEqual(paths['frame1'],[directory/'frame1.pt'])
            self.assertIsNotNone(common.load_fields(paths['frame1'][0]))
            self.assertEqual(unity.existing_cache_index(root,'lidar','baseline'),{})

    def test_old_cache_direct_and_nested_fields(self):
        fields={'rgb_emb':torch.ones(2),'lidar_emb':torch.ones(3)}
        self.assertIsNotNone(common.cached_fields(fields))
        self.assertIsNotNone(common.cached_fields({'fields':fields}))
        self.assertIsNone(common.cached_fields({'residuals':torch.ones(2)}))

if __name__=='__main__': unittest.main()
