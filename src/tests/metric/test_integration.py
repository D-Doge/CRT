import unittest
import numpy as np
import torch
from data.data_sample_interface import DataSample
from metric.joint_distribution_metrics import JointFrechetFloat64, JointPolynomialMMD
from metric.rgb_reference_metrics import rgb_uint8_chw, RGBReferenceConfig, compute_reference_rgb_metrics
from metric.distribution_core import polynomial_mmd_subsets

class IntegrationTests(unittest.TestCase):
    def test_wrappers_and_original_embeddings_unchanged(self):
        a = [DataSample('A',str(i),joint_emb=torch.tensor([float(i),1.])) for i in range(4)]
        b = [DataSample('B',str(i),joint_emb=torch.tensor([float(i)+1,1.])) for i in range(4)]
        result = JointFrechetFloat64()(a,b)
        self.assertAlmostEqual(result.score,1)
        self.assertEqual((result.n_A,result.n_B),(4,4))
        self.assertEqual(a[0].joint_emb.dtype,torch.float32)
        result = JointPolynomialMMD(subset_size=3,subsets=5)(a,b)
        self.assertEqual(result.details['subset_size_used'],3)

    def test_kid_matches_dependency_on_features(self):
        from torch_fidelity.metric_kid import kid_features_to_metric
        rng = np.random.RandomState(5)
        a,b = rng.randn(8,4),rng.randn(9,4)
        mean,details = polynomial_mmd_subsets(a,b,5,10,8)
        reference = kid_features_to_metric(torch.from_numpy(a),torch.from_numpy(b),
            kid_subset_size=5,kid_subsets=10,rng_seed=8,verbose=False,
            kid_kernel_poly_degree=3,kid_kernel_poly_gamma=None,
            kid_kernel_poly_coef0=1.,kid_kernel='poly')
        self.assertAlmostEqual(mean,reference['kernel_inception_distance_mean'],places=12)
        self.assertAlmostEqual(details['std'],reference['kernel_inception_distance_std'],places=12)

    def test_rgb_contract(self):
        x=torch.tensor([[[0.,0.5,1.]]]*3)
        self.assertEqual(rgb_uint8_chw(x,'0_1')[0,0].tolist(),[0,128,255])
        with self.assertRaises(ValueError): rgb_uint8_chw(x)
        with self.assertRaises(ValueError): rgb_uint8_chw(x*2,'0_1')
        y=torch.zeros((10,12,3),dtype=torch.uint8)
        self.assertEqual(tuple(rgb_uint8_chw(y).shape),(3,10,12))

if __name__=='__main__': unittest.main()
