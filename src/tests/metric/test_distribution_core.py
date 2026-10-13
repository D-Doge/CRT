import unittest
import numpy as np
from scipy.linalg import sqrtm
from metric.distribution_core import frechet_distance, polynomial_mmd, polynomial_mmd_subsets


class DistributionTests(unittest.TestCase):
    def test_frechet_matches_covariance_reference(self):
        rng = np.random.RandomState(8)
        for n, m, d in [(20, 25, 5), (5, 7, 30), (5, 40, 10)]:
            a, b = rng.randn(n, d), rng.randn(m, d) + 0.2
            ca, cb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
            root_a = sqrtm(ca).real
            middle = root_a @ cb @ root_a
            trace_root = np.sqrt(np.maximum(0, np.linalg.eigvalsh((middle + middle.T) / 2))).sum()
            reference = np.sum((a.mean(0)-b.mean(0))**2) + np.trace(ca)+np.trace(cb)-2*trace_root
            score, _ = frechet_distance(a, b)
            self.assertAlmostEqual(score, reference, delta=2e-6)
            reverse, _ = frechet_distance(b, a)
            self.assertAlmostEqual(score, reverse, places=10)

    def test_frechet_identity_and_mean_shift(self):
        rng = np.random.RandomState(7)
        a = rng.randn(10, 100)
        self.assertAlmostEqual(frechet_distance(a, a)[0], 0, places=10)
        self.assertAlmostEqual(frechet_distance(a, a + 0.5)[0], 25, places=10)
        self.assertEqual(frechet_distance(np.zeros((3, 4)), np.zeros((7, 4)))[0], 0)

    def test_polynomial_matches_explicit_pairs(self):
        a = np.array([[1., 2.], [3., 1.], [0., 2.]])
        b = np.array([[2., 0.], [1., 1.]])
        k = lambda x, y: (x @ y / 2 + 1)**3
        expected = sum(k(x,y) for i,x in enumerate(a) for j,y in enumerate(a) if i!=j)/6
        expected += sum(k(x,y) for i,x in enumerate(b) for j,y in enumerate(b) if i!=j)/2
        expected -= 2*sum(k(x,y) for x in a for y in b)/6
        self.assertAlmostEqual(polynomial_mmd(a,b), expected)

    def test_negative_preserved(self):
        a = np.array([[0.], [1.]])
        self.assertLess(polynomial_mmd(a,a), 0)

    def test_subsets_repeatable_and_capped(self):
        rng = np.random.RandomState(2)
        a, b = rng.randn(8,4), rng.randn(10,4)
        x = polynomial_mmd_subsets(a,b,1000,10,3)
        self.assertEqual(x, polynomial_mmd_subsets(a,b,1000,10,3))
        self.assertEqual(x[1]['subset_size_used'],8)

    def test_invalid_inputs(self):
        for a,b in [(np.ones((1,3)),np.ones((2,3))),
                    (np.ones((2,3)),np.ones((2,4))),
                    (np.full((2,3),np.nan),np.ones((2,3)))]:
            with self.assertRaises(ValueError): frechet_distance(a,b)
            with self.assertRaises(ValueError): polynomial_mmd(a,b)

if __name__ == '__main__': unittest.main()
