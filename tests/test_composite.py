import unittest
from dhcomp.composite import composite, SoftComposite
import unittest
import numpy as np
from numpy.typing import NDArray
from numpy.testing import assert_array_max_ulp


class TestComposite(unittest.TestCase):

    def test_ones(self):
        for i in range(1, 3):
            cfrom = np.arange(0, 10)
            cto = np.arange(1, 11)
            samplefrom = np.arange(0,10,0.1)
            sampleto = samplefrom+0.1
            array = np.ones((samplefrom.shape[0],2,i))
            x, _  = composite(cfrom, cto, samplefrom, sampleto, array)
        
            y = np.ones((cfrom.shape[0],2,i))
            assert_array_max_ulp(x,y,10)

    def test_composite_from_to_shape_fail(self):
        fr = np.arange(0, 10)
        to = np.arange(1, 10)
        with self.assertRaises(AssertionError):
            composite(fr, to, np.zeros(1), np.ones(1), np.ones((1, 1)))

    def test_composite_from_to_shape(self):
        fr = np.arange(0, 10)
        to = np.arange(1, 11)
        cfr = np.arange(1)
        cto = np.arange(1)+1
        composite(fr, to, cfr, cto, np.ones((1, 1)))

    def test_random_composite_lengths(self):
        rng = np.random.default_rng(seed=42)
        for i in range(100):
            # create random length composite intervals
            sample_lengths = rng.uniform(0.1, 1,10)
            sum_lengths = np.cumsum(sample_lengths)
            fr = sum_lengths[0:-1]
            to = sum_lengths[1:]
            cfr = np.arange(0,10,0.1)
            cto = cfr+0.1
            array = np.ones((cto.shape[0],1))
            x, _  = composite(fr, to, cfr, cto, array)
            y = np.ones((fr.shape[0],1))
            assert_array_max_ulp(x,y,10)

    def test_random_sample_lengths(self):
        rng = np.random.default_rng(seed=42)
        for i in range(100):
            # create random length sample intervals
            fr = np.arange(0, 10)
            to = np.arange(1, 11)
            sample_lengths = rng.uniform(0.01, 0.2,100)
            sum_lengths = np.cumsum(sample_lengths)
            cfr = sum_lengths[0:-1]
            cto = sum_lengths[1:]
            array = np.ones((cto.shape[0],1))
            x, _  = composite(fr, to, cfr, cto, array)
            y = np.ones((fr.shape[0],1))
            assert_array_max_ulp(x,y,10)
    
    def test_SoftComposite(self):
        cfr = np.arange(0,10,0.1)
        cto = cfr+0.1
        dims = (cfr.shape[0],2,2)
        array = np.ones(dims)

        depths,x,coverage = SoftComposite(samplefrom=cfr, sampleto=cto, array=array,interval=1, offset=0.5, drop_empty_intervals=True,min_coverage=0.1)

        y = np.ones((depths.shape[0],*dims[1:]))
        assert_array_max_ulp(x,y,10)

if __name__ == "__main__":
    unittest.main()
