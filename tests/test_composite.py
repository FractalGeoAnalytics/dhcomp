import unittest
from dhcomp.composite import (
    composite,
    SoftComposite,
    _greedy_composite,
    _global_composite,
    HardComposite,
)
import unittest
import numpy as np
from numpy.typing import NDArray
from numpy.testing import assert_array_max_ulp, assert_array_equal


class TestComposite(unittest.TestCase):
    def test_ones(self):
        cfrom = np.arange(0, 10)
        cto = np.arange(1, 11)
        samplefrom = np.arange(0, 10, 0.1)
        sampleto = samplefrom + 0.1
        rows = samplefrom.shape[0]
        for i in [(rows, 2), (rows, 2, 2)]:
            array = np.ones(i)
            x, _ = composite(cfrom, cto, samplefrom, sampleto, array)
            shape_comp = list(i)
            shape_comp[0] = 10
            y = np.ones(shape_comp)
            assert_array_max_ulp(x, y)

    def test_composite_from_to_shape_fail(self):
        fr = np.arange(0, 10)
        to = np.arange(1, 10)
        with self.assertRaises(AssertionError):
            composite(fr, to, np.zeros(1), np.ones(1), np.ones((1, 1)))

    def test_composite_from_to_shape(self):
        fr = np.arange(0, 10)
        to = np.arange(1, 11)
        cfr = np.arange(1)
        cto = np.arange(1) + 1
        composite(fr, to, cfr, cto, np.ones((1, 1)))

    def test_random_composite_lengths(self):
        rng = np.random.default_rng(seed=42)
        for i in range(100):
            # create random length composite intervals
            sample_lengths = rng.uniform(0.1, 1, 10)
            sum_lengths = np.cumsum(sample_lengths)
            fr = sum_lengths[0:-1]
            to = sum_lengths[1:]
            cfr = np.arange(0, 10, 0.1)
            cto = cfr + 0.1
            array = np.ones((cto.shape[0], 1))
            x, _ = composite(fr, to, cfr, cto, array)
            y = np.ones((fr.shape[0], 1))
            assert_array_max_ulp(x, y, 10)

    def test_random_sample_lengths(self):
        rng = np.random.default_rng(seed=42)
        for i in range(100):
            # create random length sample intervals
            fr = np.arange(0, 10)
            to = np.arange(1, 11)
            sample_lengths = rng.uniform(0.01, 0.2, 100)
            sum_lengths = np.cumsum(sample_lengths)
            cfr = sum_lengths[0:-1]
            cto = sum_lengths[1:]
            array = np.ones((cto.shape[0], 1))
            x, _ = composite(fr, to, cfr, cto, array)
            y = np.ones((fr.shape[0], 1))
            assert_array_max_ulp(x, y, 10)

    def test_SoftComposite(self):
        cfr = np.arange(0, 10, 0.1).reshape(-1, 1)
        cto = cfr + 0.1
        dims = (cfr.shape[0], 2)
        array = np.ones(dims)

        depths, x, coverage = SoftComposite(
            samplefrom=cfr,
            sampleto=cto,
            array=array,
            interval=1,
            offset=0.5,
            drop_empty_intervals=True,
            min_coverage=0.1,
        )

        y = np.ones((depths.shape[0], *dims[1:]))
        assert_array_max_ulp(x, y, 10)

    def test_greedy_forward_divisible(self):
        cfr = np.arange(0, 10, 0.1).reshape(-1, 1)
        cto = cfr + 0.1
        depths = np.unique(np.concatenate([cfr, cto]))
        comps = _greedy_composite(depths, 2.0, "forwards")
        assert_array_equal(comps, [0, 2, 4, 6, 8, 10])

    def test_greedy_backward_divisible(self):
        cfr = np.arange(0, 10, 0.1).reshape(-1, 1)
        cto = cfr + 0.1
        depths = np.unique(np.concatenate([cfr, cto]))
        comps = _greedy_composite(depths, 2.0, "backwards", min_length=0)
        assert_array_equal(comps, [0, 2, 4, 6, 8, 10])

    def test_greedy_forward_with_short_sample(self):
        depth_from = np.arange(0, 11, 0.1).reshape(-1, 1)
        depth_to = depth_from + 0.1
        composite_length = 2
        depths = np.unique(np.concatenate([depth_from, depth_to]))
        comps = _greedy_composite(depths, composite_length, "forwards")
        assert_array_equal(comps, [0, 2, 4, 6, 8, 10, 11])

    def test_greedy_backward_with_short_sample(self):
        cfr = np.arange(0, 11, 0.1).reshape(-1, 1)
        cto = cfr + 0.1
        depths = np.unique(np.concatenate([cfr, cto]))
        comps = _greedy_composite(depths, 2.0, "backwards")
        assert_array_equal(comps, [0, 1, 3, 5, 7, 9, 11])

    def test_global_divisible(self):
        cfr = np.arange(0, 10, 0.1).reshape(-1, 1)
        cto = cfr + 0.1
        depths = np.unique(np.concatenate([cfr, cto]))
        comps = _global_composite(depths, 2.0)
        assert_array_equal(comps, [0, 2, 4, 6, 8, 10])

    def test_HardComposite(self):
        cfr = np.arange(0, 10, 0.1).reshape(-1, 1)
        cto = cfr + 0.1
        dims = (cfr.shape[0], 2)
        array = np.ones(dims)

        depths, x, coverage = HardComposite(
            samplefrom=cfr,
            sampleto=cto,
            array=array,
            interval=1,
            drop_empty_intervals=True,
            min_coverage=0.1,
        )

        y = np.ones((depths.shape[0], *dims[1:]))
        assert_array_max_ulp(x, y, 10)

    def test_HardComposite_greedy_minlength(self):
        cfr = np.arange(0, 11, 0.1).reshape(-1, 1)
        cto = cfr + 0.1
        dims = (cfr.shape[0], 2)
        array = np.ones(dims)
        min_length = 2
        depths, x, coverage = HardComposite(
            samplefrom=cfr,
            sampleto=cto,
            array=array,
            interval=2,
            drop_empty_intervals=True,
            min_coverage=0.1,
            min_length=min_length,
        )

        udepths = np.unique(depths)
        assert np.all(np.diff(udepths) >= min_length)

    def test_HardComposite_greedy_backward(self):
        cfr = np.arange(0, 10, 0.1).reshape(-1, 1)
        cto = cfr + 0.1
        dims = (cfr.shape[0], 2)
        array = np.ones(dims)

        depths, x, coverage = HardComposite(
            samplefrom=cfr,
            sampleto=cto,
            array=array,
            interval=1,
            drop_empty_intervals=True,
            direction="backwards",
            min_coverage=0.1,
        )

        y = np.ones((depths.shape[0], *dims[1:]))
        assert_array_max_ulp(x, y, 10)

    def test_HardComposite_global(self):
        cfr = np.arange(0, 10, 0.1).reshape(-1, 1)
        cto = cfr + 0.1
        dims = (cfr.shape[0], 2)
        array = np.ones(dims)

        depths, x, coverage = HardComposite(
            samplefrom=cfr,
            sampleto=cto,
            array=array,
            interval=1,
            drop_empty_intervals=True,
            method="global",
            min_coverage=0.1,
        )

        y = np.ones((depths.shape[0], *dims[1:]))
        assert_array_max_ulp(x, y, 10)

    def test_HardComposite_global_minlength_check(self):
        cfr = np.arange(0, 10, 0.3).reshape(-1, 1)
        cto = cfr + 0.1
        dims = (cfr.shape[0], 2)
        array = np.ones(dims)

        depths, x, coverage = HardComposite(
            samplefrom=cfr,
            sampleto=cto,
            array=array,
            interval=1,
            drop_empty_intervals=True,
            method="global",
            min_coverage=0.1,
            min_length=1,
        )

        udepths = np.unique(depths)
        assert np.all(np.diff(udepths) >= 1)

    def test_HardComposite_global_minlength_0(self):
        cfr = np.arange(0, 10, 0.3).reshape(-1, 1)
        cto = cfr + 0.1
        dims = (cfr.shape[0], 2)
        array = np.ones(dims)
        min_length = 0
        depths, x, coverage = HardComposite(
            samplefrom=cfr,
            sampleto=cto,
            array=array,
            interval=1,
            drop_empty_intervals=True,
            method="global",
            min_coverage=0.1,
            min_length=0,
        )

        udepths = np.unique(depths)
        assert np.all(np.diff(udepths) >= min_length)

    def test_HardComposite_global_fuzz(self):

        rng = np.random.default_rng(seed=42)
        for i in range(1000):
            # create random length composite intervals
            sample_lengths = rng.uniform(0.1, 1, 10)
            sum_lengths = np.cumsum(sample_lengths)
            fr = sum_lengths[0:-1]
            to = sum_lengths[1:]
            array = np.ones((to.shape[0], 1))
            min_length = 1
            depths, x, coverage = HardComposite(
                samplefrom=fr,
                sampleto=to,
                array=array,
                interval=1,
                drop_empty_intervals=True,
                method="global",
                min_coverage=0.1,
                min_length=min_length,
            )
            udepths = np.unique(depths)
            assert np.all(np.diff(udepths) >= min_length)

    def test_HardComposite_global_min_length_validation(self):
        cfr = np.arange(0, 10, 0.3).reshape(-1, 1)
        cto = cfr + 0.1
        dims = (cfr.shape[0], 2)
        array = np.ones(dims)
        with self.assertRaises(ValueError) as context:
            depths, x, coverage = HardComposite(
                samplefrom=cfr,
                sampleto=cto,
                array=array,
                interval=1,
                drop_empty_intervals=True,
                method="global",
                min_coverage=0.1,
                min_length=2,
            )


if __name__ == "__main__":
    unittest.main()
