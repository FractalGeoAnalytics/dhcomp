import unittest
from src.composite import composite
import unittest
import numpy as np
from numpy.typing import NDArray
from numpy.testing import assert_array_equal


class TestComposite(unittest.TestCase):
    def test_enclosed(self):
        fr: float = 0
        to: float = 1
        samplefrom: NDArray = np.arange(0, 2, 0.1)
        sampleto: NDArray = samplefrom + 0.1

        idx = composite._enclosed(fr, to, samplefrom, sampleto)
        idxtest = samplefrom < 1
        assert_array_equal(idx, idxtest)

    def test_top_partial(self):
        self.assertFalse(1)

    def test_bottom_partial(self):
        self.assertFalse(1)

    def test_composite_from_to_shape_fail(self):
        fr = np.arange(0, 10)
        to = np.arange(1, 10)
        with self.assertRaises(AssertionError):
            composite(fr, to, 0, 1, np.ones((1, 1)))

    def test_composite_from_to_shape(self):
        fr = np.arange(0, 10)
        to = np.arange(1, 11)
        composite(fr, to, 0, 1, np.ones((1, 1)))


if __name__ == "__main__":
    unittest.main()
