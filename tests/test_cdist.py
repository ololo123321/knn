import numpy as np
from src.utils import cdist_nd, cdist_2d_v1, cdist_2d_v2
from scipy.spatial.distance import cdist

n = 2
m = 3
d = 4
a = np.random.uniform(size=(n, d))
b = np.random.uniform(size=(m, d))
expected = cdist(a, b)


def test_cdist_nd():
    actual = cdist_nd(a[None, None, ...], np.tile(b[None, None, ...], [1, 3, 1, 1]), squared=False)[0, 0]
    assert np.allclose(actual, expected)


def test_cdist_2d():
    actual_v1 = cdist_2d_v1(a, b, squared=False)
    assert np.allclose(actual_v1, expected)
    actual_v2 = cdist_2d_v2(a, b, squared=False)
    assert np.allclose(actual_v2, expected)
    a_norm_squared = np.sq
