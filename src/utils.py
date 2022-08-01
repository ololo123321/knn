import numpy as np


def cdist_nd(a: np.ndarray, b: np.ndarray, squared: bool = True) -> np.ndarray:
    """
    Numpy analogue of torch.cdist.

    * common case:
    a: [..., n, d]
    b: [..., m, d]
    return c: [..., n, m], c[..., i, j] = norm(diff(a[..., k, :], d[..., k, :]))

    * partial case (for code books; nb - number of blocks):
    a: [nb, n, d]
    b: [nb, m, d]
    return: c: [nb, n, m]; c[k, i, j] = norm(diff(a[k, i, :], b[k, j, :]))

    * complexity (suppose first dimentions are n0, ..., nk. N = n0 * n1 * ... * nk):
    N * m * d                               multiplies by -1
    N * n * m * (d^2 + d - 1)               sum
    N * n * m * d                           sqr
    """
    # [nb, n, m, d]; N * m * d multiplies by -1, # N * n * m * d * d sum
    diff = np.expand_dims(a, a.ndim - 1) - np.expand_dims(b, b.ndim - 2)
    norms = np.square(diff).sum(-1)  # [nb, n, m]  # N * n * m * d sqr, N * n * m * (d - 1) sum
    if not squared:
        norms = np.sqrt(norms)
    return norms


def cdist_2d_v1(a: np.ndarray, b: np.ndarray, squared: bool = True) -> np.ndarray:
    """
    Numpy analogue of scipy.spatial.distance.cdist. Implemented as partial case of `cdist_nd`.
    Due to redundant broadcasting this realization is much slower, than `cdist_2d_v2`

    a: [n, d]
    b: [m, d]
    return: c: [n, m]; c[i, j] = norm(diff(a[i], b[j]))
    """
    return cdist_nd(a[None], b[None], squared=squared).squeeze(0)


def cdist_2d_v2(
        a: np.ndarray,
        b: np.ndarray,
        squared: bool = True,
        a_norm_squared: np.ndarray = None,
        b_norm_squared: np.ndarray = None
) -> np.ndarray:
    """
    Numpy analogue of scipy.spatial.distance.cdist. Implemented with minimal number of broadcasting.

    a: [n, d]
    b: [m, d]
    a_norm_squared: [n]
    b_norm_squared: [m]
    return: c: [n, m]; c[i, j] = norm(diff(a[i], b[j]))

    * complexity:
    n * m                                               multiplies by -2
    n * m * d                                           prod
    n * m * (d + 1) + (n + m) * (d - 1)                 sum
    n * d + m * d                                       sqr
    """
    res = -2.0 * a @ b.T  # [n, m]; n * m * d prod, n * m * (d - 1) sum
    if a_norm_squared is not None:
        res += a_norm_squared[:, None]  # [n, 1]; n * m sum
    else:
        res += get_norm(a)[:, None]  # [n, 1]; n * d sqr, n * (d - 1) sum, n * m sum
    if b_norm_squared is not None:
        res += b_norm_squared[None, :]  # [1, m]; n * m sum
    else:
        res += get_norm(b)[None, :]  # [1, m]; m * d sqr, m * (d - 1) sum, n * m sum
    res = np.maximum(res, 0.0)  # might be values like -7.105427357601002e-15
    if not squared:
        res = np.sqrt(res)
    return res


def get_norm(x, axis: int = -1, squared: bool = True):
    res = np.square(x).sum(axis)
    if not squared:
        res = np.sqrt(res)
    return res


def get_centroids_distortion(data: np.ndarray, centroids: np.ndarray) -> float:
    """
    data: [n, d], float
    centroids: [m, d], float
    return: float
    """
    dists = cdist_2d_v2(data, centroids)
    xs = np.arange(data.shape[0]).astype(np.int32)
    ys = dists.argmin(1)
    return dists[xs, ys].mean()
