import random
import tqdm
from abc import ABC, abstractmethod
from typing import NamedTuple
from collections import defaultdict

import numpy as np

from src.utils import cdist_2d_v2 as cdist, cdist_nd
from src.kmeans import KMeans


SearchResult = NamedTuple("SearchResult", [("indices", np.ndarray), ("distances", np.ndarray)])


def dists2outputs(dists: np.ndarray, k: int) -> SearchResult:
    """
    dists: [m, n], float32
    k: int
    """
    indices = dists.argsort(1)  # [m, n], int
    indices = indices[:, :k]  # [m, k]
    dists_sorted = np.zeros((dists.shape[0], k), dtype=dists.dtype)
    for i in range(dists.shape[0]):
        dists_sorted[i] = dists[i, indices[i]]
    return SearchResult(indices=indices, distances=dists_sorted)


def get_train_data(data: np.ndarray, n_train: int = -1, random_state: int = None):
    if (n_train <= 0) or (n_train >= data.shape[0]):
        return data
    rng = random.Random(random_state)
    idx = list(range(data.shape[0]))
    train_idx = rng.sample(idx, n_train)
    return data[train_idx]


class DistanceComputationModes:
    """
    ADC - asynchronous distance computation
    SDC - synchronous distance computation
    see https://lear.inrialpes.fr/pubs/2011/JDS11/jegou_searching_with_quantization.pdf
    """
    ADC = "adc"
    SDC = "sdc"


class BaseIndexer(ABC):
    @abstractmethod
    def fit(self, data: np.ndarray) -> None:
        """
        data: [n, D], float32
        """

    @abstractmethod
    def search(self, queries: np.ndarray, k: int = 10, **kwargs) -> SearchResult:
        """
        queries: [m, D]
        return: (indices: [m, k], int32; distances: [m, k], float32)
        """


class IndexerFlat(BaseIndexer):
    def __init__(self):
        self._data = None

    def fit(self, data: np.ndarray):
        self._data = data

    def search(self, queries: np.ndarray, k: int = 10, **kwargs) -> SearchResult:
        dists = cdist(queries, self._data, squared=False)  # [m, n]
        return dists2outputs(dists, k=k)


class IndexerPQ(BaseIndexer):
    def __init__(self, D: int, d: int, nc: int, n_loops: int = 0):
        assert D % d == 0
        assert n_loops in [0, 2, 3]
        self.D = D
        self.d = d
        self.nb = D // d
        self.nc = nc
        self.n_loops = n_loops

        self._data_enc = None  # [n, nb], uint8
        self._cb = None  # [nc, D], float32
        self._cb_reshaped = None  # [nb, nc, d], float32
        self._adc_dists = None  # [nb, nc, nc], float32
        self._adc_kmeans = []  # [nb], KMeans

    def fit(self, data: np.ndarray, n_train: int = -1, random_state: int = None):
        self._data_enc = np.zeros((data.shape[0], self.nb), dtype=np.uint8)
        self._cb = np.zeros((self.nc, self.D), dtype=data.dtype)
        train_data = get_train_data(data, n_train, random_state)
        for i in tqdm.trange(self.nb):
            s = np.s_[i * self.d:(i + 1) * self.d]
            kmeans = KMeans(n_clusters=self.nc)
            kmeans.fit(train_data[:, s])
            self._cb[:, s] = kmeans.cluster_centers_  # [num_codes, d]
            self._data_enc[:, i] = kmeans.transform(data[:, s]).argmin(1)
            self._adc_kmeans.append(kmeans)
        self._cb_reshaped = self._cb.reshape(-1, self.nb, self.d).transpose(1, 0, 2)  # [nb, nc, d]
        self._adc_dists = cdist_nd(self._cb_reshaped, self._cb_reshaped, squared=True)

    def search(self, queries: np.ndarray, k: int = 10, **kwargs) -> SearchResult:
        dists = self._search(queries, self._data_enc, **kwargs)
        return dists2outputs(dists, k=k)

    def _search(self, queries: np.ndarray, data_enc: np.ndarray, **kwargs) -> np.ndarray:
        dc_mode = kwargs.get("dc_mode", DistanceComputationModes.ADC)
        if dc_mode == DistanceComputationModes.ADC:
            queries_blocked = queries.reshape(-1, self.nb, self.d).transpose(1, 0, 2)  # [m, D] -> [m, nb, d] -> [nb, m, d]
            codebook_dist = cdist_nd(queries_blocked, self._cb_reshaped, squared=True)  # [nb, m, nc]
            if self.n_loops == 3:
                f = self._get_dists_3_loops
            elif self.n_loops == 2:
                f = self._get_dists_2_loops
            elif self.n_loops == 0:
                f = self._get_dists_no_loops
            else:
                raise
            dists = f(data_enc, codebook_dist)
        elif DistanceComputationModes.SDC:
            queries_enc = np.zeros((queries.shape[0], self.nb), dtype=np.uint8)
            for i in range(self.nb):
                s = np.s_[i * self.d:(i + 1) * self.d]
                kmeans = self._adc_kmeans[i]
                queries_enc[:, i] = kmeans.transform(queries[:, s]).argmin(1)
            xs = np.arange(self.nb).astype(np.int32)[:, None, None]
            ys = queries_enc.transpose(1, 0)[:, :, None]  # [nb, m, 1]
            zs = self._data_enc.transpose(1, 0)[:, None, :]  # [nb, 1, n]
            dists = self._adc_dists[xs, ys, zs].transpose(1, 2, 0).sum(-1)  # [nb, m, n] -> [m, n, nb] -> [m, n]
        else:
            raise
        return dists ** 0.5

    def _get_dists_3_loops(self, data_enc: np.ndarray, codebook_dist: np.ndarray) -> np.ndarray:
        """
        data_enc: [n, nb], uint8
        codebook_dist: [nb, m, nc], float32
        return: dists: [m, n], float32
        """
        n = data_enc.shape[0]
        m = codebook_dist.shape[1]
        dists = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                for b in range(self.nb):
                    c = data_enc[j, b]
                    dists[i, j] += codebook_dist[b, i, c]
        return dists

    def _get_dists_2_loops(self, data_enc: np.ndarray, codebook_dist: np.ndarray) -> np.ndarray:
        n = data_enc.shape[0]
        m = codebook_dist.shape[1]
        dists = np.zeros((m, n))
        xs = np.arange(self.nb).astype(np.int32)
        for i in range(m):
            for j in range(n):
                dists[i, j] = codebook_dist[xs, i, data_enc[j]].sum()
        return dists

    def _get_dists_no_loops(self, data_enc: np.ndarray, codebook_dist: np.ndarray) -> np.ndarray:
        xs = np.arange(self.nb).astype(np.int32)[None]  # [1, nb], broadcast to all rows
        # [nb, m, nc] -> [m, nb, nc] -> [m, n, nb] -> [m, n]
        return codebook_dist.transpose(1, 0, 2)[:, xs, data_enc].sum(-1)


class IndexerIVFPQ(IndexerPQ):
    def __init__(self, num_leaves: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_leaves = num_leaves

        self._kmeans = None
        self._leaf_to_datapoints = None

    def fit(self, data: np.ndarray, n_train: int = -1, random_state: int = None) -> None:
        train_data = get_train_data(data, n_train, random_state)

        # partitioning data into `num_leaves` clusters
        self._kmeans = KMeans(n_clusters=self.num_leaves)
        self._kmeans.fit(train_data)
        datapoint_to_leaf = self._kmeans.transform(data).argmin(1)

        # create leaf -> datapoints map
        self._leaf_to_datapoints = defaultdict(list)
        for i in range(data.shape[0]):
            self._leaf_to_datapoints[datapoint_to_leaf[i]].append(i)

        # compute residuals
        residuals = np.zeros_like(data)
        for i in range(data.shape[0]):
            residuals[i] = data[i] - self._kmeans.cluster_centers_[datapoint_to_leaf[i]]

        # encode residuals
        super().fit(residuals)

    def search(self, queries: np.ndarray, k: int = 10, nprobe: int = 1) -> SearchResult:
        m = queries.shape[0]
        leaves_sorted = self._kmeans.transform(queries).argsort(1)  # [m, num_leves]
        dists = np.zeros((m, k), dtype=np.float32)
        indices = np.zeros((m, k), dtype=np.int32)
        for i in range(m):
            qi = queries[i]  # [D]
            dists_i = []
            indices_i = []
            for j in range(nprobe):
                leaf = leaves_sorted[i, j]
                datapoints = self._leaf_to_datapoints[leaf]
                residuals_enc_ij = np.zeros((len(datapoints), self.nb),
                                            dtype=self._data_enc.dtype)  # [num_candidates_ij, nb]
                for idx, p in enumerate(datapoints):
                    residuals_enc_ij[idx] = self._data_enc[p]
                    indices_i.append(p)
                qc = qi - self._kmeans.cluster_centers_[leaf]  # [D]
                dists_ij = self._search(qc[None], residuals_enc_ij)  # [1, num_candidates_ij]
                for d in dists_ij.squeeze(0):
                    dists_i.append(d)
            dists_i = np.array(dists_i)  # [sum(num_candidates_ij for j in range(nprobe))]
            indices_i = np.array(indices_i)

            # choose top-k closest candidates
            pi = dists_i.argsort()[:k]
            dists_i = dists_i[pi]
            indices_i = indices_i[pi]

            # ensure each query has exactly top-k candidates
            k_actual = dists_i.shape[0]
            dists[i] = np.pad(dists_i, (0, k - k_actual), mode='edge')
            indices[i] = np.pad(indices_i, (0, k - k_actual), mode='edge')
        return SearchResult(indices=indices, distances=dists)
