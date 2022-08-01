import numpy as np
import tqdm

from src.utils import cdist_2d_v2 as cdist, get_centroids_distortion, get_norm


class InitMethods:
    RANDOM = 'random'
    KMEANS_PLUS_PLUS = 'k-means++'


class KMeans:
    """
    отличия от sklearn.cluster.KMeans:
    init: в sklearn дефолтное значение 'k-means++', я пока не реализовал только 'random'
    precompute_distances: не делал
    algorithm: не делал, потому что реализовывал один алгоритм
    """
    def __init__(
            self,
            n_clusters: int = 8,
            init: str = InitMethods.KMEANS_PLUS_PLUS,
            n_init: int = 10,
            max_iter: int = 300,
            tol: float = 1e-4,
            random_state: int = 42,
            n_loops: int = 0,
            verbose: bool = False,
            algorithm: str = "lloyd"  # to be compatible with sklearn
    ):
        self.n_clusters = n_clusters
        if init not in [InitMethods.RANDOM, InitMethods.KMEANS_PLUS_PLUS]:
            raise NotImplementedError
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        if n_loops not in [0, 1]:
            raise NotImplementedError
        self.n_loops = n_loops
        self.verbose = verbose
        self.rng = np.random.RandomState(random_state)

        if algorithm not in ["lloyd", "full"]:
            raise NotImplementedError
        self.algorithm = algorithm

        self.cluster_centers_ = None

    def fit(self, data: np.ndarray):
        d = data.shape[1]
        best_score = 1e10
        self.cluster_centers_ = np.zeros((self.n_clusters, d))
        for _ in tqdm.trange(self.n_init, disable=not self.verbose):
            centroids = self._init_centroids(data)
            centroids = self._fit_centroids(data, centroids)
            score = get_centroids_distortion(data=data, centroids=centroids)
            if score < best_score:
                best_score = score
                self.cluster_centers_ = centroids

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        data: [n, D], float32
        return: dists: [n, n_clusters], float32
        """
        return cdist(data, self.cluster_centers_)

    def _init_centroids(self, data: np.ndarray) -> np.ndarray:
        if self.init == InitMethods.RANDOM:
            return self._random_init(data)
        elif self.init == InitMethods.KMEANS_PLUS_PLUS:
            return self._kmeans_plus_plus_init(data)
        else:
            raise NotImplementedError

    def _random_init(self, data: np.ndarray) -> np.ndarray:
        idx = self.rng.randint(0, data.shape[0], size=(self.n_clusters,))
        return data[idx]

    def _kmeans_plus_plus_init(self, data: np.ndarray) -> np.ndarray:
        """
        default init method of sklearn.cluster.KMeans
        http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf

        с такой инициализацией алгоритм сходится значительно лучше
        """
        centroids = np.zeros((self.n_clusters, data.shape[1]))
        n = data.shape[0]
        # choose first centroid
        i = self.rng.randint(0, data.shape[0])
        centroids[0] = data[i]
        a_norm_squared = get_norm(data)  # чтоб не пересчитывать n_clusters - 1 раз

        # choose other k-1 centroids
        k = 1
        while k != self.n_clusters:
            dists = cdist(data, centroids[:k], squared=False, a_norm_squared=a_norm_squared)  # [n, k]
            dists_min = dists.min(1)  # [n]
            p = dists_min / dists_min.sum()  # [n], чем больше расстояние до ближайшего кластера, тем лучше
            j = self.rng.choice(n, p=p)
            centroids[k] = data[j]
            k += 1
        return centroids

    def _fit_centroids(self, data: np.ndarray, centroids_init: np.ndarray) -> np.ndarray:
        """
        return centroids and norm of diff between last two iterations
        """
        centroids = centroids_init
        curr_iter = 0
        curr_norm = 1e10
        while (curr_iter < self.max_iter) and (curr_norm > self.tol):
            centroids_upd = self._step(data, centroids)
            curr_norm = np.square(centroids - centroids_upd).sum() ** 0.5
            curr_iter += 1
            centroids = centroids_upd
        if self.verbose:
            print(f"converged in {curr_iter} iterations with tol {curr_norm}")
        return centroids

    def _step(self, data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        if self.n_loops == 0:
            return self._step_no_loop(data, centroids)
        elif self.n_loops == 1:
            return self._step_1_loop(data, centroids)
        else:
            raise NotImplementedError

    def _step_1_loop(self, data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        centroids_upd = np.zeros_like(centroids)
        dists = cdist(data, centroids)  # [n, k]
        cluster_ids = dists.argmin(1)  # [n]
        for i in range(self.n_clusters):
            mask = cluster_ids == i
            data_i = data[mask]
            if data_i.shape[0] > 0:
                centroids_upd[i] = data_i.mean(0)
        return centroids_upd

    @staticmethod
    def _step_no_loop(data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        centroids_upd = np.zeros_like(centroids)
        dists = cdist(data, centroids)  # [n, k]
        cluster_ids = dists.argmin(1)  # [n]
        pi = np.argsort(cluster_ids)  # [n]
        # list of n_clusters arrays of shape [ni, D], where ni - number of points in i-th cluster
        cluster_ids_unique, indices = np.unique(cluster_ids[pi], return_index=True)
        groups = np.split(data[pi], indices[1:], axis=0)
        for i in range(cluster_ids_unique.shape[0]):
            centroids_upd[cluster_ids_unique[i]] = groups[i].mean(0)
        return centroids_upd
