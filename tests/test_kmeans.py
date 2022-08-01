import time
import numpy as np
from sklearn.cluster import KMeans

from src.kmeans import KMeans as KMeansMy, InitMethods
from src.utils import get_centroids_distortion


def test_kmeans():
    n = 10000
    d = 32
    kmeans_kwargs = {
        "n_clusters": 10,
        "n_init": 10,
        "max_iter": 300,
        "random_state": 42,
        "tol": 1e-4
    }

    rng = np.random.RandomState(228)
    data = rng.uniform(size=(n, d))

    def run(kmeans_cls, kw):
        print(kw)
        kmeans = kmeans_cls(**kw)
        t0 = time.time()
        kmeans.fit(data)
        print("time elapsed:", time.time() - t0)
        print("error:", get_centroids_distortion(data=data, centroids=kmeans.cluster_centers_))

    print("sklearn:")
    for init in [InitMethods.RANDOM, InitMethods.KMEANS_PLUS_PLUS]:
        run(KMeans, {"init": init, **kmeans_kwargs})
        print("-" * 30)
    print("=" * 50)

    for init in [InitMethods.RANDOM, InitMethods.KMEANS_PLUS_PLUS]:
        for n_loops in [0, 1]:
            run(KMeansMy, {"init": init, "n_loops": n_loops, **kmeans_kwargs})
            print("-" * 30)
