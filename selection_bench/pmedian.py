"""OR-Library uncapacitated p-median. All solutions contain zero-based node IDs.

Official instance format, including the last-edge-wins rule:
https://people.brunel.ac.uk/~mastjjb/jeb/orlib/pmedinfo.html
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


BASE_URL = "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/"
NAMES = tuple(f"pmed{i}" for i in range(1, 41))
# SHA-256 of the original bytes, in pmed1..pmed40, pmedopt order.
HASHES = """
31d18cd12703580dad2429ab91ba640f6197e3af0fac366240884e76b33edf3a
cf3327037f8f60d2207275a20c094d3ccd0d0bda92b918494e205f3857e55c26
e2aaee0b5a9814baeee2946d4e52ca49f285fad1535e5b9b5b7f48876a5f2930
1fc970b17b52c5abd6d7f393ed7e7fa8d5f5f01dd49d3cb1770ee21e41821e18
e97fb7dca5f9cff2662746f7931f99a1ae1dc59b74d622019eaef61ff644fc2d
f0a2e1b40b21f43bf34eaf7b400c6d6f89be50214d9c51d30720e92fc14ea308
29211d102978f8458564fa9104f93035d39fccd4baa8626ffe16aac98e45cf73
f1c2d7fffae2854b95d7c9a8248d188fa113ccb6ca10847b8e3eea770bf2470f
6525a70e683c5b18844112f1ebafd3d112f58b6a2e034647f29c78791f8aed3c
d4e04cf8c8130647eb172cafc85eca3d3dcb2dcc391a56e09a0b9f7f12fb8595
3e6b3a68bab766f2905594141aad1aaf6e677098a7883f45724e2ace27ca09f3
a615bb53ebcd5349caae85691ef5fde2df18d0dcfa5e042cd01e9dcdc1434cf1
aeb09dc827f0df5912b43c58df780322b59ef0517d831050bbc1e58762b28ce4
1b8555432c5358bde35fa1bb059993d83a8533807783481ac2ad0ae106fe2c63
18d7deeabe27d0ec9246283379ac01654b3419136d59bf0dcdc2b016d4e96724
02b00e39f042dd282f5ff290e09b612811d536e5dacaa131d17284126a3e85c4
6f7f1255371888e0722e6bbb7ae08bdaa592c15304cf2dfc1b371f5cf04bab6c
40a34f9a66782f28451f4f2db418f79164aa621f0d319085a02ebe556c5bab3f
4a4007d8e468499312350da2c1c1bfb9b100e9f42d78a93ae5e3427373895d1b
ddf30dbad6a390469d15bd05717c5e21f159f98851f664dbe9f0ec78504cf035
400cb66540a4308bb636ac72456ea4938a50eb5d11c89a4125d9dd957fb78571
1d77f99adfd139e00ba23938f2dc02c33f0ec5a5daa38d8fc4655eeb6a70876e
dd4bd5e125821569dd18beb248ecc282275e50763fd7709019a1ada3dd715000
ab53fad00b1815e5be97fd07e11750c1aa2bbdad30b6a9739ae46c25bbb44d61
1b71ef22f832a2ad9b88b903958586ca0c66a5eadfc25a6ad1d524d9e72d562b
951721fc56b22ae2263aa102abeea8488e3918928c9ce741bbb6226d6432a080
40752239c6d3dd49040559ff473129f3d2b708de1b996a39142bf2c60d45e0d4
138e88b0912ba643cb837e2dc89a85868a3dfb2b7435f6e9e60ffbdc5f02949d
301cea1f3668b7a45ed5a965366a54682382ca8d18915a6bc2e3c2e92dc21085
8d46543da866b5c266e98f786954d6134a76be42801691496a9d8bbfff52d3c1
e33ad4c7ea1c25c3fe58c5eccc0c0c65e1f085354737435b2a6218c7bc66debb
88101a4c42445f9a7cd209460416fb81643802f89731e07c4fd51f28cd09416f
9e42f9d50d137759735a86c68a3b9b7b6fe16f14c6da5e16311ce15fb5793b0e
4d5ee91bc786685d223dd9a03a1e8debbddc321f6c951989ef2ac0f6590b1dac
74de37ca5ff67b581dfef10cf7e3e5e01adf19d874a4a2e80fd431d98b79b913
04b54310893f5b9d5370aaf138d34111897b65f31a072d71e6d4f5f61f00d9c0
0cca0db623e99482e52ce0ad91520ac257fe5c3377707afb8f90349d589fbbb2
2ce983f05d3f0f055fa884b41a85a2ba9711dfdbcde26472f41b44c568dd9cfe
7a979e9aa1a807a386ca9fa2093bfc7ddedfb280e88f06170e5defe7cf1ad06c
da79491cbb797eef79b4014fae77bfbd9429204cc09ec1f2b692b3e142096887
ad5a13a266bf932a551b7fd323f8c5e831b855cdc532b092657d24ee9d0c315c
""".split()
DIGEST = dict(zip((*NAMES, "pmedopt"), HASHES, strict=True))


def _read(cache: Path, name: str) -> bytes:
    data = (cache / "pmedian" / (name + ".txt")).read_bytes()
    if sha256(data).hexdigest() != DIGEST[name]:
        raise ValueError(f"Data checksum mismatch for {name}. Run prepare(cache) to repair.")
    return data


def prepare(cache: Path) -> None:
    """Download missing or damaged files. Valid cached files require no network."""
    directory = Path(cache) / "pmedian"
    directory.mkdir(parents=True, exist_ok=True)

    def fetch(name: str) -> None:
        path = directory / (name + ".txt")
        if path.is_file() and sha256(path.read_bytes()).hexdigest() == DIGEST[name]:
            return
        with urlopen(BASE_URL + path.name, timeout=60) as response:
            data = response.read(1_000_001)
        if sha256(data).hexdigest() != DIGEST[name]:
            raise ValueError(f"Official data changed or download was incomplete: {name}")
        with NamedTemporaryFile(dir=directory, delete=False) as temp:
            temp.write(data)
            temp_path = Path(temp.name)
        try:
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, DIGEST))


@dataclass
class PMedian:
    name: str
    reference: float
    p: int
    distances: np.ndarray
    adjacent: np.ndarray
    degree: np.ndarray
    edge_cost: np.ndarray
    diameter: float

    minimize = True
    feature_names = ("center_degree_mean", "center_degree_sd", "center_incident_cost_mean",
                     "center_pair_edge_fraction", "center_pair_distance_mean", "center_pair_distance_max")

    def _centers(self, solution):
        sites = np.asarray(solution)
        if (sites.shape != (self.p,) or sites.dtype.kind not in "iu" or
                sites.min() < 0 or sites.max() >= len(self.degree) or np.unique(sites).size != self.p):
            raise ValueError(f"Expected {self.p} distinct valid integer center IDs")
        return sites

    def initial(self, rng):
        return tuple(sorted(map(int, rng.choice(len(self.degree), self.p, replace=False))))

    def propose(self, best, rng):
        """Uniform one-center swap. Candidate generation does not score a solution."""
        sites = self._centers(best).copy()
        unused = np.ones(len(self.degree), dtype=bool)
        unused[sites] = False
        sites[rng.integers(self.p)] = rng.choice(np.flatnonzero(unused))
        return tuple(sorted(map(int, sites)))

    def features(self, solution):
        """Read node and center-to-center structure; never assign clients to centers."""
        sites = self._centers(solution)
        i, j = np.triu_indices(self.p, k=1)
        between = self.distances[sites[i], sites[j]]
        degree = self.degree[sites]
        return np.asarray([degree.mean(), degree.std(), self.edge_cost[sites].mean(),
                           self.adjacent[sites[i], sites[j]].mean() if i.size else 0.,
                           between.mean() / self.diameter if i.size else 0.,
                           between.max() / self.diameter if i.size else 0.])

    def evaluate(self, solution):
        sites = self._centers(solution)
        return float(self.distances[sites].min(axis=0).sum(dtype=np.int64))


def _case(name: str, data: bytes, reference: float) -> PMedian:
    rows = np.loadtxt(data.splitlines(), dtype=np.int64, ndmin=2)
    n, m, p = map(int, rows[0])
    if rows.shape != (m + 1, 3) or not 0 < p < n:
        raise ValueError(f"Invalid header in {name}")
    # A dictionary processes duplicate undirected edges in input order, preserving
    # the last weight specified by the OR-Library protocol.
    edges = {}
    for a, b, cost in rows[1:]:
        if not (1 <= a <= n and 1 <= b <= n and cost >= 0 and a != b):
            raise ValueError(f"Invalid edge in {name}")
        u, v = sorted((int(a - 1), int(b - 1)))
        edges[u, v] = cost
    u, v = np.asarray(list(edges)).T
    costs = np.asarray(list(edges.values()))
    graph = csr_matrix((np.tile(costs, 2), (np.r_[u, v], np.r_[v, u])), shape=(n, n))
    # Nonnegative weights: Dijkstra from each vertex gives the same matrix as Floyd.
    distances = dijkstra(graph, directed=False)
    if not np.isfinite(distances).all() or distances.max() > np.iinfo(np.int32).max:
        raise ValueError(f"Unreachable nodes or distance overflow in {name}")
    adjacent = np.zeros((n, n), dtype=bool)
    adjacent[u, v] = adjacent[v, u] = True
    degree = adjacent.sum(axis=1)
    mean_edge_cost = np.asarray(graph.sum(axis=1)).ravel() / degree / max(1, costs.max())
    return PMedian(name, float(reference), p, distances.astype(np.int32), adjacent,
                   degree / degree.max(), mean_edge_cost, max(1., float(distances.max())))


def load_cases(cache: Path) -> list[PMedian]:
    """Read all 40 cases in numeric order. The optimum is for reporting only."""
    cache = Path(cache)
    refs = {name: float(value) for name, value in
            (line.split() for line in _read(cache, "pmedopt").decode("ascii").splitlines()[1:] if line.strip())}
    if set(refs) != set(NAMES):
        raise ValueError("Incomplete official optimum table")
    return [_case(name, _read(cache, name), refs[name]) for name in NAMES]
