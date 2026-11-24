"""The complete 15-instance OR-Library uncapacitated warehouse location suite.

Source and format: https://people.brunel.ac.uk/~mastjjb/jeb/orlib/uncapinfo.html
The published references in uncapopt.txt are used exclusively for scoring.
"""
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import tarfile
from tempfile import NamedTemporaryFile
from time import sleep
from urllib.request import urlopen

import numpy as np
from scipy.stats import rankdata


BASE_URL = "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/"
CHECKSUMS = {
    "cap71": "dda7a533a674bd5c82766f58aa3ccb995f9cf7f776210f9aa1a7a49406ded8dc",
    "cap72": "fbbc19aef762f3db1d1ba79500fee3268b36e4d482d6062a81153787b58174e1",
    "cap73": "36ed79748e049e01fe7e2f254aeb62ef64ebef5090caf770083242089352c703",
    "cap74": "a740928ead01307256663517055486660d8303b6c61b1a0b2b1e9f2b2fc3af06",
    "cap101": "7f1d0e32808b8c1bd9e3f49ee42dd758abb5a157da63ed52182f22bd14919ebe",
    "cap102": "52725adf3d065fa7e6f365817c9ba5d9cec1261f93403b30fd15e00dd2cb8b40",
    "cap103": "c22544480223e067fad39960171796d681e44b5e20041c4280779b9d57b65198",
    "cap104": "cc4c0ab975191650dc12d85d7c93b6cc649ec5a448a29218fa39953451e16833",
    "cap131": "3ea70c64d029f21a2065b588c4ffa7b0d60cab6ee82fd7775360e0f7e007b36a",
    "cap132": "f9826bdee15ab1b0594170a9fafbcee7911e8dcf61e441886883fab77a4c1886",
    "cap133": "c200ca2e51cd58b3b3fc5a43c584aa3c4c5b6ddee91fd90983c998ebdb90f402",
    "cap134": "664737506cac1ac7c641e7daf618602e890a53d04cfde31389c98fff38112d7b",
    "capa": "99df07aec953ac1e1d5e63578a0600aa3b899606a6a19fc1dfcf1a24739783f8",
    "capb": "1f35015e05b629877ae805f737c575e50ece0c71d4b818c7b63c0687f14f7728",
    "capc": "0c6e58103427b45c23829ab1a5b9fa92d01a3bfe0bac29085e3246ff23753011",
    "uncapopt": "88136637f2045dfd8b4ab9ba900f618951fde7ea9055f7677ab292098176106f",
}
INSTANCES = tuple(k for k in CHECKSUMS if k != "uncapopt")


def _check(name, content):
    if hashlib.sha256(content).hexdigest() != CHECKSUMS[name]:
        raise ValueError(f"OR-Library checksum mismatch: {name}.txt")
    return content


def prepare(cache: Path):
    """Fetch missing official files under cache/ufl and verify all 16 SHA-256 digests."""
    folder = Path(cache) / "ufl"
    folder.mkdir(parents=True, exist_ok=True)
    for name in CHECKSUMS:
        path = folder / (name + ".txt")
        if path.exists():
            _check(name, path.read_bytes())
            continue
        for attempt in range(3):
            try:
                with urlopen(BASE_URL+path.name, timeout=45) as response:
                    raw = _check(name, response.read())
                break
            except OSError:
                if attempt == 2: raise
                sleep(2**attempt)
        with NamedTemporaryFile(dir=folder, delete=False) as pending:
            pending.write(raw)
        stage = Path(pending.name)
        try:
            stage.replace(path)
        finally:
            stage.unlink(missing_ok=True)


@dataclass(eq=False)
class UFLCase:
    name: str
    reference: float
    fixed_costs: np.ndarray
    service_costs: np.ndarray   # [customer, facility], costs for the customer's entire demand
    _ranks: np.ndarray = field(init=False, repr=False)
    minimize = True
    reference_kind = "proven_optimum"
    feature_names = ("open_fraction", "mean_relative_opening_cost", "first_choice_coverage",
                     "top3_coverage", "mean_relative_best_rank", "p90_relative_best_rank",
                     "top5_backup_coverage")

    def __post_init__(self):
        self.fixed_costs = np.asarray(self.fixed_costs, dtype=float)
        self.service_costs = np.asarray(self.service_costs, dtype=float)
        if (self.fixed_costs.ndim != 1 or self.service_costs.ndim != 2
                or self.service_costs.shape[1] != len(self.fixed_costs)
                or len(self.fixed_costs) < 2 or len(self.service_costs) == 0):
            raise ValueError("Expected costs for at least two warehouses and one customer")
        for costs in (self.fixed_costs, self.service_costs):
            if not np.isfinite(costs).all() or (costs < 0).any():
                raise ValueError("Warehouse and service costs must be finite and nonnegative")
        self._ranks = rankdata(self.service_costs, method="min", axis=1).astype(np.int32)

    def _open(self, solution):
        bits = np.asarray(solution)
        if (bits.shape != self.fixed_costs.shape or not np.isin(bits, (0,1)).all()
                or not np.any(bits)):
            raise ValueError("A solution is a binary vector with at least one open warehouse")
        return bits.astype(bool)

    def evaluate(self, solution):
        opened = self._open(solution)
        return float(self.fixed_costs[opened].sum() + self.service_costs[:, opened].min(axis=1).sum())

    def initial(self, rng):
        # Log-uniform cardinality gives both sparse and dense starts; sites are uniform within size.
        m = len(self.fixed_costs)
        count = min(m, int(np.exp(rng.uniform(0, np.log(m+1)))))
        bits = np.zeros(m, dtype=int)
        bits[rng.choice(m, size=count, replace=False)] = 1
        return tuple(bits.tolist())

    def propose(self, best, rng):
        """Uniformly choose a legal move type and a site or pair, without calling evaluate()."""
        opened = self._open(best)
        on, off = np.flatnonzero(opened), np.flatnonzero(~opened)
        moves = []
        if len(off): moves.extend(("open", "swap"))
        if len(on) > 1: moves.append("close")
        move = rng.choice(moves)
        if move in ("open", "swap"): opened[rng.choice(off)] = True
        if move in ("close", "swap"): opened[rng.choice(on)] = False
        return tuple(opened.astype(int).tolist())

    def features(self, solution):
        """Coverage uses within-customer ordinal ranks, never unqueried objective values."""
        opened = self._open(solution)
        ranks = self._ranks[:, opened]
        nearest = ranks.min(axis=1)
        relative_rank = (nearest - 1) / (len(opened) - 1)
        max_cost = self.fixed_costs.max()
        relative_cost = self.fixed_costs[opened].mean() / max_cost if max_cost else 0.
        return np.array([opened.mean(), relative_cost, (nearest == 1).mean(), (nearest <= 3).mean(),
                         relative_rank.mean(), np.quantile(relative_rank, .9),
                         ((ranks <= 5).sum(axis=1) >= 2).mean()])


def _parse(name, raw, reference):
    words = raw.split()
    m, n = map(int, words[:2])
    split = 2 + 2*m
    if len(words) != split + n*(m+1): raise ValueError(f"Invalid OR-Library data length for {name}")
    # The capacity column can literally be 'capacity' in capa/b/c. Both capacity and
    # customer demand are omitted from UFL; the supplied assignment cost already covers demand.
    fixed = np.array(words[3:split:2], dtype=float)
    customers = np.array(words[split:], dtype=float).reshape(n,m+1)
    return UFLCase(name, reference, fixed, customers[:,1:])


def load_cases(cache: Path):
    """Read verified local inputs without contacting a server."""
    folder = Path(cache)/"ufl"
    def read(name):
        return _check(name, (folder/(name+".txt")).read_bytes())
    refs = {k:float(v) for k,v in (line.split() for line in read("uncapopt").decode('ascii').splitlines()[1:] if line.strip())}
    if set(refs) != set(INSTANCES): raise ValueError("Unexpected UFL optimum table")
    return [_parse(name, read(name), refs[name]) for name in INSTANCES]


def verify_public_solutions(cache: Path):
    """Offline scorer audit with independently published optima, never used by search."""
    archive = Path(cache)/"ufl/ORLIB.tgz"
    if hashlib.sha256(archive.read_bytes()).hexdigest() != "68444896aa2c3e4df184ebb38fa453404b1d05e1e102bcabe2f4ea346580326c":
        raise ValueError("Unexpected RWTH UflLib archive (https://algo.rwth-aachen.de/ufllib/data/ORLIB.tgz)")
    differences = {}
    with tarfile.open(archive) as source:
        witnesses = {Path(m.name).name:m for m in source.getmembers() if "/ORLIB-uncap/" in m.name and m.name.endswith(".txt.opt")}
        for case in load_cases(cache):
            with source.extractfile(witnesses[case.name+".txt.opt"]) as handle:
                tokens = handle.read().split()
            assignments = np.array(tokens[:-1], dtype=int)
            n,m = case.service_costs.shape
            if len(assignments) != n or np.any((assignments<0)|(assignments>=m)):
                raise ValueError(f"Illegal facility assignment for {case.name}")
            opened = np.isin(np.arange(m),assignments)
            score = case.evaluate(opened)
            provided = case.fixed_costs[opened].sum()+case.service_costs[np.arange(n),assignments].sum()
            if max(abs(score-case.reference),abs(score-float(tokens[-1])),abs(score-provided)) > .001:
                raise ValueError(f"Public optimum score mismatch for {case.name}")
            differences[case.name] = score-case.reference
    return differences
