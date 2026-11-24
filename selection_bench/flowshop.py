"""Taillard's complete 120 permutation flow-shop instances; minimum makespan.

OR-Library supplies the original seeds. Brunner v13 provides frozen best-known
schedules (Zenodo 24342515); Gmys (Zenodo 4542886) identifies proven optima.
"""
import csv
from hashlib import sha256
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

import numpy as np


ASSETS = {
    "flowshop2.txt": (
        "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/flowshop2.txt",
        "616fa752381d9f4757a07abe0662b699fa6b69fefd1b354724825525c2e73d7b",
    ),
    "Taillard_UB_Schedules_v13.csv": (
        "https://zenodo.org/records/24342515/files/Taillard_UB_Schedules_v13.csv?download=1",
        "b69504914a498f827a8e256d85761ddefb8e0c885b055f0a61c623699efa4b84",
    ),
    "gmys-2021.csv": (
        "https://zenodo.org/records/4542886/files/Taillard_UB_Schedules.csv?download=1",
        "d316dd99d7939afe5143f4357c1804720c3341ce8f45dc5c83dc37f50670db28",
    ),
}


def prepare(cache: Path):
    folder = Path(cache) / "flowshop"
    folder.mkdir(parents=True, exist_ok=True)
    for filename, (url, checksum) in ASSETS.items():
        path = folder / filename
        if path.is_file() and sha256(path.read_bytes()).hexdigest() == checksum:
            continue
        with urlopen(url, timeout=45) as response:
            raw = response.read(200_000)
        if sha256(raw).hexdigest() != checksum:
            raise ValueError(f"Upstream digest differs for {filename}")
        with NamedTemporaryFile(dir=folder, delete=False) as f:
            f.write(raw)
        Path(f.name).replace(path)


def processing_times(n, m, seed):
    """Machine-major uniform draws from Taillard's Park–Miller seed generator."""
    values = np.empty((m, n), dtype=np.int64)
    modulus = 2_147_483_647
    for row in values:
        for job in range(n):
            seed = (16_807 * seed) % modulus
            row[job] = 1 + (99 * seed) // modulus
    return values


class FlowShopCase:
    minimize = True
    feature_names = (
        "head_upstream_load", "head_downstream_load", "tail_upstream_load",
        "tail_downstream_load", "workload_position_trend", "upstream_bias_position_trend",
        "neighbor_workload_difference", "neighbor_upstream_bias_difference",
    )

    def __init__(self, name, processing, reference, lower_bound, proven):
        self.name = name
        self.processing = np.asarray(processing, dtype=np.int64)
        self.machines, self.jobs = self.processing.shape
        self.reference = float(reference)
        self.lower_bound = float(lower_bound)
        self.reference_kind = "proven_optimum" if proven else "best_known"
        split = self.machines // 2
        self._up = self.processing[:split].mean(axis=0) / 99
        self._down = self.processing[split:].mean(axis=0) / 99
        self._load = self.processing.mean(axis=0) / 99
        self._bias = self._up - self._down
        self._position = np.linspace(-1, 1, self.jobs) / self.jobs

    def initial(self, rng):
        return rng.permutation(self.jobs)

    def propose(self, best, rng):
        """One random transposition or insertion; exactly two distinct positions."""
        order = np.array(best, copy=True)
        a, b = rng.choice(self.jobs, 2, replace=False)
        if rng.integers(2):
            order[a], order[b] = order[b], order[a]
        else:
            job = order[a]
            if a < b:
                order[a:b] = order[a+1:b+1]
            else:
                order[b+1:a+1] = order[b:a]
            order[b] = job
        return order

    def _order(self, solution):
        order = np.asarray(solution)
        if (order.shape != (self.jobs,) or order.dtype.kind not in "iu"
                or order.min() < 0 or order.max() >= self.jobs
                or len(np.unique(order)) != self.jobs):
            raise ValueError(f"{self.name} expects a permutation of 0..{self.jobs - 1}")
        return order

    def features(self, solution):
        order = self._order(solution)
        width = max(1, self.jobs // 4)
        head, tail = order[:width], order[-width:]
        load, bias = self._load[order], self._bias[order]
        return np.array([
            self._up[head].mean(), self._down[head].mean(),
            self._up[tail].mean(), self._down[tail].mean(),
            load @ self._position, bias @ self._position,
            np.abs(np.diff(load)).mean(), np.abs(np.diff(bias)).mean(),
        ])

    def evaluate(self, solution):
        order = self._order(solution)
        times = self.processing[:, order]
        # For each machine, finish[j] = t[j]+max(previous_machine[j], finish[j-1]).
        # Prefix sums and a cumulative maximum give that same recurrence for all jobs.
        sums = np.cumsum(times, axis=1)
        finish = sums[0]
        for machine in range(1, self.machines):
            finish = sums[machine] + np.maximum.accumulate(finish - (sums[machine] - times[machine]))
        return float(finish[-1])


def load_cases(cache: Path):
    folder = Path(cache) / "flowshop"
    for filename, (_, checksum) in ASSETS.items():
        if sha256((folder/filename).read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Incorrect cached {filename}; run prepare")
    seeds = {}
    for line in (folder/"flowshop2.txt").read_text(encoding="ascii").splitlines():
        size = re.fullmatch(r"\s*(\d+)\s+jobs\s+(\d+)\s+machines\s*", line, flags=re.I)
        row = re.fullmatch(r"\s*(\d+)\s+(ta\d{3})\s*", line)
        if size:
            n, m = map(int, size.groups())
        if row:
            seeds[row[2]] = (n, m, int(row[1]))
    with (folder/"Taillard_UB_Schedules_v13.csv").open(encoding="utf-8-sig", newline="") as stream:
        best = {r["Name"].lower(): r for r in csv.DictReader(stream)}
    with (folder/"gmys-2021.csv").open(encoding="utf-8-sig", newline="") as stream:
        earlier = {"ta"+r["Ta"]: r for r in csv.DictReader(stream)}
    names = [f"ta{i:03d}" for i in range(1, 121)]
    if set(names) != seeds.keys() or set(names) != best.keys() or set(names) != earlier.keys():
        raise ValueError("Incomplete or unmatched Taillard suite")
    cases = []
    for name in names:
        n, m, seed = seeds[name]
        row, proof = best[name], earlier[name]
        for reference in (row, proof):
            if (int(reference["n"]), int(reference["m"])) != (n, m):
                raise ValueError(f"Size mismatch on {name}")
        optimum = proof["Optimal"] == "1" and int(proof["LB"]) == int(row["UB"])
        case = FlowShopCase(name, processing_times(n, m, seed), int(row["UB"]), int(proof["LB"]), optimum)
        for reference in (row, proof):
            witness = [int(j)-1 for j in reference["Permutation"].split()]
            if case.evaluate(witness) != int(reference["UB"]):
                raise ValueError(f"Cannot reproduce reference schedule for {name}")
        cases.append(case)
    return cases
