"""Net flux = 0 on a vessel graph: a weighted least-squares flux network (section 40 of the plan).

Every edge carries one flux q, signed along the edge's own points. Where a speed was read, the edge has a
measurement q_hat = line mass x velocity with its sigma; elsewhere q is free. At every vertex the junction's
reading partitions the arms into lumens (a bifurcation: one group of three; a crossing: two groups of two; a
vessel passing through: one group of two), and what comes into a lumen leaves it:

    sum over the lumen's arms of  s * q  =  0,     s = +1 where the edge's points run out of the vertex, -1 in.

The solve minimises  sum (q - q_hat)^2 / sigma^2  over the measured edges, with every balance allowed to miss by
a share `sigma_flux` of the lumen's mean absolute flux (the balance is soft: density x velocity is not red-cell
flow exactly). Measurements are reweighted iteratively, so that one wrong speed does not spoil a whole tree.

What comes out, per edge: a flux (measured or inferred) with its sigma, hence a direction with a confidence;
per lumen: its imbalance in sigmas, measured with the network solved without that lumen's own balance (an
imbalance read from a solve that enforced the balance is pulled towards zero by construction); and the chi-square
of the fit. The imbalance is meant to enter the junction reading as one test among its others (the notebook's
section 27 scores every partition by its tests and takes the most specific reading that fits); the chi-squares of
two readings are compared only between readings with the same number of balances, because a reading with fewer
balances always fits at least as well.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Lumen = tuple[tuple[int, int], ...]  # ((edge, sign), ...): the arms of one lumen at one vertex, sign +1 out / -1 in


@dataclass
class FluxSolution:
    q: np.ndarray                 # flux per edge, signed along the edge's points
    sigma: np.ndarray             # its sigma per edge (inf for an edge no balance reaches and no speed was read on)
    measured: np.ndarray          # bool per edge: a speed was read on it
    weight: np.ndarray            # the robust weight each measurement ended with (0..1); 0 where not measured
    flipped: np.ndarray           # bool per edge: the network found the measured direction wrong and turned it round
    imbalance: np.ndarray         # per lumen: its net flux with the network solved without this lumen's own
                                  # balance, in sigmas of what the balance may miss plus the prediction's spread;
                                  # NaN where nothing but its own balance says what its edges carry
    chi2: float                   # the fit: sum of weighted squared misfits plus the balances' squared misses
    lumens: list[Lumen] = field(default_factory=list)

    @property
    def direction(self) -> np.ndarray:
        """+1 where the flux runs along the edge's points, -1 against, 0 where nothing is known."""
        return np.where(np.isfinite(self.sigma) & (self.sigma > 0), np.sign(self.q), 0).astype(int)

    @property
    def direction_confidence(self) -> np.ndarray:
        """How many sigmas the flux is from zero: the confidence of the direction."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.isfinite(self.sigma) & (self.sigma > 0), np.abs(self.q) / self.sigma, 0.0)


def _lumen_scale(lumen: Lumen, q_hat: np.ndarray, measured: np.ndarray, floor: float) -> float:
    """What a lumen's balance may miss is a share of its mean absolute flux; where nothing on it is measured,
    the mean over the whole graph stands in (so that the balance still ties the free edges to each other)."""
    own = [abs(q_hat[edge]) for edge, _ in lumen if measured[edge]]
    return max(float(np.mean(own)) if own else floor, 1e-9)


def _solve_linear(q_hat, sigma_hat, measured, weight, balance, scale, reached):
    """One weighted least-squares solve: minimise |W (q - q_hat)|^2 + |B q / scale|^2. The normal equations are
    (W^2 + B^T B) q = W^2 q_hat; the covariance is the inverse of that matrix. Returns q, the covariance (NaN off
    the reached edges), the measurements' leverages h (how much of its own measurement each edge's flux is) and the
    balances' leverages (how much of its own balance each lumen's net flux is)."""
    count = len(q_hat)
    precision = np.where(measured, weight / np.maximum(sigma_hat, 1e-12) ** 2, 0.0)
    normal = np.diag(precision) + (balance / scale[:, None]).T @ (balance / scale[:, None])
    right = precision * np.where(measured, q_hat, 0.0)
    inside = np.flatnonzero(reached)
    q = np.zeros(count)
    covariance = np.full((count, count), np.nan)
    if len(inside):
        sub = normal[np.ix_(inside, inside)]
        # a lumen made only of free edges leaves a null space (any common flux round a free loop): regularise it
        # very weakly, so that the solve picks the smallest such flux and reports a large sigma for it
        sub = sub + np.eye(len(inside)) * 1e-12 * max(np.trace(sub) / len(inside), 1e-12)
        inverse = np.linalg.pinv(sub)
        q[inside] = inverse @ right[inside]
        covariance[np.ix_(inside, inside)] = inverse
    leverage = np.where(measured, precision * np.nan_to_num(np.diag(covariance)), 0.0)
    rows = balance / scale[:, None]
    row_leverage = np.einsum("ij,jk,ik->i", rows, np.nan_to_num(covariance), rows) if len(rows) else np.zeros(0)
    return q, covariance, leverage, row_leverage


def solve_flux_network(q_hat: np.ndarray, sigma_hat: np.ndarray, lumens: list[Lumen], sigma_flux: float = 0.45,
                       p: float = 0.85, robust_rounds: int = 5, huber: float = 3.0) -> FluxSolution:
    """Solve the flux network.

    q_hat, sigma_hat  per edge: the measured flux and its sigma; NaN where no speed was read.
    lumens            the balances: per vertex and lumen, ((edge, sign), ...) with sign +1 where the edge's points
                      run out of the vertex and -1 where they run in.
    sigma_flux        the share of a lumen's mean absolute flux that its balance may miss.
    p                 the share of measured directions that are right (the notebook's section 33 measures it
                      without truth). A wrong direction is a flipped sign, not Gaussian noise: so a measurement is
                      taken to be either as read (probability p) or turned round (1 - p), and the network decides
                      between the two from what the rest of the graph says the edge carries (its leave-one-out
                      prediction). Measurements that the network turns round are reported as `flipped`.
    robust_rounds     rounds of reweighting after that: a measurement that is still off by more than `huber`
                      sigmas is weighted down by huber / its misfit (Huber's weights), and the solve is repeated.

    The problem is linear (see _solve_linear). An edge that no balance reaches and no speed was read on makes the
    matrix singular: it is given a flux of zero with an infinite sigma, and the rest is solved.
    """
    q_hat = np.array(q_hat, dtype=float)
    sigma_hat = np.asarray(sigma_hat, dtype=float)
    count = len(q_hat)
    measured = np.isfinite(q_hat) & np.isfinite(sigma_hat) & (sigma_hat > 0)
    floor = float(np.mean(np.abs(q_hat[measured]))) if measured.any() else 1.0
    balance = np.zeros((len(lumens), count))
    scale = np.zeros(len(lumens))
    for row, lumen in enumerate(lumens):
        scale[row] = sigma_flux * _lumen_scale(lumen, q_hat, measured, floor)
        for edge, sign in lumen:
            balance[row, edge] += sign
    reached = measured | (np.abs(balance).sum(axis=0) > 0)
    weight = measured.astype(float)
    flipped = np.zeros(count, dtype=bool)
    prior = np.log(max(p, 1e-9) / max(1 - p, 1e-9))  # log odds that a direction is right, before the network speaks
    # 1. the signs: turn a measurement round where the rest of the network says so more strongly than p says not
    for _ in range(count + 1):
        q, covariance, leverage, row_leverage = _solve_linear(q_hat, sigma_hat, measured, weight, balance, scale, reached)
        changed = False
        for edge in np.flatnonzero(measured):
            if 1 - leverage[edge] < 1e-6:
                continue  # nothing but its own measurement reaches this edge: the network has no opinion
            without = (q[edge] - leverage[edge] * q_hat[edge]) / (1 - leverage[edge])  # the leave-one-out prediction
            spread = np.sqrt(covariance[edge, edge] / (1 - leverage[edge]) + sigma_hat[edge] ** 2)
            size = abs(q_hat[edge])
            as_read, turned = (without - np.sign(q_hat[edge]) * size) ** 2, (without + np.sign(q_hat[edge]) * size) ** 2
            odds = prior + (turned - as_read) / (2 * spread ** 2)  # log odds for the direction as read
            if odds < 0:
                q_hat[edge], flipped[edge], changed = -q_hat[edge], not flipped[edge], True
                break  # one at a time: the others are judged again against the corrected network
        if not changed:
            break
    # 2. the sizes: Huber reweighting of the measurements
    for _ in range(max(robust_rounds, 1)):
        q, covariance, leverage, row_leverage = _solve_linear(q_hat, sigma_hat, measured, weight, balance, scale, reached)
        misfit = np.where(measured, np.abs(q - np.where(measured, q_hat, 0.0)) / np.maximum(sigma_hat, 1e-12), 0.0)
        new_weight = np.where(measured, np.where(misfit > huber, huber / np.maximum(misfit, 1e-12), 1.0), 0.0)
        if np.allclose(new_weight, weight):
            break
        weight = new_weight
    sigma = np.where(reached, np.sqrt(np.maximum(np.nan_to_num(np.diag(covariance)), 0.0)), np.inf)
    # Leave-one-lumen-out: the fitted net flux of a balance row, divided by its own scale and by sqrt(1 - h) with h
    # the row's leverage, is what the rest of the network predicts for that lumen when its balance is not enforced,
    # in sigmas of (what the balance may miss, plus the prediction's spread). Where the leverage is 1 nothing but the
    # lumen's own balance says what its edges carry, and there is no test to make.
    left = 1 - row_leverage
    with np.errstate(divide="ignore", invalid="ignore"):
        imbalance = np.where(left > 1e-6, (balance @ q) / scale / np.sqrt(np.maximum(left, 1e-12)), np.nan)
    chi2 = float(np.sum(np.where(measured, weight * ((q - np.where(measured, q_hat, 0.0)) / np.maximum(sigma_hat, 1e-12)) ** 2, 0.0))
                 + np.sum(((balance @ q) / scale) ** 2) + flipped.sum() * 2 * prior)  # a flip costs what p says it costs
    return FluxSolution(q=q, sigma=sigma, measured=measured, weight=weight, flipped=flipped, imbalance=imbalance,
                        chi2=chi2, lumens=list(lumens))


def lumens_of(partitions: dict[int, list[list[int]]], arms: dict[int, list[tuple[int, int]]]) -> list[Lumen]:
    """Turn junction readings into balances.

    partitions  per vertex: its reading, a list of groups of arm numbers (a bifurcation [[0, 1, 2]], a crossing
                [[0, 2], [1, 3]], a vessel passing through [[0, 1]]).
    arms        per vertex: its arms in order, each (edge, end) with end 0 where the edge's first point is at the
                vertex (its points run out of the vertex) and 1 where its last point is (they run in).
    A group of one arm gives no balance (a free end).
    """
    lumens: list[Lumen] = []
    for vertex, groups in partitions.items():
        for group in groups:
            if len(group) < 2:
                continue
            lumens.append(tuple((arms[vertex][arm][0], 1 if arms[vertex][arm][1] == 0 else -1) for arm in group))
    return lumens


def directed_cycles(edges: list[tuple[int, int]], direction: np.ndarray) -> list[list[int]]:
    """The directed cycles of a digraph: blood does not go round a loop, so each one holds a wrong direction or a
    crossing read as one lumen. `edges` are (first vertex, last vertex) per edge, `direction` +1 / -1 / 0 per edge
    (0: the edge takes part in no cycle). Returns one list of edges per elementary directed cycle, found by a
    depth-first search from every vertex (fine for graphs of hundreds of edges)."""
    out: dict[int, list[tuple[int, int]]] = {}
    for number, ((first, last), way) in enumerate(zip(edges, direction)):
        if way > 0:
            out.setdefault(first, []).append((last, number))
        elif way < 0:
            out.setdefault(last, []).append((first, number))
    found: list[list[int]] = []
    keys: set[frozenset] = set()

    def walk(start, vertex, path, on_path):
        """Every directed path from `start` through vertices above it (so that each cycle is found once, from its
        lowest vertex); a step back to `start` closes a cycle."""
        for following, edge in out.get(vertex, []):
            if following == start:
                key = frozenset(path + [edge])
                if key not in keys:
                    keys.add(key)
                    found.append(path + [edge])
            elif following > start and following not in on_path:
                walk(start, following, path + [edge], on_path | {following})

    for start in sorted(out):
        walk(start, start, [], {start})
    return found


def compare_readings(q_hat, sigma_hat, lumens_a: list[Lumen], lumens_b: list[Lumen], sigma_flux=0.45) -> float:
    """The flux's evidence between two readings that have the same number of balances: chi2 under reading B minus
    chi2 under reading A. Positive: A balances better. Each is half the log-likelihood ratio in favour of A (for
    Gaussian misfits). Readings with different numbers of balances are refused (ValueError): the one with fewer
    always fits at least as well, so the difference would only say which is looser. Between such readings the
    imbalances are tests, and the junction reading takes the most specific reading that fits (section 27)."""
    if len(lumens_a) != len(lumens_b):
        raise ValueError(f"readings with {len(lumens_a)} and {len(lumens_b)} balances cannot be compared by chi-square: "
                         "the looser one always fits at least as well; use the imbalances as tests instead")
    a = solve_flux_network(q_hat, sigma_hat, lumens_a, sigma_flux)
    b = solve_flux_network(q_hat, sigma_hat, lumens_b, sigma_flux)
    return b.chi2 - a.chi2
