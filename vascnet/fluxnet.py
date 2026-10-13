"""Net flux = 0 on a vessel graph: a weighted least-squares flux network (section 40 of the plan).

Every edge carries one flux q, signed along the edge's own points. Where a speed was read, the edge has a
measurement q_hat = line mass x velocity with its sigma; elsewhere q is free. At every vertex the junction's
reading partitions the arms into lumens (a bifurcation: one group of three; a crossing: two groups of two; a
vessel passing through: one group of two), and what comes into a lumen leaves it:

    sum over the lumen's arms of  s * q  =  0,     s = +1 where the edge's points run out of the vertex, -1 in.

The solve minimises  sum (q - q_hat)^2 / sigma^2  over the measured edges, with every balance allowed to miss by
a share `sigma_flux` of the sum of the absolute fluxes through the lumen (the notebook's SIGMA_FLUX, in the same
unit as its per-junction flux test). The balance is soft because density x velocity is not red-cell flow exactly.

Three things are done with the one solve, all from its covariance C:

- the signs: a wrong direction is a flipped sign, not noise. Within every lumen, every set of its measured arms is
  a proposal to turn those measurements round; the change of the minimised chi-square under a proposal is in
  closed form from C, with a cost of 2 ln(p / (1 - p)) per turned measurement (p: the share of directions that are
  right, the notebook's section 33). The best proposal is applied while one lowers the total. A set within a lumen
  is one move, so two wrong arms at one junction are turned together, where turning one at a time would turn the
  right ones instead.
- the sizes: Huber weights on the balance residuals (a lumen with a missing arm must not bend its neighbours) and
  on the measurement misfits; and per measurement the studentised residual against what the rest of the network
  predicts for it (the leave-one-out identity), which is the wrong-speed detector.
- the tests: per lumen, its imbalance with its own balance not enforced; and for a vertex, every candidate lumen
  of every candidate reading scored from one leave-one-vertex-out solve (a rank-k update of C), so that the
  junction reading can take the most specific reading whose tests fit (the notebook's section 27).

The chi-squares of two readings are compared only between readings with the same number of balances: one with
fewer always fits at least as well. Flips and directed cycles are reported, never counted as measures of p.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

Lumen = tuple[tuple[int, int], ...]  # ((edge, sign), ...): the arms of one lumen at one vertex, sign +1 out / -1 in


@dataclass
class FluxSolution:
    q: np.ndarray                 # flux per edge, signed along the edge's points
    sigma: np.ndarray             # its sigma per edge (inf for an edge no balance reaches and no speed was read on)
    covariance: np.ndarray        # the solve's covariance of q (NaN off the reached edges)
    measured: np.ndarray          # bool per edge: a speed was read on it
    q_hat: np.ndarray             # the measurements as used: turned round where flipped; NaN where not measured
    weight: np.ndarray            # the robust weight each measurement ended with (0..1); 0 where not measured
    flipped: np.ndarray           # bool per edge: the network turned the measured direction round
    forced: np.ndarray            # per lumen: how many flips it took to make its measured pattern possible (all
                                  # in or all out cannot be); evidence against the reading that made the lumen
    studentised: np.ndarray       # per measurement: (q_hat - what the rest of the network predicts) / its spread;
                                  # 0 where not measured or where nothing else reaches the edge
    scale: np.ndarray             # per lumen: what its balance may miss (sigma_flux x the sum of |q| through it)
    balance_weight: np.ndarray    # per lumen: the robust weight its balance ended with
    imbalance: np.ndarray         # per lumen: its net flux with its own balance not enforced, in sigmas of what
                                  # the balance may miss plus the prediction's spread; NaN where nothing but its
                                  # own balance says what its edges carry
    chi2: float                   # the robust fit: Huber losses of the misfits and the balances, plus the flips' cost
    lumens: list[Lumen] = field(default_factory=list)
    balance: np.ndarray | None = None  # the balance matrix (lumens x edges) of signs

    @property
    def direction(self) -> np.ndarray:
        """+1 where the flux runs along the edge's points, -1 against, 0 where nothing is known."""
        return np.where(np.isfinite(self.sigma) & (self.sigma > 0), np.sign(self.q), 0).astype(int)

    @property
    def direction_confidence(self) -> np.ndarray:
        """How many sigmas the flux is from zero: the confidence of the direction (0 where nothing is known)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.isfinite(self.sigma) & (self.sigma > 0), np.abs(self.q) / self.sigma, 0.0)


def _huber_loss(z, c):
    """Huber's loss of a standardised residual z: z^2 inside c, linear beyond."""
    z = np.abs(z)
    return np.where(z <= c, z ** 2, 2 * c * z - c ** 2)


def _solve_linear(right, precision, rows, reached):
    """Minimise |W (q - q_hat)|^2 + |X q|^2 with `rows` = X (one scaled row per balance). The normal equations are
    (P + X^T X) q = P q_hat = `right`. Returns q and the covariance (the inverse of the normal matrix; NaN off the
    reached edges). A lumen made only of free edges leaves a null space; a very weak ridge picks the smallest flux
    there and gives it a large sigma."""
    count = len(right)
    normal = np.diag(precision) + rows.T @ rows
    q = np.zeros(count)
    covariance = np.full((count, count), np.nan)
    inside = np.flatnonzero(reached)
    if len(inside):
        sub = normal[np.ix_(inside, inside)]
        sub = sub + np.eye(len(inside)) * 1e-12 * max(np.trace(sub) / len(inside), 1e-12)
        inverse = np.linalg.pinv(sub)
        q[inside] = inverse @ right[inside]
        covariance[np.ix_(inside, inside)] = inverse
    return q, covariance


def _flip_sets(arms, most=6):
    """The sets of measured arms of a lumen that may be turned round together: every nonempty subset while the
    lumen has at most `most` measured arms, else every single arm and its complement."""
    arms = list(arms)
    if len(arms) <= most:
        for size in range(1, len(arms) + 1):
            yield from itertools.combinations(arms, size)
    else:
        for arm in arms:
            yield (arm,)
            yield tuple(other for other in arms if other != arm)


def solve_flux_network(q_hat: np.ndarray, sigma_hat: np.ndarray, lumens: list[Lumen], sigma_flux: float = 0.45,
                       p: float = 0.85, robust_rounds: int = 5, huber: float = 3.0, flips: bool = True,
                       lumen_confidence: np.ndarray | None = None) -> FluxSolution:
    """Solve the flux network.

    q_hat, sigma_hat  per edge: the measured flux and its sigma; NaN where no speed was read.
    lumens            the balances: per vertex and lumen, ((edge, sign), ...) with sign +1 where the edge's points
                      run out of the vertex and -1 where they run in.
    sigma_flux        the share of the sum of the absolute fluxes through a lumen that its balance may miss (the
                      notebook's SIGMA_FLUX, in the same unit as its per-junction flux test). The sum is taken
                      over all the lumen's arms, the unmeasured ones at their fluxes from a first solve.
    p                 the share of measured directions that are right (section 33 measures it without truth).
    robust_rounds     rounds of Huber reweighting of the balances and the measurements.
    flips             whether the sign step is made at all.
    lumen_confidence  optional, per lumen: a confidence in (0, 1] that the lumen exists (a bridge's); its balance
                      is softened by 1 / confidence, so that a doubtful connection barely constrains.
    """
    q_hat = np.array(q_hat, dtype=float)
    sigma_hat = np.asarray(sigma_hat, dtype=float)
    count = len(q_hat)
    measured = np.isfinite(q_hat) & np.isfinite(sigma_hat) & (sigma_hat > 0)
    balance = np.zeros((len(lumens), count))
    for row, lumen in enumerate(lumens):
        for edge, sign in lumen:
            balance[row, edge] += sign
    reached = measured | (np.abs(balance).sum(axis=0) > 0)
    softness = np.ones(len(lumens)) if lumen_confidence is None else 1 / np.clip(np.asarray(lumen_confidence, float), 1e-3, 1)
    floor = float(np.mean(np.abs(q_hat[measured]))) if measured.any() else 1.0
    prior = np.log(max(p, 1e-9) / max(1 - p, 1e-9))  # log odds that a direction is right, before the network speaks
    precision_base = np.where(measured, 1 / np.maximum(sigma_hat, 1e-12) ** 2, 0.0)
    weight = measured.astype(float)
    balance_weight = np.ones(len(lumens))
    flipped = np.zeros(count, dtype=bool)

    def lumen_scale(through):
        """What a balance may miss: sigma_flux x the sum of the absolute fluxes of the lumen's arms (the notebook's
        unit), each arm's flux taken robustly (see robust_through)."""
        return sigma_flux * np.maximum(np.abs(balance) @ np.maximum(through, 1e-9), floor * 1e-3) * softness

    def robust_through(q_now, cov_now, precision_now):
        """An arm's flux for sizing its balances: a measured arm counts the smaller of its measurement and what
        the rest of the network predicts for it (the leave-one-out identity), so that a wrong speed cannot widen
        its own lumen's balance to fit itself; an unmeasured arm counts its current flux."""
        leverage = np.where(measured, precision_now * np.diag(cov_now), 0.0)
        left = 1 - leverage
        with np.errstate(divide="ignore", invalid="ignore"):
            without = np.where(left > 1e-6, (q_now - leverage * np.nan_to_num(q_hat)) / np.maximum(left, 1e-12), np.nan_to_num(q_hat))
        return np.where(measured, np.minimum(np.abs(np.nan_to_num(q_hat)), np.abs(without)), np.abs(q_now))

    # For the signs the balances are sized by the measurements alone (an unmeasured arm at the graph's mean
    # measured flux after a first solve): the sign step must not feed on its own solution, which on a plexus
    # tightens the balances around a wrong neighbour and turns right measurements round.
    scale = lumen_scale(np.where(measured, np.abs(np.nan_to_num(q_hat)), floor))
    for sizing in range(2):
        precision = precision_base * weight
        rows = balance * (np.sqrt(balance_weight) / scale)[:, None]
        q, covariance = _solve_linear(precision * np.nan_to_num(q_hat), precision, rows, reached)
        cov = np.nan_to_num(covariance)
        if sizing == 0:
            scale = lumen_scale(np.where(measured, np.abs(np.nan_to_num(q_hat)), np.abs(q)))
    # 1. the signs: sets of measured arms within one lumen, turned round together; closed form from C.
    # Two kinds of evidence: the balance (soft, Gaussian) and the pattern (hard: within a lumen the blood has to
    # come in and go out, so measured arms all in or all out cannot be the truth, whatever the sizes; this is the
    # rule of the notebook's section 33). A proposal that leaves a lumen impossible is never the end; one that
    # makes a lumen impossible is never taken.
    lumens_of_edge: dict[int, list[int]] = {}
    for row, lumen in enumerate(lumens):
        for edge, _ in lumen:
            lumens_of_edge.setdefault(edge, []).append(row)
    IMPOSSIBLE = 1e6

    def impossible(row, signs):
        """All arms measured, and all in or all out: with an unmeasured arm the blood may pass through it."""
        if not all(measured[edge] for edge, _ in lumens[row]):
            return False
        outs = [signs[edge] * sign for edge, sign in lumens[row]]
        return len(outs) >= 2 and abs(sum(outs)) == len(outs)

    forced = np.zeros(len(lumens), dtype=int)  # per lumen: flips it took to make its pattern possible
    if flips:
        right = precision * np.nan_to_num(q_hat)
        signs = np.sign(np.nan_to_num(q_hat)).astype(int)
        for _ in range(count + 1):
            best, best_gain, best_rows = None, 0.0, ()
            for lumen in lumens:
                arms = [edge for edge, _ in lumen if measured[edge]]
                for subset in _flip_sets(arms):
                    S = list(subset)
                    r = precision[S] * q_hat[S]
                    # chi2_min = q_hat^T P q_hat - right^T C right; turning S round changes right by -2 r on S
                    delta = 4 * float(r @ q[S]) - 4 * float(r @ cov[np.ix_(S, S)] @ r)
                    cost = 2 * prior * (sum(not flipped[e] for e in S) - sum(flipped[e] for e in S))
                    touched = sorted({row for e in S for row in lumens_of_edge[e]})
                    before = sum(impossible(row, signs) for row in touched)
                    signs[S] = -signs[S]
                    after = sum(impossible(row, signs) for row in touched)
                    signs[S] = -signs[S]
                    gain = delta + cost + IMPOSSIBLE * (after - before)
                    if gain < best_gain - 1e-9:
                        best, best_gain, best_rows = S, gain, touched
            if best is None:
                break
            q_hat[best] = -q_hat[best]
            flipped[best] = ~flipped[best]
            signs[best] = -signs[best]
            for row in best_rows:
                forced[row] += len(best) if not impossible(row, signs) and best_gain < -IMPOSSIBLE / 2 else 0
            right = precision * np.nan_to_num(q_hat)
            q = cov @ right
    # 2. the sizes: now that the signs are settled the balances are sized robustly (a wrong speed must not widen
    # its own lumen's balance to fit itself), then Huber weights on the balances and on the measurements, re-solving
    scale = lumen_scale(robust_through(q, cov, precision))
    for _ in range(max(robust_rounds, 1)):
        precision = precision_base * weight
        rows = balance * (np.sqrt(balance_weight) / scale)[:, None]
        q, covariance = _solve_linear(precision * np.nan_to_num(q_hat), precision, rows, reached)
        cov = np.nan_to_num(covariance)
        misfit = np.where(measured, np.abs(q - np.nan_to_num(q_hat)) / np.maximum(sigma_hat, 1e-12), 0.0)
        new_weight = np.where(measured, np.where(misfit > huber, huber / np.maximum(misfit, 1e-12), 1.0), 0.0)
        net = np.abs(balance @ q) / scale
        new_balance_weight = np.where(net > huber, huber / np.maximum(net, 1e-12), 1.0)
        if np.allclose(new_weight, weight) and np.allclose(new_balance_weight, balance_weight):
            break
        weight, balance_weight = new_weight, new_balance_weight
    sigma = np.where(reached, np.sqrt(np.maximum(np.diag(cov), 0.0)), np.inf)
    # per measurement: what the rest of the network predicts for it, and the measurement against that
    leverage = np.where(measured, precision * np.diag(cov), 0.0)
    studentised = np.zeros(count)
    for edge in np.flatnonzero(measured):
        left = 1 - leverage[edge]
        if left < 1e-6:
            continue  # nothing but its own measurement reaches this edge
        without = (q[edge] - leverage[edge] * q_hat[edge]) / left
        spread = np.sqrt(cov[edge, edge] / left + sigma_hat[edge] ** 2)
        studentised[edge] = (q_hat[edge] - without) / spread
    # per lumen: its imbalance with its own balance not enforced (the row's studentised residual)
    rows = balance * (np.sqrt(balance_weight) / scale)[:, None]
    row_leverage = np.einsum("ij,jk,ik->i", rows, cov, rows) if len(lumens) else np.zeros(0)
    left = 1 - row_leverage
    with np.errstate(divide="ignore", invalid="ignore"):
        imbalance = np.where(left > 1e-6, (balance @ q) / scale / np.sqrt(np.maximum(left, 1e-12)), np.nan)
    chi2 = float(np.sum(np.where(measured, _huber_loss((q - np.nan_to_num(q_hat)) / np.maximum(sigma_hat, 1e-12), huber), 0.0))
                 + np.sum(_huber_loss((balance @ q) / scale, huber)) + flipped.sum() * 2 * prior)
    return FluxSolution(q=q, sigma=sigma, covariance=covariance, measured=measured, q_hat=q_hat, weight=weight,
                        flipped=flipped, forced=forced, studentised=studentised, scale=scale,
                        balance_weight=balance_weight, imbalance=imbalance, chi2=chi2, lumens=list(lumens),
                        balance=balance)


def vertex_tests(solution: FluxSolution, vertex_lumens: list[int], candidates: list[list[Lumen]],
                 sigma_flux: float = 0.45) -> list[list[float]]:
    """Score the candidate readings of one vertex from the network solved without that vertex's own balances.

    vertex_lumens  the indices (into solution.lumens) of the balances the vertex has in the solve.
    candidates     the readings to score: per reading, its lumens as `lumens_of` gives them.
    Returns, per reading, one z per lumen: the net flux the rest of the network predicts through it, over the
    spread of that prediction plus what the balance may miss (sigma_flux x the sum of |q| through it); NaN where
    the rest of the network says nothing about it. These are the `flux` tests of the junction reading: a lumen
    fits when |z| is small, and the reading taken is the most specific whose tests all fit (section 27).

    The vertex's own rows are taken out of the one solve by a rank-k update of its covariance (Woodbury), so the
    cost is k x k per vertex, not a solve per candidate.
    """
    cov = np.nan_to_num(solution.covariance)
    q = solution.q
    if vertex_lumens:
        rows = solution.balance[vertex_lumens] * (np.sqrt(solution.balance_weight[vertex_lumens]) / solution.scale[vertex_lumens])[:, None]
        inner = np.eye(len(vertex_lumens)) - rows @ cov @ rows.T
        gain = cov @ rows.T @ np.linalg.pinv(inner)
        cov_without = cov + gain @ rows @ cov
        q_without = q + gain @ (rows @ q)
    else:
        cov_without, q_without = cov, q
    through = np.where(solution.measured, np.minimum(np.abs(np.nan_to_num(solution.q_hat)), np.abs(q_without)), np.abs(q_without))
    scores = []
    for reading in candidates:
        zs = []
        for lumen in reading:
            b = np.zeros(len(q))
            for edge, sign in lumen:
                b[edge] += sign
            spread2 = float(b @ cov_without @ b)
            scale = sigma_flux * max(float(np.abs(b) @ np.maximum(through, 1e-9)), 1e-9)
            if spread2 > 1e6 * scale ** 2 or not np.isfinite(spread2):
                zs.append(np.nan)  # nothing but this lumen's own balance would say what its edges carry
            else:
                zs.append(float(b @ q_without) / np.sqrt(spread2 + scale ** 2))
        scores.append(zs)
    return scores


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


def directed_cycles(edges: list[tuple[int, int]], direction: np.ndarray, confidence: np.ndarray | None = None,
                    cutoff: float = 0.0) -> list[list[int]]:
    """Where the digraph has directed cycles: blood does not go round a loop, so each one holds a wrong direction
    or a crossing read as one lumen. `edges` are (first vertex, last vertex) per edge, `direction` +1 / -1 / 0 per
    edge; with `confidence` (|q| / sigma per edge) only edges at or above `cutoff` take part, so that a cycle is
    never made of directions that are noise. Returns one cycle (a list of edges) per strongly connected component
    that holds one, found by Tarjan's algorithm (linear) and a breadth-first search inside the component; the
    cycles are not enumerated, which would be exponential on a plexus."""
    out: dict[int, list[tuple[int, int]]] = {}
    for number, ((first, last), way) in enumerate(zip(edges, direction)):
        if confidence is not None and confidence[number] < cutoff:
            continue
        if way > 0:
            out.setdefault(first, []).append((last, number))
        elif way < 0:
            out.setdefault(last, []).append((first, number))
    # Tarjan, iterative
    index, low, on_stack, stack, components = {}, {}, set(), [], []
    counter = 0
    for root in list(out):
        if root in index:
            continue
        work = [(root, iter(out.get(root, [])))]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            vertex, following = work[-1]
            advanced = False
            for other, _ in following:
                if other not in index:
                    index[other] = low[other] = counter
                    counter += 1
                    stack.append(other)
                    on_stack.add(other)
                    work.append((other, iter(out.get(other, []))))
                    advanced = True
                    break
                if other in on_stack:
                    low[vertex] = min(low[vertex], index[other])
            if advanced:
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[vertex])
            if low[vertex] == index[vertex]:
                component = set()
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.add(member)
                    if member == vertex:
                        break
                components.append(component)
    cycles = []
    for component in components:
        members = sorted(component)
        start = members[0]
        has_cycle = len(component) > 1 or any(other == start for other, _ in out.get(start, []))
        if not has_cycle:
            continue
        # one cycle through `start`: breadth-first inside the component back to start
        parent = {start: None}
        queue = [start]
        found = None
        while queue and found is None:
            vertex = queue.pop(0)
            for other, edge in out.get(vertex, []):
                if other not in component:
                    continue
                if other == start:
                    found = (vertex, edge)
                    break
                if other not in parent:
                    parent[other] = (vertex, edge)
                    queue.append(other)
        if found is None:
            continue
        path, vertex = [found[1]], found[0]
        while parent[vertex] is not None:
            vertex, edge = parent[vertex]
            path.append(edge)
        cycles.append(path[::-1])
    return cycles


def compare_readings(q_hat, sigma_hat, lumens_a: list[Lumen], lumens_b: list[Lumen], sigma_flux=0.45) -> float:
    """The flux's evidence between two readings that have the same number of balances: chi2 under reading B minus
    chi2 under reading A. Positive: A balances better. Readings with different numbers of balances are refused
    (ValueError): the one with fewer always fits at least as well, so the difference would only say which is
    looser. Between such readings the imbalances are tests (vertex_tests), and the junction reading takes the most
    specific reading that fits (section 27)."""
    if len(lumens_a) != len(lumens_b):
        raise ValueError(f"readings with {len(lumens_a)} and {len(lumens_b)} balances cannot be compared by chi-square: "
                         "the looser one always fits at least as well; use the imbalances as tests instead")
    a = solve_flux_network(q_hat, sigma_hat, lumens_a, sigma_flux)
    b = solve_flux_network(q_hat, sigma_hat, lumens_b, sigma_flux)
    return b.chi2 - a.chi2
