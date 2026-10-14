"""Net flux = 0 on a vessel graph: a weighted least-squares flux network (section 40 of the plan).

Every edge carries one flux q, signed along the edge's own points. Where a speed was read, the edge has a
measurement q_hat = line mass x velocity with its sigma; elsewhere q is free. At every vertex the junction's
reading partitions the arms into lumens (a bifurcation: one group of three; a crossing: two groups of two; a
vessel passing through: one group of two), and what comes into a lumen leaves it:

    sum over the lumen's arms of  s * q  =  0,     s = +1 where the edge's points run out of the vertex, -1 in.

The solve minimises  sum (q - q_hat)^2 / sigma^2  over the measured edges, with every balance allowed to miss by
a share `sigma_flux` of the sum of the absolute fluxes through the lumen (the notebook's SIGMA_FLUX, in the same
unit as its per-junction flux test). The balance is soft because density x velocity is not red-cell flow exactly.

What is done with the one solve, all from its covariance C:

- the signs. A wrong direction is a flipped sign, not noise, and the unit of a direction is a *measurement*: the
  notebook's section 32 reads one direction per whole vessel, so every edge of that vessel turns round together.
  Within every lumen, every set of the measurements touching it is a move to turn those round; the change of the
  minimised chi-square under a move is in closed form from C, with a cost of 2 ln(p / (1 - p)) per measurement
  turned (p: the share of directions that are right, section 33). A lumen's reading may be wrong too: a move
  may set a lumen aside (its balance dropped) at the cost ln(c / (1 - c)) of its reading's confidence c, so that
  a lumen that no cheap flip balances is reported against its reading rather than balanced by turning a right
  measurement round (measured: without this move, two vessels ending near each other and read as one passing
  through had a right measurement turned). The best move is applied while one lowers the total. The pattern
  rule of section 33 (a lumen with every arm measured, all in or all out, cannot be the truth) is reported per
  lumen and never acted on: the junction reading already fails such a lumen (`in and out`, section 27).
- the sizes. Huber weights on the balance residuals (a lumen with a missing arm must not bend its neighbours)
  and on the measurement misfits; per measurement the studentised residual against what the rest of the network
  predicts for it (the leave-one-out identity), which detects a wrong speed only where the lumens around it are
  tight (measured on a plexus with 70% of its edges measured: it does not).
- the tests. Per lumen, its imbalance with its own balance not enforced; for a vertex, every candidate lumen of
  every candidate reading scored from one leave-one-vertex-out solve (a rank-k update of C). These are tests
  for the junction reading (section 27), where a NaN means no test, never a failed one.
- doubtful edges (bridges). An edge with a confidence c < 1 gets a prior on its flux, a pseudo-measurement of 0
  with sigma c / (1 - c) times the graph's typical flux: at c near 0 it carries nothing and the imbalance it
  would have hidden is exposed; at c near 1 it is an ordinary edge. Softening a balance does not express doubt
  about an edge (measured: the edge keeps conducting and the exposing imbalance is divided away).

The chi-squares of two readings are compared only between readings with the same number of balances: one with
fewer always fits at least as well. Flips, impossible lumens and directed cycles are reported, never counted
as measures of p.
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
    measurement: np.ndarray       # per edge: which measurement its direction belongs to (-1: none); the edges of
                                  # one whole vessel share one
    q_hat: np.ndarray             # the measurements as used: turned round where flipped; NaN where not measured
    weight: np.ndarray            # the robust weight each measurement ended with (0..1); 0 where not measured
    flipped: np.ndarray           # bool per edge: the network turned the measured direction round
    studentised: np.ndarray       # per measured edge: (q_hat - what the rest of the network predicts) / spread;
                                  # 0 where not measured or where nothing else reaches the edge
    scale: np.ndarray             # per lumen: what its balance may miss (sigma_flux x the sum of |q| through it)
    balance_weight: np.ndarray    # per lumen: the robust weight its balance ended with
    imbalance: np.ndarray         # per lumen: its net flux with its own balance not enforced, in sigmas of what
                                  # the balance may miss plus the prediction's spread; NaN where nothing but its
                                  # own balance says what its edges carry
    impossible: np.ndarray        # per lumen: every arm measured and all in or all out, after the sign step;
                                  # the reading that made the lumen, or several directions, must be wrong
    disabled: np.ndarray          # per lumen: the sign step set its balance aside as cheaper than any flip that
                                  # would balance it: the directions speak against the reading that made it
    chi2: float                   # the robust fit: Huber losses of the misfits and the balances, the priors on
                                  # doubtful edges, plus the flips' cost
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


def _flip_sets(items, most=6):
    """The sets of measurements touching a lumen that may be turned round together: every nonempty subset while
    there are at most `most`, else every single one, every pair and every complement of a single one."""
    items = list(items)
    if len(items) <= most:
        for size in range(1, len(items) + 1):
            yield from itertools.combinations(items, size)
    else:
        yield from ((item,) for item in items)
        yield from itertools.combinations(items, 2)
        yield from (tuple(other for other in items if other != item) for item in items)


def solve_flux_network(q_hat: np.ndarray, sigma_hat: np.ndarray, lumens: list[Lumen], sigma_flux: float = 0.45,
                       p: float = 0.85, robust_rounds: int = 5, huber: float = 3.0, flips: bool = True,
                       measurement: np.ndarray | None = None, edge_confidence: np.ndarray | None = None,
                       edge_scale: np.ndarray | None = None, lumen_confidence: np.ndarray | float | None = None,
                       lumen_group: np.ndarray | None = None, flip_margin: float = 0.0) -> FluxSolution:
    """Solve the flux network.

    q_hat, sigma_hat  per edge: the measured flux and its sigma; NaN where no speed was read.
    lumens            the balances: per vertex and lumen, ((edge, sign), ...) with sign +1 where the edge's points
                      run out of the vertex and -1 where they run in.
    sigma_flux        the share of the sum of the absolute fluxes through a lumen that its balance may miss (the
                      notebook's SIGMA_FLUX, in the same unit as its per-junction flux test).
    p                 the share of measured directions that are right (section 33 measures it without truth).
    robust_rounds     rounds of Huber reweighting of the balances and the measurements.
    flips             whether the sign step is made at all.
    measurement       per edge, which measurement its direction belongs to (the whole vessel of section 32); -1
                      or absent where not measured. By default every measured edge is its own measurement.
    edge_confidence   per edge, a confidence in (0, 1] that the edge exists (a bridge's); below 1 the edge gets
                      a prior of zero flux with sigma c / (1 - c) times `edge_scale`.
    edge_scale        per edge, the flux the prior is scaled by: the flux of the free end the bridge continues
                      (its arm's measurement). Where none is given for a doubtful edge (NaN, 0, or no array), the
                      flux of the vessel it continues stands in (the other arms of its lumens, from a first solve
                      without priors), else its own, else the graph's mean measured flux; the graph's mean alone
                      is wrong at both ends of the range (measured: a right thick bridge at c = 0.2 carried a sixth
                      of its flux, a wrong capillary bridge was barely held back).
    lumen_confidence  per lumen (or one number): the probability that the *structure* that made the lumen is
                      right, never a goodness of fit. Setting the lumen aside costs ln(c / (1 - c)), against
                      2 ln(p / (1 - p)) per measurement turned round (3.5 at p = 0.85). Section 27's
                      `reading_confidence` is a goodness of fit (its median is about 0.65 under calibrated tests,
                      a fifth of it below 0.5) and must not be passed here; what belongs here is: 1 - 1e-6 for a
                      vertex that has one structure (three arms; two pieces meeting end to end in the skeleton),
                      the margin of the best partition over the second best for a vertex of four or more arms
                      (cost = that margin in log units), a bridge's or an extension's confidence for a join it
                      made, and 0.5 (cost 0: no balance) for a join section 32 undid because its sides ran
                      against each other. By default 1 - 1e-6 everywhere: nothing is set aside unless the caller
                      says the structure is in doubt.
    lumen_group       per lumen, an id: lumens with the same id are set aside together at one cost (the two
                      through-lumens of a crossing are one reading, not two).
    flip_margin       a move to turn measurements round is taken only when it lowers the total by more than this
                      (0: whenever it lowers it). A precision knob for the harness to sweep.
    """
    q_hat = np.array(q_hat, dtype=float)
    sigma_hat = np.asarray(sigma_hat, dtype=float)
    count = len(q_hat)
    measured = np.isfinite(q_hat) & np.isfinite(sigma_hat) & (sigma_hat > 0)
    if measurement is None:
        measurement = np.where(measured, np.arange(count), -1)
    measurement = np.where(measured, np.asarray(measurement), -1).astype(int)
    balance = np.zeros((len(lumens), count))
    for row, lumen in enumerate(lumens):
        for edge, sign in lumen:
            balance[row, edge] += sign
    floor = float(np.mean(np.abs(q_hat[measured]))) if measured.any() else 1.0
    prior_precision = np.zeros(count)
    reached = measured | (np.abs(balance).sum(axis=0) > 0)
    if edge_confidence is not None:
        confidence = np.clip(np.asarray(edge_confidence, dtype=float), 1e-6, 1.0)
        doubtful = confidence < 1
        unit = np.full(count, np.nan) if edge_scale is None else np.asarray(edge_scale, dtype=float).copy()
        unknown = doubtful & ~(np.isfinite(unit) & (unit > 0))
        if unknown.any():
            # no scale given for a doubtful edge (a free end whose arm is unmeasured): what the network makes it
            # carry without any prior, from a first solve, else the graph's mean measured flux
            first_precision = np.where(measured, 1 / np.maximum(sigma_hat, 1e-12) ** 2, 0.0)
            first_rows = balance / (sigma_flux * np.maximum(np.abs(balance) @ np.where(measured, np.abs(np.nan_to_num(q_hat)), floor), 1e-9))[:, None]
            first_q, _ = _solve_linear(first_precision * np.nan_to_num(q_hat), first_precision, first_rows, reached)
            for edge in np.flatnonzero(unknown):
                # the flux of the vessel the doubtful edge continues: the other arms of the lumens it is in
                others = [other for lumen in lumens if any(e == edge for e, _ in lumen) for other, _ in lumen if other != edge]
                carried = [abs(first_q[other]) for other in others if abs(first_q[other]) > floor * 1e-3]
                unit[edge] = max(carried) if carried else (abs(first_q[edge]) if abs(first_q[edge]) > floor * 1e-3 else floor)
        prior_precision[doubtful] = 1 / (np.maximum(unit[doubtful], floor * 1e-6) * confidence[doubtful] / (1 - confidence[doubtful])) ** 2
    reached = reached | (prior_precision > 0)
    log_odds = np.log(max(p, 1e-9) / max(1 - p, 1e-9))  # that a direction is right, before the network speaks
    precision_base = np.where(measured, 1 / np.maximum(sigma_hat, 1e-12) ** 2, 0.0)
    weight = measured.astype(float)
    balance_weight = np.ones(len(lumens))
    flipped = np.zeros(count, dtype=bool)
    disabled = np.zeros(len(lumens), dtype=bool)
    if lumen_confidence is None:
        lumen_confidence = 1 - 1e-6
    reading = np.clip(np.broadcast_to(np.asarray(lumen_confidence, dtype=float), len(lumens)), 1e-6, 1 - 1e-6)
    aside_cost = np.log(reading / (1 - reading))  # what setting a lumen aside costs: its structure's log odds
    group = np.arange(len(lumens)) if lumen_group is None else np.asarray(lumen_group)
    members = {g: np.flatnonzero(group == g) for g in set(group.tolist())}  # the lumens set aside together
    edges_of = {}  # measurement -> its edges
    for edge in np.flatnonzero(measured):
        edges_of.setdefault(int(measurement[edge]), []).append(int(edge))

    def lumen_scale(through):
        """What a balance may miss: sigma_flux x the sum of the absolute fluxes of the lumen's arms (the notebook's
        unit), each arm's flux as given in `through`."""
        return sigma_flux * np.maximum(np.abs(balance) @ np.maximum(through, 1e-9), floor * 1e-3)

    def robust_through(q_now, cov_now, precision_now):
        """An arm's flux for sizing its balances: a measured arm counts the smaller of its measurement and what
        the rest of the network predicts for it (the leave-one-out identity), so that a wrong speed cannot widen
        its own lumen's balance to fit itself; an unmeasured arm counts its current flux."""
        leverage = np.where(measured, precision_now * np.diag(cov_now), 0.0)
        left = 1 - leverage
        with np.errstate(divide="ignore", invalid="ignore"):
            without = np.where(left > 1e-6, (q_now - leverage * np.nan_to_num(q_hat)) / np.maximum(left, 1e-12), np.nan_to_num(q_hat))
        return np.where(measured, np.minimum(np.abs(np.nan_to_num(q_hat)), np.abs(without)), np.abs(q_now))

    def solve(scale_now):
        precision = precision_base * weight + prior_precision
        rows = balance * (np.sqrt(balance_weight) / scale_now)[:, None]
        q_now, covariance_now = _solve_linear(precision_base * weight * np.nan_to_num(q_hat), precision, rows, reached)
        return q_now, covariance_now, precision

    # For the signs the balances are sized by the measurements alone (an unmeasured arm at its first-solve flux):
    # the sign step must not feed on its own solution, which on a plexus tightens the balances around a wrong
    # neighbour and turns right measurements round (measured).
    scale = lumen_scale(np.where(measured, np.abs(np.nan_to_num(q_hat)), floor))
    q, covariance, precision = solve(scale)
    scale = lumen_scale(np.where(measured, np.abs(np.nan_to_num(q_hat)), np.abs(q)))
    q, covariance, precision = solve(scale)
    cov = np.nan_to_num(covariance)
    # 1. the signs: sets of measurements touching one lumen, turned round together, in closed form from C; or a
    # lumen set aside (its balance dropped, which removes its row: the fall of the minimised chi-square is its
    # scaled residual squared over 1 - its leverage, the identity for deleting an observation)
    if flips:
        for _ in range(count + len(lumens) + 1):
            # the best single move, a flip set or a lumen set aside, by the fall of the penalised total; which
            # wins in a given case is the model's arithmetic (the balances' tightness, p, the reading's confidence)
            best, best_gain = None, 0.0
            for lumen in lumens:
                touching = sorted({int(measurement[edge]) for edge, _ in lumen if measured[edge]})
                for subset in _flip_sets(touching):
                    S = [edge for item in subset for edge in edges_of[item]]
                    r = precision_base[S] * weight[S] * q_hat[S]
                    # chi2_min = q_hat^T P q_hat - right^T C right; turning S round changes right by -2 r on S
                    delta = 4 * float(r @ q[S]) - 4 * float(r @ cov[np.ix_(S, S)] @ r)
                    cost = 2 * log_odds * sum(-1 if flipped[edges_of[item][0]] else 1 for item in subset)
                    gain = delta + cost
                    if gain < best_gain - 1e-9 and gain < -flip_margin:
                        best, best_gain = S, gain
            rows = balance * (np.sqrt(balance_weight) / scale)[:, None]
            aside = None
            for g, these in members.items():
                these = [number for number in these if not disabled[number]]
                if not these:
                    continue
                # the fall of the minimised chi-square when these rows are deleted together: r^T (I - H)^-1 r
                # over the rows' block, the identity for deleting a set of observations
                X = rows[these]
                hat = X @ cov @ X.T
                residual = X @ q
                inner = np.eye(len(these)) - hat
                # a row of leverage 1 (a lumen with an arm nothing else reaches) makes I - H singular; that row
                # holds no residual and nothing is gained or lost by deleting it, so the pseudo-inverse is exact
                gain = -float(residual @ np.linalg.pinv(inner, rcond=1e-10) @ residual) + float(aside_cost[these].max())
                if gain < best_gain - 1e-9:
                    aside, best_gain = these, gain
            if aside is not None:
                disabled[aside], balance_weight[aside] = True, 0.0
                q, covariance, precision = solve(scale)
                cov = np.nan_to_num(covariance)
                continue
            if best is None:
                break
            q_hat[best] = -q_hat[best]
            flipped[best] = ~flipped[best]
            q = cov @ (precision_base * weight * np.nan_to_num(q_hat))
    # 2. the sizes: now that the signs are settled the balances are sized robustly (a wrong speed must not widen
    # its own lumen's balance to fit itself), then Huber weights on the balances and on the measurements, re-solving
    scale = lumen_scale(robust_through(q, cov, precision))
    for _ in range(max(robust_rounds, 1)):
        q, covariance, precision = solve(scale)
        cov = np.nan_to_num(covariance)
        misfit = np.where(measured, np.abs(q - np.nan_to_num(q_hat)) / np.maximum(sigma_hat, 1e-12), 0.0)
        new_weight = np.where(measured, np.where(misfit > huber, huber / np.maximum(misfit, 1e-12), 1.0), 0.0)
        net = np.abs(balance @ q) / scale
        new_balance_weight = np.where(disabled, 0.0, np.where(net > huber, huber / np.maximum(net, 1e-12), 1.0))
        if np.allclose(new_weight, weight) and np.allclose(new_balance_weight, balance_weight):
            break
        weight, balance_weight = new_weight, new_balance_weight
    sigma = np.where(reached, np.sqrt(np.maximum(np.diag(cov), 0.0)), np.inf)
    # per measured edge: what the rest of the network predicts for it, and the measurement against that
    own = precision_base * weight
    leverage = np.where(measured, own * np.diag(cov), 0.0)
    studentised = np.zeros(count)
    for edge in np.flatnonzero(measured):
        left = 1 - leverage[edge]
        if left < 1e-6:
            continue  # nothing but its own measurement reaches this edge
        without = (q[edge] - leverage[edge] * q_hat[edge]) / left
        spread = np.sqrt(cov[edge, edge] / left + sigma_hat[edge] ** 2)
        studentised[edge] = (q_hat[edge] - without) / spread
    # per lumen: its imbalance with its own balance not enforced (the row's studentised residual, exact at weight 1)
    rows = balance * (np.sqrt(balance_weight) / scale)[:, None]
    row_leverage = np.einsum("ij,jk,ik->i", rows, cov, rows) if len(lumens) else np.zeros(0)
    left = 1 - row_leverage
    with np.errstate(divide="ignore", invalid="ignore"):
        imbalance = np.where(left > 1e-6, (balance @ q) / scale / np.sqrt(np.maximum(left, 1e-12)), np.nan)
    signs = np.sign(np.nan_to_num(q_hat)).astype(int)
    impossible = np.zeros(len(lumens), dtype=bool)
    for row, lumen in enumerate(lumens):
        if len(lumen) >= 2 and all(measured[edge] for edge, _ in lumen):
            outs = [signs[edge] * sign for edge, sign in lumen]
            impossible[row] = abs(sum(outs)) == len(outs)
    chi2 = float(np.sum(np.where(measured, _huber_loss((q - np.nan_to_num(q_hat)) / np.maximum(sigma_hat, 1e-12), huber), 0.0))
                 + np.sum(np.where(disabled, 0.0, _huber_loss((balance @ q) / scale, huber))) + np.sum(prior_precision * q ** 2)
                 + len({int(measurement[edge]) for edge in np.flatnonzero(flipped)}) * 2 * log_odds
                 + float(sum(aside_cost[these].max() for these in members.values() if disabled[these].all())))
    return FluxSolution(q=q, sigma=sigma, covariance=covariance, measured=measured, measurement=measurement,
                        q_hat=q_hat, weight=weight, flipped=flipped, studentised=studentised, scale=scale,
                        balance_weight=balance_weight, imbalance=imbalance, impossible=impossible, disabled=disabled,
                        chi2=chi2, lumens=list(lumens), balance=balance)


def vertex_tests(solution: FluxSolution, vertex_lumens: list[int], candidates: list[list[Lumen]],
                 sigma_flux: float = 0.45) -> list[list[float]]:
    """Score the candidate readings of one vertex from the network solved without that vertex's own balances.

    vertex_lumens  the indices (into solution.lumens) of the balances the vertex has in the solve.
    candidates     the readings to score: per reading, its lumens as `lumens_of` gives them.
    Returns, per reading, one z per lumen: the net flux the rest of the network predicts through it, over the
    spread of that prediction plus what the balance may miss; NaN where the rest of the network says nothing
    about it. These are the `flux` tests of the junction reading (section 27): a lumen fits when |z| is small, a
    NaN is no test (not a failed one), and the reading taken is the most specific whose tests all fit. At the
    notebook's softness the test fails little; it is one test among the geometric ones, not a verdict.

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
                [[0, 2], [1, 3]], a vessel passing through [[0, 1]], a free end with two candidate bridges
                [[0, 1, 2]]: the end's arm and both bridges one lumen, so that the flux splits between them).
    arms        per vertex: its arms in order, each (edge, end) with end 0 where the edge's first point is at the
                vertex (its points run out of the vertex) and 1 where its last point is (they run in).
    A group of one arm gives no balance (a free end); see flux_at_free_ends for what arrives there.
    """
    lumens: list[Lumen] = []
    for vertex, groups in partitions.items():
        for group in groups:
            if len(group) < 2:
                continue
            lumens.append(tuple((arms[vertex][arm][0], 1 if arms[vertex][arm][1] == 0 else -1) for arm in group))
    return lumens


def blame(solution: FluxSolution, threshold: float = 3.0) -> dict:
    """A missing arm, a wrong join and a wrong speed are one signal, an imbalance, and the rule for which it is:

    - a lumen of two arms that is imbalanced (or set aside, or impossible) is a wrong join or a missing arm there:
      it goes to the graph (section 39, step 3), never to the speeds;
    - a measurement is a wrong-speed candidate only when its studentised residual is beyond `threshold` AND the
      edge sits in lumens of three or more arms at both of its ends that are imbalanced beyond `threshold`
      (a wrong speed shows at both ends of its edge; a missing arm shows at one lumen only).
    Returns {"lumens": [lumen numbers to send to the graph], "speeds": [edges whose speed is in doubt]}; a
    measurement whose residual is large but which the rule does not blame is left to the lumens around it
    (measured: a trunk with one daughter missing from the graph had its right speed weighted down to 0.61 and
    reported at 5 sigma; this rule sends that to the lumen instead). The speeds list is a report, not a
    finding: on trees with a tenth of the speeds three times too fast it stays empty at every softness the plan
    uses (measured by the fifth review), and the plan does not claim to detect wrong speeds."""
    lumens_of_edge: dict[int, list[int]] = {}
    for number, lumen in enumerate(solution.lumens):
        for edge, _ in lumen:
            lumens_of_edge.setdefault(edge, []).append(number)
    off = np.zeros(len(solution.lumens), dtype=bool)
    for number, lumen in enumerate(solution.lumens):
        z = solution.imbalance[number]
        off[number] = solution.disabled[number] or solution.impossible[number] or (np.isfinite(z) and abs(z) > threshold)
    to_graph = [number for number, lumen in enumerate(solution.lumens) if off[number] and len(lumen) <= 2]
    speeds = []
    for edge in np.flatnonzero(solution.measured):
        if abs(solution.studentised[edge]) <= threshold:
            continue
        mine = lumens_of_edge.get(int(edge), [])
        wide = [number for number in mine if len(solution.lumens[number]) >= 3]
        if len(wide) >= 2 and all(off[number] for number in wide):
            speeds.append(int(edge))
        else:
            to_graph += [number for number in mine if off[number] and number not in to_graph]
    return {"lumens": sorted(to_graph), "speeds": speeds}


def flux_at_free_ends(solution: FluxSolution, ends: list[tuple[int, int]]) -> list[tuple[float, float]]:
    """The flux that leaves the graph at every free end: per (edge, end) of a vertex with one arm, the flux along
    the edge's points towards that end (positive: blood arrives there and is lost to the graph; negative: it
    appears there) with its sigma. A large one ranks the end for bridging, or marks a vessel diving out of focus."""
    out = []
    for edge, end in ends:
        sign = 1 if end == 1 else -1  # the edge's points run into the vertex at its last end
        out.append((float(sign * solution.q[edge]), float(solution.sigma[edge])))
    return out


def directed_cycles(edges: list[tuple[int, int]], direction: np.ndarray, confidence: np.ndarray | None = None,
                    cutoff: float = 3.0) -> list[list[int]]:
    """Where the digraph has directed cycles: blood does not go round a loop, so each one holds a wrong direction
    or a crossing read as one lumen. `edges` are (first vertex, last vertex) per edge, `direction` +1 / -1 / 0 per
    edge; with `confidence` (|q| / sigma per edge) only edges at or above `cutoff` take part, so that a cycle is
    never made of directions that are noise (unmeasured edges of a plexus sit near 0.7). Returns one cycle (a list
    of edges) per strongly connected component that holds one, found by Tarjan's algorithm (linear) and a
    breadth-first search inside the component; the cycles are not enumerated, which would be exponential on a
    plexus. A cycle is a report: it lowers nothing by itself and measures nothing."""
    out: dict[int, list[tuple[int, int]]] = {}
    for number, ((first, last), way) in enumerate(zip(edges, direction)):
        if confidence is not None and confidence[number] < cutoff:
            continue
        if way > 0:
            out.setdefault(first, []).append((last, number))
        elif way < 0:
            out.setdefault(last, []).append((first, number))
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
        start = min(component)
        if not (len(component) > 1 or any(other == start for other, _ in out.get(start, []))):
            continue
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


def compare_readings(q_hat, sigma_hat, lumens_a: list[Lumen], lumens_b: list[Lumen], sigma_flux=0.45, **options) -> float:
    """chi2 under reading B minus chi2 under reading A, for two readings with the same number of balances:
    positive, A balances better. Readings with different numbers of balances are refused (ValueError): the one
    with fewer always fits at least as well, so the difference would only say which is looser. This is a
    diagnostic (the log-determinant terms of a likelihood ratio are not in it); the junction reading uses the
    tests of vertex_tests."""
    if len(lumens_a) != len(lumens_b):
        raise ValueError(f"readings with {len(lumens_a)} and {len(lumens_b)} balances cannot be compared by chi-square: "
                         "the looser one always fits at least as well; use the imbalances as tests instead")
    a = solve_flux_network(q_hat, sigma_hat, lumens_a, sigma_flux, **options)
    b = solve_flux_network(q_hat, sigma_hat, lumens_b, sigma_flux, **options)
    return b.chi2 - a.chi2
