"""The flux network of docs/cyclic_graph_recursion_plan.md, section 40, on a toy graph and on a grid.

Vertices and edges (first vertex -> last vertex; the flux runs along the edge's points):

    0 -e0-> 1 -e1-> 2 -e4-> 5 -e7-> 7        e0 trunk (10), e1 / e2 daughters (6 / 4) of the bifurcation at 1
            |       ^       ^                 e3 -> e5 a second vessel crossing at 2 (2)
           e2      e3      e6                 e6 the daughter e2 rejoining at the confluence 5 (a loop 1-2-5-3-1
            v       |       |                 in the undirected graph, not a directed one)
            3 -e6-> 5      4 -e3-> 2 -e5-> 6
"""

import numpy as np
import pytest

from vascnet.fluxnet import compare_readings, directed_cycles, lumens_of, solve_flux_network, vertex_tests

EDGES = [(0, 1), (1, 2), (1, 3), (4, 2), (2, 5), (2, 6), (3, 5), (5, 7)]
TRUE = np.array([10.0, 6.0, 4.0, 2.0, 6.0, 2.0, 4.0, 10.0])
ARMS = {1: [(0, 1), (1, 0), (2, 0)],   # trunk in, two daughters out
        2: [(1, 1), (4, 0), (3, 1), (5, 0)],
        3: [(2, 1), (6, 0)],
        5: [(4, 1), (6, 1), (7, 0)]}
READING = {1: [[0, 1, 2]], 2: [[0, 1], [2, 3]], 3: [[0, 1]], 5: [[0, 1, 2]]}


def measured(*which, sigma=1.0, flip=()):
    q_hat = np.full(len(EDGES), np.nan)
    sigma_hat = np.full(len(EDGES), np.nan)
    for edge in which:
        q_hat[edge] = -TRUE[edge] if edge in flip else TRUE[edge]
        sigma_hat[edge] = sigma
    return q_hat, sigma_hat


def grid_case(n, seed, measured_share=0.7, wrong_share=0.15, sigma_share=0.15, **options):
    """A plexus: an n x n grid with harmonic flow from one corner to the opposite one, one lumen per interior
    vertex (every arm one lumen, as a vertex read as one junction of four), a share of the edges measured, a
    share of those with the wrong direction, and a relative sigma on the sizes."""
    rng = np.random.default_rng(seed)
    vertex = lambda i, j: i * n + j
    edges = []
    for i in range(n):
        for j in range(n):
            if j + 1 < n:
                edges.append((vertex(i, j), vertex(i, j + 1)))
            if i + 1 < n:
                edges.append((vertex(i, j), vertex(i + 1, j)))
    count, vertices = len(edges), n * n
    laplacian = np.zeros((vertices, vertices))
    for a, b in edges:
        laplacian[a, a] += 1
        laplacian[b, b] += 1
        laplacian[a, b] -= 1
        laplacian[b, a] -= 1
    source, sink = 0, vertices - 1
    inject = np.zeros(vertices)
    inject[source], inject[sink] = 1, -1
    keep = [v for v in range(vertices) if v != sink]
    potential = np.zeros(vertices)
    potential[keep] = np.linalg.solve(laplacian[np.ix_(keep, keep)], inject[keep])
    true = np.array([potential[a] - potential[b] for a, b in edges])
    lumens = []
    for v in range(vertices):
        if v in (source, sink):
            continue
        lumens.append(tuple([(e, 1) for e, (a, b) in enumerate(edges) if a == v]
                            + [(e, -1) for e, (a, b) in enumerate(edges) if b == v]))
    is_measured = rng.random(count) < measured_share
    wrong = rng.random(count) < wrong_share
    q_hat = np.where(is_measured, true * (1 + sigma_share * rng.standard_normal(count)) * np.where(wrong, -1, 1), np.nan)
    sigma_hat = np.where(is_measured, np.abs(true) * sigma_share + 1e-9, np.nan)
    solution = solve_flux_network(q_hat, sigma_hat, lumens, **options)
    return {"edges": edges, "true": true, "measured": is_measured, "wrong": wrong, "solution": solution,
            "as read": float(np.mean(np.sign(q_hat[is_measured]) == np.sign(true[is_measured]))),
            "after": float(np.mean(np.sign(solution.q[is_measured]) == np.sign(true[is_measured]))),
            "all after": float(np.mean(solution.direction == np.sign(true))),
            "flips right": int(np.sum(solution.flipped & wrong)), "flips wrong": int(np.sum(solution.flipped & ~wrong))}


def test_lumens_follow_the_readings():
    lumens = lumens_of(READING, ARMS)
    assert ((0, -1), (1, 1), (2, 1)) in lumens          # the bifurcation: trunk in, daughters out
    assert ((1, -1), (4, 1)) in lumens and ((3, -1), (5, 1)) in lumens  # the crossing: two lumens
    assert len(lumens) == 5


def test_unmeasured_fluxes_are_inferred_from_the_balances():
    q_hat, sigma_hat = measured(0, 1, 3, 7)
    solution = solve_flux_network(q_hat, sigma_hat, lumens_of(READING, ARMS))
    assert np.allclose(solution.q, TRUE, atol=0.3)
    assert (solution.direction == 1).all()
    assert np.isfinite(solution.sigma).all()
    assert (solution.sigma[~solution.measured] > 0).all()
    assert np.nanmax(np.abs(solution.imbalance)) < 0.5
    # the lumen (e3 in, e5 out) at the crossing: e5 is measured nowhere and in no other balance, so without this
    # lumen's own balance nothing says what it carries: no test to make
    lumens = lumens_of(READING, ARMS)
    assert np.isnan(solution.imbalance[lumens.index(((3, -1), (5, 1)))])


def test_a_wrong_direction_is_turned_round_by_the_network():
    q_hat, sigma_hat = measured(0, 1, 2, 3, 4, 5, 7, flip=(1,))
    solution = solve_flux_network(q_hat, sigma_hat, lumens_of(READING, ARMS))
    assert solution.flipped[1] and not solution.flipped[[0, 2, 3, 4, 5, 7]].any()
    assert solution.q[1] > 0 and abs(solution.q[1] - 6.0) < 0.5
    assert solution.direction[6] == 1  # the unmeasured loop edge is inferred the right way as well


@pytest.mark.parametrize("sigma_flux", [0.3, 0.2, 0.15])
def test_two_wrong_directions_at_one_junction_are_turned_round_together(sigma_flux):
    # the trunk and one daughter wrong: a search that turns one measurement at a time turns the three right
    # ones instead and reverses the tree; a set within a lumen is one move, and the pattern rule (all in or all
    # out cannot be, where every arm is measured) says a move is due. With balances no softer than 0.3 of the
    # absolute sum the two are turned together
    q_hat, sigma_hat = measured(0, 1, 2, 3, 4, 5, 7, flip=(0, 1))
    solution = solve_flux_network(q_hat, sigma_hat, lumens_of(READING, ARMS), sigma_flux=sigma_flux)
    assert set(np.flatnonzero(solution.flipped)) == {0, 1}
    assert (np.sign(solution.q) == np.sign(TRUE)).all()


def test_two_wrong_directions_under_the_notebook_softness_are_not_hidden():
    # at 0.45 of the absolute sum one wrong direction is six times likelier a priori than two, and the balances
    # are too soft to pay for the second flip: the network turns one measurement round and leaves the rest as an
    # imbalance it reports, rather than reversing the tree as a one-at-a-time search did
    q_hat, sigma_hat = measured(0, 1, 2, 3, 4, 5, 7, flip=(0, 1))
    solution = solve_flux_network(q_hat, sigma_hat, lumens_of(READING, ARMS), sigma_flux=0.45)
    assert solution.flipped.sum() <= 2
    assert (np.sign(solution.q) == np.sign(TRUE)).sum() >= 5, "no cascade through the right measurements"
    if set(np.flatnonzero(solution.flipped)) != {0, 1}:
        assert np.nanmax(np.abs(solution.imbalance)) > 1.5, "what was not turned round is left visible"
        assert solution.forced.sum() >= 1


def test_a_wrong_direction_on_a_leaf_is_turned_round():
    # e7 leaves the field: it is contradicted at one end only, where it makes the confluence all in
    q_hat, sigma_hat = measured(0, 1, 2, 3, 4, 5, 7, flip=(7,))
    solution = solve_flux_network(q_hat, sigma_hat, lumens_of(READING, ARMS))
    assert set(np.flatnonzero(solution.flipped)) == {7}


def test_a_lone_measurement_is_never_turned_round():
    q_hat, sigma_hat = measured(0, 1, flip=(1,))
    solution = solve_flux_network(q_hat, sigma_hat, [])  # no balances: the network has no opinion
    assert not solution.flipped.any() and solution.q[1] < 0


def test_the_pattern_rule_reports_what_it_forced():
    # two vessels ending near each other, read as one vessel passing through: both measured towards the vertex,
    # all in; the network has to turn one round, and says the lumen cost a flip
    q_hat, sigma_hat = measured(2, 6)
    q_hat[6] = -4.0  # e6 (3 -> 5) read as running towards vertex 3, like e2 (1 -> 3)
    lumens = lumens_of({3: [[0, 1]]}, ARMS)
    solution = solve_flux_network(q_hat, sigma_hat, lumens)
    assert solution.flipped.sum() == 1 and solution.forced[0] == 1


def test_the_right_pairing_at_the_crossing_balances_better():
    q_hat, sigma_hat = measured(1, 3, 4, 5, sigma=0.5)  # 6 and 2 pass through
    right = lumens_of({2: [[0, 1], [2, 3]]}, ARMS)
    wrong = lumens_of({2: [[0, 3], [2, 1]]}, ARMS)   # e1 with e5, e3 with e4
    one_lumen = lumens_of({2: [[0, 1, 2, 3]]}, ARMS)  # all four arms one lumen (a confluence of two into two)
    assert compare_readings(q_hat, sigma_hat, right, wrong) > 0.5
    with pytest.raises(ValueError):
        compare_readings(q_hat, sigma_hat, right, one_lumen)  # fewer balances always fit at least as well
    # two vessels whose fluxes differ by 20% cannot be told apart by a balance that may miss by 45% of the sum
    close = q_hat.copy()
    close[[3, 5]] = 5.0
    assert compare_readings(close, sigma_hat, right, wrong) < 0.5


def test_the_imbalance_is_a_test_of_each_lumen_without_its_own_balance():
    # the same crossing, read three ways; the leave-one-out imbalance says which lumens balance, and the junction
    # reading of section 27 takes the most specific reading whose tests all fit. With the notebook's softness
    # (0.45 of the absolute sum) 6 against 2 is only about one sigma off: the flux test is weak by design
    q_hat, sigma_hat = measured(1, 3, 4, 5, sigma=0.5)
    right = solve_flux_network(q_hat, sigma_hat, lumens_of({2: [[0, 1], [2, 3]]}, ARMS))
    wrong = solve_flux_network(q_hat, sigma_hat, lumens_of({2: [[0, 3], [2, 1]]}, ARMS))
    one_lumen = solve_flux_network(q_hat, sigma_hat, lumens_of({2: [[0, 1, 2, 3]]}, ARMS))
    assert np.nanmax(np.abs(right.imbalance)) < 0.5, "6 in, 6 out and 2 in, 2 out: both lumens balance"
    assert np.nanmin(np.abs(wrong.imbalance)) > 1.0, "6 in, 2 out: neither lumen balances"
    assert np.nanmax(np.abs(one_lumen.imbalance)) < 0.5, "8 in, 8 out balances too: specificity is section 27's job"
    sharper = solve_flux_network(q_hat, sigma_hat, lumens_of({2: [[0, 3], [2, 1]]}, ARMS), sigma_flux=0.15)
    assert np.nanmin(np.abs(sharper.imbalance)) > 2.5, "a tighter balance sees it"


def test_candidate_readings_are_scored_from_one_solve_without_the_vertex():
    # every edge measured; the vertex at the crossing is read as a crossing in the solve, and its three candidate
    # readings are scored with its own balances taken out, so that no candidate is tested under its own assumption
    q_hat, sigma_hat = measured(0, 1, 2, 3, 4, 5, 6, 7, sigma=0.5)
    lumens = lumens_of(READING, ARMS)
    solution = solve_flux_network(q_hat, sigma_hat, lumens)
    own = [number for number, lumen in enumerate(lumens) if {edge for edge, _ in lumen} <= {1, 3, 4, 5}]
    candidates = [lumens_of({2: reading}, ARMS) for reading in ([[0, 1], [2, 3]], [[0, 3], [2, 1]], [[0, 1, 2, 3]])]
    scores = vertex_tests(solution, own, candidates)
    assert max(abs(z) for z in scores[0]) < 0.5, "the crossing's two lumens balance"
    assert min(abs(z) for z in scores[1]) > 1.0, "the wrong pairing's do not"
    assert max(abs(z) for z in scores[2]) < 0.5, "one lumen balances too"
    # the scores do not depend on which reading the solve enforced at the vertex: the vertex's own rows are out
    others = [lumen for number, lumen in enumerate(lumens) if number not in own]
    enforced = solve_flux_network(q_hat, sigma_hat, others + candidates[1])
    again = vertex_tests(enforced, [len(others), len(others) + 1], candidates)
    for one, other in zip(scores, again):
        assert np.allclose(one, other, atol=1e-3, equal_nan=True)


def test_a_wrong_speed_shows_in_the_studentised_residual_not_in_the_imbalance():
    q_hat, sigma_hat = measured(0, 1, 2, 3, 4, 5, 6, 7, sigma=0.5)
    q_hat[0] = 50.0  # the trunk read five times too fast
    solution = solve_flux_network(q_hat, sigma_hat, lumens_of(READING, ARMS))
    assert abs(solution.studentised[0]) > 3, "the trunk's measurement is far from what the rest predicts"
    assert np.abs(solution.studentised[[1, 2, 3, 4, 5, 6, 7]]).max() < 2, "the others are not"
    assert solution.balance_weight[0] < 1, "the bifurcation's balance was weighted down, not the neighbours bent"


@pytest.mark.parametrize("sigma_flux", [0.15, 0.45, 1.0])
def test_balances_are_soft(sigma_flux):
    q_hat, sigma_hat = measured(0, 1, 2, sigma=0.5)
    q_hat[0] = 12.0  # the trunk reads 20% more than its daughters add up to
    solution = solve_flux_network(q_hat, sigma_hat, lumens_of({1: [[0, 1, 2]]}, ARMS), sigma_flux=sigma_flux)
    assert abs(solution.q[0] - 12.0) < abs(12.0 - 10.0)
    assert solution.chi2 > 0


def test_softer_balances_move_the_fluxes_less():
    q_hat, sigma_hat = measured(0, 1, 2, sigma=0.5)
    q_hat[0] = 12.0
    moved = [abs(solve_flux_network(q_hat, sigma_hat, lumens_of({1: [[0, 1, 2]]}, ARMS), sigma_flux=s).q[0] - 12.0)
             for s in (0.1, 0.3, 1.0)]
    assert moved[0] > moved[1] > moved[2]


def test_an_enforced_balance_would_hide_its_own_imbalance():
    q_hat, sigma_hat = measured(0, 1, 2, sigma=0.5)
    q_hat[0] = 14.0  # the trunk reads 14 against daughters of 6 and 4
    solution = solve_flux_network(q_hat, sigma_hat, lumens_of({1: [[0, 1, 2]]}, ARMS))
    scale = 0.45 * (10.0 + 6.0 + 4.0)  # the notebook's unit, the trunk counted at what the daughters predict for it
    expected = (14.0 - 6.0 - 4.0) / np.sqrt(scale ** 2 + 3 * 0.5 ** 2)
    assert abs(abs(solution.imbalance[0]) - expected) < 0.15 * expected
    # with a balance tight enough to have leverage, the fitted net flux is pulled towards zero and the
    # leave-one-out imbalance is not
    tight = solve_flux_network(q_hat, sigma_hat, lumens_of({1: [[0, 1, 2]]}, ARMS), sigma_flux=0.03)
    fitted = (tight.q[0] - tight.q[1] - tight.q[2]) / tight.scale[0]
    assert abs(fitted) < 0.8 * abs(tight.imbalance[0])


def test_directed_cycles():
    ring = [(0, 1), (1, 2), (2, 0)]
    assert [sorted(c) for c in directed_cycles(ring, np.array([1, 1, 1]))] == [[0, 1, 2]]
    assert directed_cycles(ring, np.array([1, 1, -1])) == []
    assert [sorted(c) for c in directed_cycles(ring, np.array([-1, -1, -1]))] == [[0, 1, 2]]
    assert directed_cycles(EDGES, np.ones(len(EDGES), dtype=int)) == []  # the loop is not a directed cycle
    flipped = np.ones(len(EDGES), dtype=int)
    flipped[[2, 6]] = -1  # 1 -> 2 -> 5 -> 3 -> 1
    assert [sorted(c) for c in directed_cycles(EDGES, flipped)] == [[1, 2, 4, 6]]
    # a direction of no confidence takes no part
    assert directed_cycles(EDGES, flipped, confidence=np.array([9, 9, 0.1, 9, 9, 9, 9, 9]), cutoff=1.0) == []
    # one cycle per strongly connected component, not every cycle: a 6 x 6 grid with every edge both ways
    n = 6
    grid = [((i * n + j), (i * n + j + 1)) for i in range(n) for j in range(n - 1)] + \
           [((i * n + j), ((i + 1) * n + j)) for i in range(n - 1) for j in range(n)]
    both = grid + [(b, a) for a, b in grid]
    assert len(directed_cycles(both, np.ones(len(both), dtype=int))) == 1


@pytest.mark.parametrize("n,seed", [(8, 1), (8, 2), (12, 1), (12, 2)])
def test_on_a_plexus_the_network_does_not_make_the_directions_worse(n, seed):
    case = grid_case(n, seed)
    assert case["after"] >= case["as read"], "the measured directions are at least as right as read"
    assert case["flips right"] >= 2 * case["flips wrong"], "the flips are mostly right"
    assert case["all after"] > 0.8
