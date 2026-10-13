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

from vascnet.fluxnet import (compare_readings, directed_cycles, flux_at_free_ends, lumens_of, solve_flux_network,
                             vertex_tests)

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


def test_a_wrong_direction_on_a_leaf_is_turned_round():
    # e7 leaves the field: it is contradicted at one end only, where it makes the confluence all in; the balance
    # there (20 against what 0.45 of 20 may miss) pays for the flip on its own
    q_hat, sigma_hat = measured(0, 1, 2, 3, 4, 5, 7, flip=(7,))
    solution = solve_flux_network(q_hat, sigma_hat, lumens_of(READING, ARMS))
    assert set(np.flatnonzero(solution.flipped)) == {7}


def test_two_wrong_directions_at_one_junction_are_turned_round_together():
    # the trunk and one daughter wrong: a search that turns one measurement at a time turns the three right
    # ones instead and reverses the tree; a set within a lumen is one move, and with the balances at 0.15 of the
    # absolute sum the two are turned together
    q_hat, sigma_hat = measured(0, 1, 2, 3, 4, 5, 7, flip=(0, 1))
    solution = solve_flux_network(q_hat, sigma_hat, lumens_of(READING, ARMS), sigma_flux=0.15)
    assert set(np.flatnonzero(solution.flipped)) == {0, 1}
    assert (np.sign(solution.q) == np.sign(TRUE)).all()


def test_two_wrong_directions_or_a_wrong_join_is_the_models_arithmetic():
    # at 0.3 the same case: two wrong directions (0.15 x 0.15 a priori) against the join at vertex 2 being wrong
    # (0.01): the model sets the join aside and reports it, and no right measurement is turned round. Whether
    # the pair is turned or the join set aside is the arithmetic of the balances' tightness, p and the
    # reading's confidence; what the test guarantees is that the outcome is reported and nothing cascades
    q_hat, sigma_hat = measured(0, 1, 2, 3, 4, 5, 7, flip=(0, 1))
    lumens = lumens_of(READING, ARMS)
    solution = solve_flux_network(q_hat, sigma_hat, lumens, sigma_flux=0.3)
    turned = set(np.flatnonzero(solution.flipped))
    assert turned <= {0, 1}, "no right measurement is turned"
    if turned != {0, 1}:
        assert solution.disabled[lumens.index(((1, -1), (4, 1)))], "the join the wrong daughter passes is set aside"
        assert solution.impossible.any() or np.nanmax(np.abs(solution.imbalance)) > 1.0


def test_two_wrong_directions_under_the_notebook_softness_are_reported_not_hidden():
    # at 0.45 one wrong direction is six times likelier a priori than two and the balances are too soft to pay
    # for the second flip; whatever the network turns, no right measurement cascades, and what is left is an
    # impossible lumen or an imbalance it reports
    q_hat, sigma_hat = measured(0, 1, 2, 3, 4, 5, 7, flip=(0, 1))
    solution = solve_flux_network(q_hat, sigma_hat, lumens_of(READING, ARMS), sigma_flux=0.45)
    assert solution.flipped.sum() <= 2
    assert (np.sign(solution.q) == np.sign(TRUE)).sum() >= 5, "no cascade through the right measurements"
    if set(np.flatnonzero(solution.flipped)) != {0, 1}:
        assert solution.impossible.any() or np.nanmax(np.abs(solution.imbalance)) > 1.5


def test_a_lone_measurement_is_never_turned_round():
    q_hat, sigma_hat = measured(0, 1, flip=(1,))
    solution = solve_flux_network(q_hat, sigma_hat, [])  # no balances: the network has no opinion
    assert not solution.flipped.any() and solution.q[1] < 0


def test_an_impossible_lumen_is_reported_and_no_right_measurement_is_turned_for_it():
    # two vessels ending near each other at vertex 3, read as one vessel passing through: e2 (1 -> 3) dives there
    # and e6 (3 -> 5) is another vessel; both are measured towards vertex 3. The lumen is impossible, and the
    # only move the balances offer is to turn e2 round, which is the right daughter of a correct bifurcation.
    # The network must not take it: the lumen is reported impossible for the junction reading to fail
    q_hat, sigma_hat = measured(0, 1, 2, 3, 4, 5, 6, 7)
    q_hat[6] = -4.0
    lumens = lumens_of(READING, ARMS)
    here = lumens.index(((2, -1), (6, 1)))
    # the join at vertex 3 was made by extending a free end into a line: it carries that extension's confidence
    confidence = np.array([0.999 if len(lumen) >= 3 else 0.99 for lumen in lumens])
    confidence[here] = 0.8
    solution = solve_flux_network(q_hat, sigma_hat, lumens, lumen_confidence=confidence)
    assert not solution.flipped.any()
    assert solution.disabled[here] and solution.impossible[here]
    assert np.abs(solution.imbalance[here]) > 2
    # a join held with more confidence than one direction is not set aside: then the cheaper explanation is a
    # wrong direction, and the network says which
    sure = solve_flux_network(q_hat, sigma_hat, lumens)
    assert not sure.disabled.any() and sure.flipped.sum() == 1


@pytest.mark.parametrize("sigma_flux", [0.3, 0.15])
def test_a_whole_vessel_is_one_measurement_and_turns_round_as_one(sigma_flux):
    # section 32 reads one direction per whole vessel: e1 and e4 are one vessel (1 -> 2 -> 5), read backwards.
    # As two independent measurements the two flips cost twice and the chain is left; as one measurement it is
    # one move, and the balances at vertices 1 and 5 pay for it (at 0.45 of the absolute sum they just fail to:
    # 1.3 sigma at each end against one flip's cost; at 0.3 they do)
    q_hat, sigma_hat = measured(0, 1, 2, 3, 4, 5, 6, 7, flip=(1, 4))
    vessel = np.array([0, 1, 2, 3, 1, 5, 6, 7])  # e1 and e4 share a measurement
    joined = solve_flux_network(q_hat, sigma_hat, lumens_of(READING, ARMS), measurement=vessel, sigma_flux=sigma_flux)
    assert set(np.flatnonzero(joined.flipped)) == {1, 4}
    assert (np.sign(joined.q) == np.sign(TRUE)).all()
    apart = solve_flux_network(q_hat, sigma_hat, lumens_of(READING, ARMS), sigma_flux=sigma_flux)
    assert set(np.flatnonzero(apart.flipped)) != {1, 4} or apart.chi2 > joined.chi2


def test_a_doubtful_edge_carries_little_and_exposes_the_imbalance_it_would_hide():
    # vessel A (edge 0, reads 4 out of vertex 1) is bridged (edge 1) to vessel B (edge 2, reads 1 out of
    # vertex 2): the bridge is wrong, or B is not A's continuation. Through-lumens at both ends
    edges = [(0, 1), (1, 2), (2, 3)]
    lumens = [((0, -1), (1, 1)), ((1, -1), (2, 1))]
    q_hat = np.array([4.0, np.nan, 1.0])
    sigma_hat = np.array([0.3, np.nan, 0.3])
    sure = solve_flux_network(q_hat, sigma_hat, lumens)
    doubt = solve_flux_network(q_hat, sigma_hat, lumens, edge_confidence=np.array([1.0, 0.02, 1.0]))
    assert 2.0 < sure.q[1] < 3.0, "trusted, the bridge carries the mean of the two sides"
    assert abs(doubt.q[1]) < 0.5, "doubted, it carries almost nothing"
    assert np.nanmin(np.abs(doubt.imbalance)) > np.nanmax(np.abs(sure.imbalance)), "and the mismatch is exposed, not divided away"
    assert np.nanmax(np.abs(doubt.imbalance)) > 2.0


def test_rival_bridges_share_one_lumen_and_the_flux_is_not_counted_twice():
    # A's free end (vertex 1) has two candidate bridges, x to B and y to C; the end's arm and both bridges are
    # one lumen, so A's flux splits between them, and a near-worthless rival carries nothing
    edges = [(0, 1), (1, 2), (1, 3)]
    lumens = [((0, -1), (1, 1), (2, 1))]
    q_hat = np.array([4.0, np.nan, np.nan])
    sigma_hat = np.array([0.3, np.nan, np.nan])
    even = solve_flux_network(q_hat, sigma_hat, lumens, edge_confidence=np.array([1.0, 0.5, 0.5]))
    assert abs(even.q[1] + even.q[2] - 4.0) < 0.5 and abs(even.q[1] - even.q[2]) < 0.1
    skewed = solve_flux_network(q_hat, sigma_hat, lumens, edge_confidence=np.array([1.0, 0.9, 0.02]))
    assert skewed.q[1] > 3.0 and abs(skewed.q[2]) < 0.3


def test_the_flux_lost_at_a_free_end_is_reported():
    q_hat, sigma_hat = measured(0, 1, 3, 7)
    solution = solve_flux_network(q_hat, sigma_hat, lumens_of(READING, ARMS))
    lost = flux_at_free_ends(solution, [(7, 1), (5, 1), (0, 0)])  # e7 into vertex 7, e5 into 6, e0 out of 0
    assert abs(lost[0][0] - 10.0) < 0.5 and abs(lost[1][0] - 2.0) < 0.5
    assert abs(lost[2][0] + 10.0) < 0.5, "blood appears at the root: negative"


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
    q_hat, sigma_hat = measured(0, 1, 2, 3, 4, 5, 6, 7, sigma=0.5)
    lumens = lumens_of(READING, ARMS)
    solution = solve_flux_network(q_hat, sigma_hat, lumens)
    own = [number for number, lumen in enumerate(lumens) if {edge for edge, _ in lumen} <= {1, 3, 4, 5}]
    candidates = [lumens_of({2: reading}, ARMS) for reading in ([[0, 1], [2, 3]], [[0, 3], [2, 1]], [[0, 1, 2, 3]])]
    scores = vertex_tests(solution, own, candidates)
    assert max(abs(z) for z in scores[0]) < 0.5, "the crossing's two lumens balance"
    assert min(abs(z) for z in scores[1]) > 1.0, "the wrong pairing's do not"
    assert max(abs(z) for z in scores[2]) < 0.5, "one lumen balances too"
    others = [lumen for number, lumen in enumerate(lumens) if number not in own]
    enforced = solve_flux_network(q_hat, sigma_hat, others + candidates[1])
    again = vertex_tests(enforced, [len(others), len(others) + 1], candidates)
    for one, other in zip(scores, again):
        assert np.allclose(one, other, atol=1e-3, equal_nan=True)


def test_a_wrong_speed_shows_in_the_studentised_residual_where_the_lumen_is_tight():
    q_hat, sigma_hat = measured(0, 1, 2, 3, 4, 5, 6, 7, sigma=0.5)
    q_hat[0] = 50.0  # the trunk read five times too fast
    solution = solve_flux_network(q_hat, sigma_hat, lumens_of(READING, ARMS))
    assert abs(solution.studentised[0]) > 3, "the trunk's measurement is far from what the daughters predict"
    assert np.abs(solution.studentised[[1, 2, 3, 4, 5, 6, 7]]).max() < 2, "the others are not"
    assert solution.balance_weight[0] < 1, "the bifurcation's balance was weighted down, not the daughters bent"


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
    tight = solve_flux_network(q_hat, sigma_hat, lumens_of({1: [[0, 1, 2]]}, ARMS), sigma_flux=0.03, flips=False)
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
    assert directed_cycles(EDGES, flipped, confidence=np.array([9, 9, 0.1, 9, 9, 9, 9, 9])) == []  # no confidence, no part
    n = 6
    grid = [((i * n + j), (i * n + j + 1)) for i in range(n) for j in range(n - 1)] + \
           [((i * n + j), ((i + 1) * n + j)) for i in range(n - 1) for j in range(n)]
    both = grid + [(b, a) for a, b in grid]
    assert len(directed_cycles(both, np.ones(len(both), dtype=int))) == 1  # one per component, not every cycle


@pytest.mark.parametrize("n,seed", [(8, 1), (8, 2), (8, 3), (12, 1), (12, 2), (12, 3)])
def test_on_a_plexus_the_network_does_not_make_the_directions_worse(n, seed):
    # over seeds the gain in measured directions right runs from 0 to +4 points and the flips from 2:0 to 9:2
    case = grid_case(n, seed)
    assert case["after"] >= case["as read"] - 1e-9, "the measured directions are at least as right as read"
    assert case["flips right"] >= case["flips wrong"], "the flips are at least as often right as wrong"
    assert case["all after"] > 0.75
