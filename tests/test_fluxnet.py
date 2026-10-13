"""The flux network of docs/cyclic_graph_recursion_plan.md, section 40, on a toy graph.

Vertices and edges (first vertex -> last vertex; the flux runs along the edge's points):

    0 -e0-> 1 -e1-> 2 -e4-> 5 -e7-> 7        e0 trunk (10), e1 / e2 daughters (6 / 4) of the bifurcation at 1
            |       ^       ^                 e3 -> e5 a second vessel crossing at 2 (5)
           e2      e3      e6                 e6 the daughter e2 rejoining at the confluence 5 (a loop 1-2-5-3-1
            v       |       |                 in the undirected graph, not a directed one)
            3 -e6-> 5      4 -e3-> 2 -e5-> 6
"""

import numpy as np
import pytest

from vascnet.fluxnet import compare_readings, directed_cycles, lumens_of, solve_flux_network

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
    # A wrong direction is a flipped sign, not noise: with the balance allowed to miss by 45% a 12-sigma
    # measurement would never be outvoted by Gaussian misfit alone. The sign step decides between "as read" and
    # "turned round" from the rest of the network, with the prior that a direction is right with probability p.
    q_hat, sigma_hat = measured(0, 1, 2, 3, 4, 5, 7, flip=(1,))
    solution = solve_flux_network(q_hat, sigma_hat, lumens_of(READING, ARMS))
    assert solution.flipped[1] and not solution.flipped[[0, 2, 3, 4, 5, 7]].any()
    assert solution.q[1] > 0 and abs(solution.q[1] - 6.0) < 0.5
    assert (solution.weight[[0, 2, 3, 4, 5, 7]] > 0.99).all()
    # the unmeasured loop edge e6 is inferred the right way as well
    assert solution.direction[6] == 1


def test_a_lone_measurement_is_never_turned_round():
    q_hat, sigma_hat = measured(0, 1, flip=(1,))
    solution = solve_flux_network(q_hat, sigma_hat, [])  # no balances: the network has no opinion
    assert not solution.flipped.any() and solution.q[1] < 0


def test_the_right_pairing_at_the_crossing_balances_better():
    q_hat, sigma_hat = measured(1, 3, 4, 5, sigma=0.5)  # 6 and 2 pass through: a difference the balance can see
    right = lumens_of({2: [[0, 1], [2, 3]]}, ARMS)
    wrong = lumens_of({2: [[0, 3], [2, 1]]}, ARMS)   # e1 with e5, e3 with e4
    one_lumen = lumens_of({2: [[0, 1, 2, 3]]}, ARMS)  # all four arms one lumen (a confluence of two into two)
    assert compare_readings(q_hat, sigma_hat, right, wrong) > 2.0
    with pytest.raises(ValueError):
        compare_readings(q_hat, sigma_hat, right, one_lumen)  # fewer balances always fit at least as well
    # two vessels whose fluxes differ by 20% cannot be told apart by a balance that may miss by 45%
    close = q_hat.copy()
    close[[3, 5]] = 5.0
    assert compare_readings(close, sigma_hat, right, wrong) < 1.0


def test_an_edge_nothing_reaches_has_no_direction():
    q_hat, sigma_hat = measured(0)
    solution = solve_flux_network(q_hat, sigma_hat, [])
    assert solution.direction[0] == 1 and solution.direction[1] == 0
    assert np.isinf(solution.sigma[1])


def test_directed_cycles():
    ring = [(0, 1), (1, 2), (2, 0)]
    assert directed_cycles(ring, np.array([1, 1, 1])) == [[0, 1, 2]]
    assert directed_cycles(ring, np.array([1, 1, -1])) == []
    assert directed_cycles(ring, np.array([-1, -1, -1])) == [[2, 1, 0]]
    # the toy graph's loop is not a directed cycle under the true directions
    assert directed_cycles(EDGES, np.ones(len(EDGES), dtype=int)) == []
    # with e6 the wrong way round (5 -> 3) there is one: 1 -> 2 -> 5 -> 3 -> 1 needs e2 backwards too
    flipped = np.ones(len(EDGES), dtype=int)
    flipped[[2, 6]] = -1
    assert directed_cycles(EDGES, flipped) == [[1, 4, 6, 2]]


@pytest.mark.parametrize("sigma_flux", [0.2, 0.45, 1.0])
def test_balances_are_soft(sigma_flux):
    q_hat, sigma_hat = measured(0, 1, 2, sigma=0.5)
    q_hat[0] = 12.0  # the trunk reads 20% more than its daughters add up to
    solution = solve_flux_network(q_hat, sigma_hat, lumens_of({1: [[0, 1, 2]]}, ARMS), sigma_flux=sigma_flux)
    # the softer the balance, the closer the fluxes stay to their measurements
    assert abs(solution.q[0] - 12.0) < abs(12.0 - 10.0)
    assert solution.chi2 > 0


def test_the_imbalance_is_a_test_of_each_lumen_without_its_own_balance():
    # the same crossing, read three ways; the leave-one-out imbalance says which lumens balance, and the junction
    # reading of section 27 takes the most specific reading whose tests all fit
    q_hat, sigma_hat = measured(1, 3, 4, 5, sigma=0.5)
    right = solve_flux_network(q_hat, sigma_hat, lumens_of({2: [[0, 1], [2, 3]]}, ARMS))
    wrong = solve_flux_network(q_hat, sigma_hat, lumens_of({2: [[0, 3], [2, 1]]}, ARMS))
    one_lumen = solve_flux_network(q_hat, sigma_hat, lumens_of({2: [[0, 1, 2, 3]]}, ARMS))
    assert np.nanmax(np.abs(right.imbalance)) < 1.0, "6 in, 6 out and 2 in, 2 out: both lumens balance"
    assert np.nanmin(np.abs(wrong.imbalance)) > 2.0, "6 in, 2 out: neither lumen balances"
    assert np.nanmax(np.abs(one_lumen.imbalance)) < 1.0, "8 in, 8 out balances too: specificity is section 27's job"


def test_an_enforced_balance_would_hide_its_own_imbalance():
    # with every arm measured the leave-one-out imbalance is the plain net flux over its spread; a solve that
    # enforced the balance would have pulled it towards zero
    q_hat, sigma_hat = measured(0, 1, 2, sigma=0.5)
    q_hat[0] = 14.0  # the trunk reads 14 against daughters of 6 and 4
    lumens = lumens_of({1: [[0, 1, 2]]}, ARMS)
    solution = solve_flux_network(q_hat, sigma_hat, lumens)
    scale = 0.45 * np.mean([14.0, 6.0, 4.0])
    expected = (14.0 - 6.0 - 4.0) / np.sqrt(scale ** 2 + 3 * 0.5 ** 2)
    assert abs(abs(solution.imbalance[0]) - expected) < 0.15 * expected
    fitted = (solution.q[0] - solution.q[1] - solution.q[2]) / scale
    assert abs(fitted) < abs(solution.imbalance[0]), "the enforced balance pulls the fitted net flux towards zero"
