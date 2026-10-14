# Plan: a clean cyclic graph from the plain ridges, rendered back and corrected by recursion

Version 5. This plan continues the burst notebook (`notebooks/development copy 2.ipynb`, sections 1–34) with sections 35–42. It has been through four adversarial reviews (`cyclic_graph_recursion_plan_review.md` to `_review_4.md`); the last part of this document says how each finding was handled. Section 40 was prototyped (`vascnet/fluxnet.py`, `tests/test_fluxnet.py`) and is described as it behaves, with the numbers the tests and the reviews measured.

**What this plan claims, plainly.**

- **Sections 37 and 41** (the graph with confidences, calibrated) should raise the harness's centerline and junction columns over the consolidated row, because they keep what the consolidation dropped and say how much each piece is worth. This is the cheapest section and the first to run; its one number (the first principle's) decides whether the rest is built on it.
- **Section 40** (the flux network) should raise the direction columns on tree-like beds: it infers the unmeasured directions along whole vessels, turns lone wrong directions round where a lumen joins different measurements (+3 to +7 points on binary trees with the thick level unmeasured, measured), and reports what it cannot settle. Its sign step reaches exactly the lumens that join *different* measurements (bifurcations, confluences, joins between whole vessels); a through-lumen of one whole vessel, and every through-lumen of a projected crossing, pairs arms of one measurement and constrains no sign. On a plexus of projected crossings the sign step is therefore inert by construction (measured), and on a true anastomotic net it gains 0 to 4 points (measured); the plexus gains are the inferred directions, not the flips.
- **Sections 38 and 39** (renderer, residual, recursion) should raise found-and-typed-right where the core's matched filter has power, which is vessels thicker than about 3 px; at capillary crossings it is marginal and geometry keeps deciding. The recursion's moves are sound in form; the flux network feeds the readings once, at the end, never in a loop.
- **Section 36** (junction cells) should raise junction recall at thick junctions; for capillaries it hangs on its first check.

What is already there, and the plan builds on rather than rebuilds:

- **The plain ridges**: `ridge_crests` → `centerlines` (section 29, `consolidate=False`). `centerlines` prunes side branches under `SPUR_LENGTH` = 12 px (a thinning artefact) and takes no setting; the consolidation's joining and dropping steps are interleaved in `consolidate_lines`, so section 37's separation of them is new code.
- **The score section 29 saved**: `static_scores` is `orientation_scores`, the pre-interaction score. Section 36 reads it.
- **Arms read on a ring** (`read_arms`, section 26); an edge that never leaves the core has its outer half read, inside the core. `contract_cores` already drops spurs shorter than the core from a node's ends.
- **The per-junction tests** (`group_tests`, `partition_reading`, section 27): a flux test over `SIGMA_FLUX` × the sum of the arms' absolute fluxes, which can never fail a partition; an `overlap` test of the core's 5 × 5 maximum density against the arms' sum or maximum over `SIGMA_DENSITY`; an `in and out` test that fails a group whose measured directions are all in or all out. A NaN in a test fails the partition, so a missing test must be left out.
- **The junction reading** (`choose_partition`): the most specific partition none of whose tests is off by more than `Z_PASS`; `reading_confidence` = goodness of fit × (1 − exp(−margin)), with the margin the best score over the second best; `geometric_reading` reads a planar four-way as a crossing.
- **`fuse_split_crossings`** (section 26) runs inside `vessel_graph`.
- **Whole vessels** (section 32): one kymograph and one direction per whole vessel; pieces joined through `join_pieces` by the geometric partition; a join whose two sides read against each other is undone, and only the count of undone joins is stored.
- **p** (section 33) from the three-in junctions and the sides.
- **vascnet's renderer** (`vascnet.synth.render`): max within a tree, transmittance blurred, multiplied across trees.

## Principles

- **Nothing is deleted below the thinning scale; everything carries a confidence; the principle has a number.** On the held-out bursts the edge confidence must tell true pieces from false ones with `told_apart` ≥ 0.75 (on the five bottom-up terms, since the sixth comes from section 38, which is built last), and the confidence curve must pass the consolidated row's point. Below that, section 29's consolidation is kept.
- **One objective, written down.** `J = χ²_w + λ Σ_elements k_e ln(n_e)`: the whitened squared residual over the valid pixels plus a price per parameter of every drawn element, `n_e` its effective number of independent pixels (from the whitened template's norm), λ the one swept multiplier. The *structure* of the drawing changes only by moves, taken when J falls. The halo is fitted and its parameters are priced; κ and R are measured on windows and arms, not optimised under J, and they jump when a move changes the windows (a merge moves the core radius): the jump is in ΔJ. The price counts fitted and measured parameters alike; the difference is what λ absorbs, which is why λ is swept.
- **A reading and the drawing never disagree.** A re-composition proposed by a reading is a move; if J vetoes it, the reading reverts. Bridges are never drawn.
- **Flow decides what a still cannot, where it can**, and section 40 states where it cannot.
- **Each section has one column it has to move**, named in advance, on the held-out bursts.

---

## 35. Synthetic bursts with the things this plan is about

Extend the generator (section 28), keeping the old seeds bit-identical: loops (`SYNTH_LOOP`); a plexus of two kinds, **projected crossings** at several depths and **true anastomoses** (mixing nodes of four arms in one lumen), recorded apart, since section 40 behaves differently on them; gaps (`SYNTH_GAP`, `SYNTH_GAP_LENGTH`, `SYNTH_GAP_DEPTH`); terminations; confluences; halos of either sign; shared lumens and collinear overlaps; a velocity profile across the vessel (`v_max (1 − (r / R)^k)`, `SYNTH_BLUNT` ≈ 3; until it is in, the harness's flux columns are upper bounds); blur in the transmittance domain; the background fit of section 12 (or a local mean with the harness marked approximate); per-junction `n` in the truth. **Column:** none; the per-seed counts, and the old seeds proven unchanged.

---

## 36. Junction cells

As in version 4: a ring reading of the pre-interaction score at `JUNCTION_RINGS` = (4, 6, 10, 16, 24) px, peaks as arms, `n` and `rate`, non-maximum suppression per ring, plaid cells at `PLAID_APART` = 45 degrees. **Checks first**: a thick vessel crossed by a thin one, a thin daughter off a thick trunk, and two T's 5–10 px apart, on the *oriented score* (on the density the 4 px ring resolves 40-degree daughters; the score is the check). **Column:** junction recall at the plain ridges' junction precision, by each junction's `n` against the truth's (new harness code).

---

## 37. The clean cyclic graph

**Input.** `ridge_crests` → close within `RIDGE_JOIN` → `centerlines` (its `SPUR_LENGTH` = 12 pruning kept: it is what keeps `extend_ends` from extending hundreds of stubs) → `extend_ends` → `ridge_lines` with `least = 0`, `to_forks = True`; none of `consolidate_lines`' dropping steps. New code, named: the separation of joining from dropping, and **provenance**: which forks `extend_ends` made (it returns a picture today) and which joins section 32 undid (`whole_vessel_flow` keeps only their count), because section 40 needs a structural confidence for every two-arm join, and a join has no reading.

**Vertices and edges** as in version 4 (cores contracted, cells added, twins kept and marked, multi-edges and loops allowed).

**Confidence per edge**: length in field lengths; strength; repeatability per edge (new code from `found_again`); `flow_share`; twin and stub; and later the rise of J without the edge. **Structural confidence per two-arm join**: 1 − ε for a join the skeleton made; the extension's confidence for one `extend_ends` made; 0.5 for one section 32 undid; a bridge's for a bridge.

**Column:** `told_apart` ≥ 0.75 on the five bottom-up terms, with the precision–recall curves against the plain and consolidated rows.

---

## 38. The renderer and the residual

**Per-edge measurements** at windows outside the core radius, σ fixed per fixation, `R_od` only above the blur, line mass elsewhere; **windowless edges** take their lumen's κ and R, or `read_arm`'s outer half and `R_ridge`, and are counted.

**The renderer**: per-lumen canvases, max within, transmittance blurred, multiplied (a local canvas is transmittance 1 outside its extent), `−log`; halos of either sign, `σ_h` ≤ `BACKGROUND_SIGMA`, fitted and priced.

**The whitened residual**, whitened per region by its own measured autocorrelation (off the graph on the quiet pixels; along the edges on one half of the edges). **The column**: the whitened RMS over the pixels the drawing touches (edges, cores, halos), whitened by the along-edge autocorrelation measured on the *other* half of the edges; 1 where the model is right, and sensitive to every move, which an RMS on quiet pixels is not.

**The core's matched filter.** The template is the difference between the two compositions rendered at the core from the arms' κ and R; its whitened inner product with the residual over its norm is a z for two vessels against one lumen. What it adds to section 27's `overlap` test: every pixel of the core instead of a 5 × 5 maximum, the right spatial template, and the noise's correlation. **Its reach, stated**: z ≈ c √n_eff with c the per-pixel contrast of the thinner vessel in noise units and n_eff the overlap's independent pixels; for two capillaries (R ≤ 2, overlap 9–16 px, n_eff 2–4, c 1.5–2) z ≈ 2–4, marginal against `Z_PASS` = 3; for R ≥ 3, n_eff ≥ 10 and c ≥ 2, decisive. A **check before section 39**: the z by radius on rendered crossings against recorded one-lumen junctions. So at a capillary four-way geometry keeps deciding, and the plan does not claim the filter there.

**Where the residual is read, the table, and the rule for new vessels** as in version 4 (cores, off the graph, along edges for what the fit cannot absorb; evidence off existing edges adds a vessel; the collinear miss counted).

**Column:** `told_apart` of the core z between one-lumen and two-vessel junctions, by radius, and the on-graph held-out whitened RMS.

---

## 39. The recursion

Moves under J, λ swept on the real burst by repeatability, loose ends, impossible junctions and directed cycles; at most `ROUNDS_MOST` = 5 rounds; the rounds end because J falls at every taken move.

**Rounds 1 to 4: structure, with the readings from geometry and the core filter only.** Step 0 halos; step 1 bridges (never accepted, rejected or drawn; confidence from similarity; rivals unresolved); step 2 merging arms through a junction by the partition reading with the core filter among its tests; step 3 arms against the cell's `n`, centerline shifts as zero-parameter moves; step 3b removal, merge, split (proposed by the cell, decided by the reading, J a veto on which the reading reverts); step 4 composition; step 5 missing vessels from the whitened residual's field off the existing edges.

**Step 6, the flux network, once, in the last round.** (a) The readings as they stand; (b) one solve on the whole graph, with a measurement id per whole vessel, structural confidences per join, bridges as doubtful edges in their end's unit; (c) every vertex's candidate readings scored by the leave-one-vertex-out tests; (d) the readings taken once more, the flux test able to fail a reading and never to choose among readings that fit; (e) **the readings are then frozen**: no further solve, because a sequence of readings and solves oscillates (measured on a grid), and the rounds would have been that loop. The network's other findings are applied in **one closing round of structural moves with the readings frozen**, by the rule of `blame`: an imbalance at a two-arm lumen, a lumen set aside, or an impossible lumen goes to the graph (a missing arm or a wrong join: step 3, step 1 for the rival bridges, ranked by the flux lost at the free ends); a speed is in doubt only when its edge is off in lumens of three or more arms at both its ends; everything else is reported with the graph.

**Column:** found-and-typed-right after each round; bridges right / wrong at the bridge confidence's cutoffs; the on-graph whitened RMS; the readings changed by step 6 (a count, since there is no loop to converge).

---

## 40. Net flux = 0: a flux network (prototyped)

`vascnet/fluxnet.py`; `tests/test_fluxnet.py` on a toy graph, on grids with harmonic flow as a true anastomotic net, and on a plexus of projected crossings. What follows is what it does and what was measured.

**Unknowns, measurements, balances.** One flux per edge; measured on edges with a clear speed, `M × v`; free elsewhere. One balance per lumen, allowed to miss by `SIGMA_FLUX × Σ|q|` over the lumen's arms, the notebook's unit. **`SIGMA_FLUX` is measured per lumen kind** (through-lumens of one vessel, where the velocity-profile factor cancels; bifurcations, where it does not), on junctions whose directions are measured and possible, with the counts printed; the population is capillaries, since the thick vessels are above the trackable limit, and the plan says so; where the real burst has too few, the harness's value stands in.

**The unit of a direction is a measurement**: one whole vessel (`join_pieces`' path), all its edges turned round together at one cost `2 ln(p / (1 − p))` (3.5 at p = 0.85). A join section 32 undid gets no balance (structural confidence 0.5, cost 0) and is reported; it is not left as a through-lumen for the network to flip one of its sides.

**What the balances know.** A through-lumen inside one whole vessel pairs two pieces that share one velocity: its balance is the two pieces' mass ratio over the softness, section 27's `density` test in a looser unit, and it carries no sign information. A projected crossing's two through-lumens pair arms of one vessel each: the same. **The sign step's reach is the lumens that join different measurements**: bifurcations and confluences with at least two measured arms, and joins between two whole vessels. A bifurcation off an unmeasured trunk gives nothing. On a plexus of projected crossings the step is inert by construction (measured: twelve vessels, two reversed, no flips, no imbalance), and the sign column there is expected to be zero; what the network gives a plexus is the inferred directions of unmeasured pieces along whole vessels.

**The signs.** Within every lumen, every set of the measurements touching it is a move to turn those round, in closed form from the one solve's covariance (exact also with Huber weights and priors, verified). **A reading may instead be set aside**, both lumens of a crossing together, at the cost `ln(c / (1 − c))` of its **structural** confidence c: the probability that the structure that made the lumen is right, never a goodness of fit. Section 27's `reading_confidence` is a goodness of fit (median about 0.65 under calibrated tests, a fifth of it below 0.5) and is not used here; what is used is 1 − ε for a vertex with one structure (three arms; a skeleton join), the margin of the best partition over the second best for a vertex of four or more arms (in log units, calibrated in section 41), the extension's or bridge's confidence for a join it made, 0.5 for an undone join. The best move is taken while one lowers the total; `flip_margin` is a precision knob for the harness. **The pattern rule** (every arm measured, all in or all out) is reported and never acted on.

What was measured:

| case | outcome |
|---|---|
| one wrong direction, contradicted at both ends | turned round |
| one wrong on a leaf, contradicted at one end (a confluence all in) | turned round at 0.45 |
| trunk and one daughter wrong at a bifurcation | at 0.15 the two are turned together; at 0.3 with the joins at 0.99 the join the wrong daughter passes is set aside and reported; at 0.45 one is turned and the rest left and reported; no right measurement cascades |
| two vessels ending near each other, read as one, the join at 0.8 | set aside and reported impossible; nothing turned; `blame` sends it to the graph |
| a whole vessel of two pieces through a crossing, read backwards | turned round as one at 0.3 and 0.15; at 0.45 the two 1.3σ balances do not quite pay for it |
| a crossing read with margin 1.0 against a 4-against-2 imbalance | set aside as one reading |
| a daughter missing from the graph | the two-arm lumen goes to the graph; the trunk's right speed is not blamed (at a tight balance the raw residual had blamed it at 5σ) |
| true anastomotic grids 8×8 to 16×16, per-edge errors, three seeds | +0 to +4 points; flips 2:0, 2:0, 0:0, 8:1, 7:1, 9:2, 11:1 |
| a crossing plexus, whole-vessel errors | inert by construction; with the arms wrongly read as one lumen of four, mixed (3:0 to 1:2 over six seeds, by the fourth review) |
| binary trees, thick level unmeasured, 15% wrong, structural confidences at their defaults | +3 to +7 points at 0.45, flips 18:5 over four seeds; near-perfect at 0.15 |
| the same trees with goodness-of-fit confidences wrongly passed as structural | 9–28 of 127 lumens set aside, gain 0 to 1 point (the fourth review); the reason the two are separated above |

**The sizes.** Huber weights on balances and misfits; the studentised residual per measurement. **Its reach**: a wrong speed is found where the lumens around it are tight; on a plexus with 70% of the edges measured, none of ×3 / ÷3 is found (measured). `blame` gives the rule for which of a missing arm, a wrong join and a wrong speed an imbalance is.

**The tests.** Per lumen the leave-one-out imbalance; per vertex the candidate readings from one leave-one-vertex-out solve (independent of the enforced reading, verified); they enter `group_tests` as the `flux` test, left out where NaN, able to fail a reading, never choosing among readings that fit; at the notebook's softness they fail little (every four-arm vertex of a single-depth plexus passes as a crossing by flux alone, measured).

**Doubtful edges.** A bridge with confidence c has a prior of zero flux with sigma `c / (1 − c)` times **its free end's own flux** (the arm's measurement, else the lumen's other arm): scaled by the graph's mean, a right thick bridge was held back and a wrong capillary bridge was not (measured). A free end's rival bridges are one lumen with the end's arm; `flux_at_free_ends` reports what arrives at every free end.

**Directed cycles**: strongly connected components over edges with `|q| / σ` ≥ 3, one cycle per component, reported only. Flips, set-asides, impossible lumens and cycles are never measures of p.

**Cost.** One dense solve per fixation, once: 3–10 s at a thousand edges and 20–180 s at two thousand across the two machines that measured it; sparse factorisation from a thousand edges on. The sign moves are in closed form; a set-aside costs one re-solve.

**Column:** directions right among inferred edges on the tree bursts (the plexus apart, as description). On the real burst the comparison with `propagate_directions` is agreement between two inferences, not a score, and is reported as such.

---

## 41. Confidences

Every confidence a sum of calibrated log-likelihood terms, the prior explicit and separate; the bottom-up terms first on section 37's graph. **The structural margin of section 27** (the best partition over the second best) is calibrated on the harness as the probability that the best structure is right, because section 40's set-aside cost is that margin. Two-arm joins get their structural confidence from provenance (section 37), since they have no reading.

The graph handed on: `vessel_digraphs[n]`, per edge the spline, radii, line mass, blur, velocity ± σ, direction with `|q| / σ`, flux ± σ, the edge confidence and its terms, `drawn`, flags; per vertex position, kind, type, partition, structural and fit confidences, `impossible`, `set aside`, imbalance; per component with a directed cycle, one cycle.

---

## 42. The harness rows this adds

Rows: `plain ridges, joined, cyclic graph`; `cyclic graph, rendered, round 1 … final`; `flux as a test`; the λ sweep. Columns: the one per section (37 `told_apart`; 36 junction recall at matched precision; 38 core-z `told_apart` by radius; 39 found-and-typed-right; 40 inferred directions right on trees), and as description: bridges right / wrong, lumens fused right, flux `z` `told_apart` with the share of readings failed, flips right / wrong, set-asides per softness, directed cycles, windowless edges, on-graph whitened RMS, readings changed by step 6. Flux columns upper bounds until the velocity profile is in.

---

## The order of work

1. Section 37 on the real burst and the harness; the first principle's number on five terms.
2. Section 40 on `whole_graphs` as they are, with provenance for the joins and `SIGMA_FLUX` per lumen kind: inferred directions, set-asides, impossible lumens, cycles, flux at free ends on the real burst; inferred directions scored on the tree bursts.
3. Section 36's three checks on the oriented score.
4. Section 38's core-z check by radius; then sections 35 and 38 on one synthetic burst, section 39 round by round, the harness.

---

## How the reviews' findings were handled

| finding | handling |
|---|---|
| R1 A1, R2 4, R3 5, R4 7 | whole-network residual; windows stop at the core; windowless edges counted and given their lumen's parameters; the core filter's reach stated by radius, what it adds over `overlap` said, a check by radius before section 39; capillary four-ways left to geometry |
| R1 A2, R2 3, R3 10, R4 6, R4 16 | whitening per region; `n_e` from the whitened template; the RMS column on the pixels the drawing touches, whitened by a held-out half of the edges; halo fitted and priced, measured parameters' jumps in ΔJ, λ absorbs the rest |
| R1 A3, R2 10 | removal, merge, split as moves; reading reverts on a veto |
| R1 A4, R2 1, R3 2, R4 3 | flux tests from one leave-one-vertex-out solve, failing only; step 6 once, in the last round, readings then frozen; a closing structural round; no loop |
| R1 A5, R2 5, R3 4, R4 5 | bridges never drawn; a prior in the free end's own unit; rivals one lumen; flux at free ends |
| R2 2, R3 1, R3 3, R3 8, R4 1, R4 2, R4 10, R4 12 | whole-vessel measurements; structural set-aside cost, never goodness of fit, per reading; the pattern rule reported; the sign step's reach stated and measured on a crossing plexus (inert) and an anastomotic grid; set-asides per softness a column; a flip margin for the harness |
| R2 6, R3 13, R4 11 | the notebook's unit; `SIGMA_FLUX` per lumen kind, counts printed, population named |
| R2 7, R4 9 | robust sizing; `blame` for one signal |
| R2 8, R3 11 | SCC cycles at cutoff 3 |
| R2 9 | transmittance blur; local canvases |
| R2 11, B7 | λ swept with impossible junctions and cycles |
| R2 12, R4 13 | per-edge repeatability and join provenance named as new code; plaid cells on the pre-interaction score |
| R2 13, B8, R3 6, R4 8 | the skeleton's pruning kept and credited; the field-length clause dropped; `told_apart` ≥ 0.75 on five terms |
| R2 14, R3 9, R4 2 | tests for whole vessels, set-asides, the prior across two orders, a crossing plexus, a missing daughter |
| R2 15, R3 12, R4 15 | cost as measured on two machines; sparse from a thousand edges |
| R2 16, B9, R3 14 | rings from 4 px, NMS per ring; the oriented-score check; per-junction `n` |
| R2 17, C5, R4 14 | prior explicit; the structural margin calibrated; set-aside per reading |
| R2 18, B1 | `flux as a test`; velocity profile; upper bounds |
| R3 7, R4 6 | the RMS on the drawing's pixels |
| R4 4 | undone joins recorded and given no balance; through-lumens of one vessel carry the mass ratio only |
| B2, B3, B4, B6 | σ per fixation; per-lumen canvases; halo of either sign bounded; flips not p |
| C3 | the flow field as a second source of edges: not in this version |

## Open choices for you

1. The line mass for the flux: the kymograph's M or the profile's `π κ R²`. The plan uses M.
2. Composition where there is no reading: sum (the plan's choice) or max.
3. The real burst's base rate of true terminations against gaps and dives.
4. Whether the flow field (section 24) should be a second source of edges now or in the next version.
