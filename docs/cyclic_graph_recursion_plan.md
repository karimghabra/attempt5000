# Plan: a clean cyclic graph from the plain ridges, rendered back and corrected by recursion

Version 6. This plan continues the burst notebook (`notebooks/development copy 2.ipynb`, sections 1–34) with sections 35–42. It has been through five adversarial reviews (`cyclic_graph_recursion_plan_review.md` to `_review_5.md`); the last part of this document says how each finding was handled, and the fifth review's verdict is that sections 35, 36, 37, the moves of 39 and section 40 on trees have converged, with what remains being for the harness to measure. This version makes the three design decisions it left open (the structural confidences, the plexus claims, the residual column) and closes its two code holes. Section 40 was prototyped (`vascnet/fluxnet.py`, `tests/test_fluxnet.py`) and is described as it behaves, with the numbers the tests and the reviews measured.

**What this plan claims, plainly.**

- **Sections 37 and 41** (the graph with confidences, calibrated) should raise the harness's centerline and junction columns over the consolidated row, because they keep what the consolidation dropped and say how much each piece is worth. This is the cheapest section and the first to run; its one number (the first principle's) decides whether the rest is built on it.
- **Section 40** (the flux network) should raise the direction columns on tree-like beds: it infers the unmeasured directions along whole vessels, turns lone wrong directions round where a lumen's arms are all measured or pinned by other lumens, which on the real pipeline means bifurcations and confluences (+3 to +7 points on binary trees with the thick level unmeasured, measured), and reports what it cannot settle. **On a plexus it does nothing, and the plan claims nothing there.** `join_pieces` joins every vessel through every crossing, so a vessel of a projected plexus is one measurement end to end and every balance pairs arms of one measurement: no sign is constrained, and a vessel with no measured piece is in the null space and gets no direction either (measured). A true anastomosis is read as a crossing by `geometric_reading`, and the one-lumen reading of four arms has no tests and is never a rival, so the network cannot set the wrong structure aside; the only thing that can read an anastomosis is section 38's core filter, which is decisive for vessels of R ≥ 3 and marginal for capillaries. The plexus column of section 42 is therefore expected to be zero until section 27 can read a thick anastomosis as one lumen.
- **Sections 38 and 39** (renderer, residual, recursion) should raise found-and-typed-right where the core's matched filter has power: for R ≥ 3 at every blur the generator makes (z 9 to 135, derived by the fifth review for the generator's noise), and for capillaries only in focus (z 18.6 at a blur of 0.8 px, 2.4 at 2 px, 0.8 at 3.5 px). The recursion's moves are sound in form; the flux network feeds the readings once, at the end, with a second solve after it and no loop.
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
- **One objective, written down.** `J = χ²_w + λ Σ_elements k_e ln(n_e)`: the squared residual over the valid pixels **whitened by the noise's two-dimensional autocorrelation measured on the quiet pixels off the graph** (one whitening, defined at every pixel; the residual's own correlation along edges is model error and is what the price is for), plus a price per parameter of every drawn element, `n_e` its effective number of independent pixels (from the whitened template's norm), λ the one swept multiplier. λ is chosen on the real burst as the largest value at which the shares of impossible junctions and of directed cycles do not exceed the consolidated graph's, with repeatability and loose ends reported beside it, not combined. The *structure* of the drawing changes only by moves, taken when J falls. The halo is fitted and its parameters are priced; κ and R are measured on windows and arms, not optimised under J, and they jump when a move changes the windows (a merge moves the core radius): the jump is in ΔJ. The price counts fitted and measured parameters alike; the difference is what λ absorbs, which is why λ is swept.
- **A reading and the drawing never disagree.** A re-composition proposed by a reading is a move; if J vetoes it, the reading reverts. Bridges are never drawn.
- **Flow decides what a still cannot, where it can**, and section 40 states where it cannot.
- **Each section has one column it has to move**, named in advance, on the held-out bursts.

---

## 35. Synthetic bursts with the things this plan is about

Extend the generator (section 28), keeping the old seeds bit-identical: **capillaries** (the generator's thinnest vessel today is about 2.5 px in radius, so no claim about capillaries can be checked on it; add trees with R of 1 to 2 px); loops (`SYNTH_LOOP`); a plexus of two kinds, **projected crossings** at several depths and **true anastomoses** (mixing nodes of four arms in one lumen), recorded apart, with the readings and measurement ids the pipeline makes for them (crossings from `geometric_reading`, one measurement per `join_pieces` path), since that is what section 40 sees; gaps (`SYNTH_GAP`, `SYNTH_GAP_LENGTH`, `SYNTH_GAP_DEPTH`); terminations; confluences; halos of either sign; shared lumens and collinear overlaps; a velocity profile across the vessel (`v_max (1 − (r / R)^k)`, `SYNTH_BLUNT` ≈ 3; until it is in, the harness's flux columns are upper bounds); blur in the transmittance domain; the background fit of section 12 (or a local mean with the harness marked approximate); per-junction `n` in the truth. **Column:** none; the per-seed counts, and the old seeds proven unchanged.

---

## 36. Junction cells

As in version 4: a ring reading of the pre-interaction score at `JUNCTION_RINGS` = (4, 6, 10, 16, 24) px, peaks as arms, `n` and `rate`, non-maximum suppression per ring, plaid cells at `PLAID_APART` = 45 degrees. **Checks first**: a thick vessel crossed by a thin one, a thin daughter off a thick trunk, and two T's 5–10 px apart, on the *oriented score* (on the density the 4 px ring resolves 40-degree daughters; the score is the check). **Column:** junction recall at the plain ridges' junction precision, by each junction's `n` against the truth's (new harness code).

---

## 37. The clean cyclic graph

**Input.** `ridge_crests` → close within `RIDGE_JOIN` → `centerlines` (its `SPUR_LENGTH` = 12 pruning kept: it is what keeps `extend_ends` from extending hundreds of stubs) → `extend_ends` → `ridge_lines` with `least = 0`, `to_forks = True`; none of `consolidate_lines`' dropping steps. New code, named: the separation of joining from dropping, and **provenance**: which forks `extend_ends` made (it returns a picture today) and which joins section 32 undid (`whole_vessel_flow` keeps only their count), because section 40 needs a structural confidence for every two-arm join, and a join has no reading.

**Vertices and edges** as in version 4 (cores contracted, cells added, twins kept and marked, multi-edges and loops allowed).

**Confidence per edge**: length in field lengths; strength; repeatability per edge (new code from `found_again`); `flow_share`; twin and stub; and later the rise of J without the edge.

**Structural confidence per join**, which section 40 consumes, and which is never 1 − ε where the new synthetic cases can make the structure wrong:

- a two-arm join: P(join right | the pair's turn, width ratio and density ratio), calibrated on the harness as a probability (the pair's tests are the only evidence a join has; what section 40 refuses is a goodness of fit standing in for a structure, not calibration); 1 − ε only for two pieces that were one connected ridge before the closing within `RIDGE_JOIN`, since a closing or thinning can fuse two vessels ending near each other or a collinear overlap, which section 35 now makes;
- a join `extend_ends` made, whatever the arm count of the fork it made: the extension's confidence;
- a join section 32 undid: 0.5, no balance;
- a bridge: its confidence;
- a three-arm vertex: not one structure. `set_partitions` allows no group of one, so every three-arm vertex is one lumen of three today, and a T whose stem dives (a through pair plus a free end) is read as a bifurcation and is silent in the network (measured: −0.2 to −1.9σ at 0.45). Section 27 gains the partition "through pair + free end" at three-arm vertices, with the far-side check of `geometric_reading` (which already reads a thin-stemmed T as a crossing and then discards it as a baseline) as its evidence; the bifurcation's structural confidence is then its margin over that rival;
- a four-arm vertex: P(two vessels, not one lumen), from section 27's `overlap` z now and section 38's core z later, calibrated; the margin of the best pairing over the second pairing does not ask this question (it is infinite above a 45-degree acute angle and never compares a crossing with one lumen).

**Provenance**, as wrappers that change no section 29 or 32 code: the pixels `extend_ends` added are `extended & ~lines`, and a fork whose incident pixels include them is extension-made; the joins section 32 undid are the partner pairs of `join_pieces(graph)` run without `undone` whose two edges carry different `vessel` numbers in `graph["readings"]`.

**Column:** `told_apart` ≥ 0.75 on the five bottom-up terms, with the precision–recall curves against the plain and consolidated rows.

---

## 38. The renderer and the residual

**Per-edge measurements** at windows outside the core radius, σ fixed per fixation, `R_od` only above the blur, line mass elsewhere; **windowless edges** take their lumen's κ and R, or `read_arm`'s outer half and `R_ridge`, and are counted.

**The renderer**: per-lumen canvases, max within, transmittance blurred, multiplied (a local canvas is transmittance 1 outside its extent), `−log`; halos of either sign, `σ_h` ≤ `BACKGROUND_SIGMA`, fitted and priced.

**The whitened residual**: one whitening, by the noise's two-dimensional autocorrelation measured on the quiet pixels off the graph; it is the noise's, so a systematic misfit (a wrong profile shape, a wrong blur) shows as residual, which a whitening by the residual's own correlation would hide (it gives an RMS of 1 for any misfit that is the same on both halves). The quiet pixels' spectrum is measured too, not only the short-lag autocorrelation, since the core filter's z rides on the white floor (shot and read noise over the kept frames), not on the texture. The cluster modulation that the frames' mean leaves on a vessel (a tenth to a quarter of the chord, correlated over 30 to 40 px along it) is absorbed by the per-window κ and is named as the dominant on-vessel residual.

**The columns, never pooled**: per edge, the one-dimensional residual along the spline in units of the noise's along-line spread (measured by drawing lines of the same shape through quiet pixels), its RMS by radius; per core, the core z; per halo, its own residual.

**The core's matched filter.** The template is the difference between the two compositions rendered at the core from the arms' κ and R; its whitened inner product with the residual over its norm is a z for two vessels against one lumen. What it adds to section 27's `overlap` test: every pixel of the core instead of a 5 × 5 maximum, the right spatial template, and the noise's correlation. **Its reach, derived for the generator's noise** (the fifth review): two vessels crossing at 90 degrees, R = 1.5 each, give z = 18.6, 2.4 and 0.8 at blurs of 0.8, 2 and 3.5 px; R = 3 each give 44, 17.5 and 9.2; R = 5 and 3 give 135. Decisive for R ≥ 3 at every blur the generator makes, and for capillaries only in focus. The z rides on the white floor of the noise, so on the real burst the quiet pixels' spectrum is what sets it. A **check before section 39**, after section 35 has capillaries: the z by radius and blur on rendered crossings against recorded one-lumen junctions. At a blurred capillary four-way geometry keeps deciding, and the plan does not claim the filter there.

**Where the residual is read, the table, and the rule for new vessels** as in version 4 (cores, off the graph, along edges for what the fit cannot absorb; evidence off existing edges adds a vessel; the collinear miss counted).

**Column:** `told_apart` of the core z between one-lumen and two-vessel junctions, by radius and blur; the per-edge RMS by radius as description.

---

## 39. The recursion

Moves under J, λ chosen on the real burst by the rule of the principles (the largest λ at which impossible junctions and directed cycles stay at or below the consolidated graph's shares; repeatability and loose ends reported beside); at most `ROUNDS_MOST` = 5 rounds; the rounds end because J falls at every taken move.

**Rounds 1 to 4: structure, with the readings from geometry and the core filter only.** Step 0 halos; step 1 bridges (never accepted, rejected or drawn; confidence from similarity; rivals unresolved); step 2 merging arms through a junction by the partition reading with the core filter among its tests; step 3 arms against the cell's `n`, centerline shifts as zero-parameter moves; step 3b removal, merge, split (proposed by the cell, decided by the reading, J a veto on which the reading reverts); step 4 composition; step 5 missing vessels from the whitened residual's field off the existing edges.

**Step 6, the flux network, in the last round: two solves, no loop.** (a) The readings as they stand; (b) one solve on the whole graph, with a measurement id per whole vessel, structural confidences per join, bridges as doubtful edges in their end's unit; (c) every vertex's candidate readings scored by the leave-one-vertex-out tests; (d) the readings taken once more, the flux test able to fail a reading and never to choose among readings that fit; (e) **a second solve under the readings of (d)**, with a vessel that (d) split carried as separate measurements with the old direction, and **no re-reading after it**: a sequence of readings and solves oscillates (measured on a grid), and two solves are not a sequence. `blame`, the imbalances, the impossible and set-aside lumens, the inferred directions, the cycles and the handed-on graph all come from the second solve. Its findings are applied in **one closing round of structural moves with the readings frozen**, by the rule of `blame`: an imbalance at a two-arm lumen, a lumen set aside, or an impossible lumen goes to the graph (a missing arm or a wrong join: step 3, step 1 for the rival bridges, ranked by the flux lost at the free ends). **Wrong speeds are not detected**: the rule that would blame a speed (its edge off in wide lumens at both ends) never fires on trees with a tenth of the speeds three times too fast, at any softness (measured), so the studentised residuals are reported with the base rate of |z| > 3 among right speeds beside them and are sent nowhere.

**Column:** found-and-typed-right after each round; as description, bridges right / wrong at the bridge confidence's cutoffs, the per-edge RMS, and the readings changed at (d) with how many of `blame`'s lumens survive the second solve.

---

## 40. Net flux = 0: a flux network (prototyped)

`vascnet/fluxnet.py`; `tests/test_fluxnet.py` on a toy graph, on grids with harmonic flow as a true anastomotic net, and on a plexus of projected crossings. What follows is what it does and what was measured.

**Unknowns, measurements, balances.** One flux per edge; measured on edges with a clear speed, `M × v`; free elsewhere. One balance per lumen, allowed to miss by `SIGMA_FLUX × Σ|q|` over the lumen's arms, the notebook's unit. **`SIGMA_FLUX` is measured per lumen kind** (through-lumens of one vessel, where the velocity-profile factor cancels; bifurcations, where it does not), on junctions whose directions are measured and possible, with the counts printed; the population is capillaries, since the thick vessels are above the trackable limit, and the plan says so; where the real burst has too few, the harness's value stands in.

**The unit of a direction is a measurement**: one whole vessel (`join_pieces`' path), all its edges turned round together at one cost `2 ln(p / (1 − p))` (3.5 at p = 0.85). A join section 32 undid gets no balance (structural confidence 0.5, cost 0) and is reported; it is not left as a through-lumen for the network to flip one of its sides.

**What the balances know.** A through-lumen inside one whole vessel pairs two pieces that share one velocity: its balance is the two pieces' mass ratio over the softness, section 27's `density` test in a looser unit, and it carries no sign information. A projected crossing's two through-lumens pair arms of one vessel each: the same. **The sign step's reach is the lumens all of whose arms are measured or pinned by other lumens**, which on the real pipeline means bifurcations and confluences: two measured daughters off an unmeasured trunk give nothing, since the trunk takes whatever balances them (measured), and "joins between two whole vessels" is empty by construction, because `join_pieces` joins every through group into one vessel and the only joins between two vessels are the ones section 32 undid, which get no balance. **On a plexus of projected crossings the network does nothing**: no sign is constrained (measured: twelve vessels, two reversed, no flips, no imbalance) and a vessel with no measured piece is in the null space and gets no direction (measured: q = 0, σ 672). The plexus column is expected to be zero, and nothing in this section claims otherwise.

**The signs.** Within every lumen, every set of the measurements touching it is a move to turn those round, in closed form from the one solve's covariance (exact also with Huber weights, priors and a measurement spanning several lumens, verified). **A reading may instead be set aside**, both lumens of a crossing together (the identity for deleting a set of rows, exact also where an arm nothing else reaches makes it singular), at the cost `ln(c / (1 − c))` of its **structural** confidence c, the probability that the structure that made the lumen is right, as section 37 defines it per join and per vertex: never a goodness of fit (section 27's `reading_confidence` has a median of about 0.65 under calibrated tests and would set a fifth of the lumens aside for nothing), never 1 − ε where the structure can be wrong, and never a tie's zero: a reading whose margin is below the notebook's `TIE` has no balance and is reported as tied, so that a tie is not a free pass. The best move is taken while one lowers the total; `flip_margin` is a precision knob for the harness. **The pattern rule** (every arm measured, all in or all out) is reported and never acted on.

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
| a crossing plexus, whole-vessel errors | inert by construction (no flips, no inferred directions) |
| an anastomotic grid read as the pipeline reads it (crossings), whole-vessel errors in chains, six seeds | active only at chain boundaries; mixed: −1.7 to +4.6 points at 0.45 with vessel flips 1:2 to 3:1, −8.7 to +9.4 at 0.15 (the fifth review). This is a wrong structure the network cannot set aside, and the plan does not claim it |
| an anastomotic grid read as one lumen of four (a reading section 27 never produces today), per-edge or chain errors | +0 to +4 points at 0.45, flips 2:0 to 11:1; +1 to +13 at 0.15, flips 16:2. What section 27 would have to produce for the network to help a true plexus |
| binary trees, thick level unmeasured, 15% wrong, structural confidences at their defaults | +3 to +7 points at 0.45, flips 18:5 over four seeds; near-perfect at 0.15 |
| the same trees with goodness-of-fit confidences wrongly passed as structural | 9–28 of 127 lumens set aside, gain 0 to 1 point (the fourth review); the reason the two are separated above |

**The sizes.** Huber weights on balances and misfits; the studentised residual per measurement, **reported, not a detector**: on trees with a tenth of the speeds three times too fast it flags none at 0.45 and as many right speeds as wrong at 0.15 (measured). `blame` gives the rule for which of a missing arm and a wrong join an imbalance is, sends both to the graph, and lists the speeds in doubt as a report.

**The tests.** Per lumen the leave-one-out imbalance; per vertex the candidate readings from one leave-one-vertex-out solve (independent of the enforced reading, verified); they enter `group_tests` as the `flux` test, left out where NaN, able to fail a reading, never choosing among readings that fit; at the notebook's softness they fail little (every four-arm vertex of a single-depth plexus passes as a crossing by flux alone, measured).

**Doubtful edges.** A bridge with confidence c has a prior of zero flux with sigma `c / (1 − c)` times **the flux of the vessel it continues** (the end arm's measurement; where the end is unmeasured, what a first solve without priors makes that vessel carry; else the graph's mean): scaled by the graph's mean, a right thick bridge was held back and a wrong capillary bridge was not (measured). A free end's rival bridges are one lumen with the end's arm; `flux_at_free_ends` reports what arrives at every free end.

**Directed cycles**: strongly connected components over edges with `|q| / σ` ≥ 3, one cycle per component, reported only. Flips, set-asides, impossible lumens and cycles are never measures of p.

**Cost.** One dense solve per fixation, once: 3–10 s at a thousand edges and 20–180 s at two thousand across the two machines that measured it; sparse factorisation from a thousand edges on. The sign moves are in closed form; a set-aside costs one re-solve.

**Column:** directions right among inferred edges on the tree bursts. As description: the plexus (expected zero), the count of sign-constraining lumens (lumens all of whose arms are measured or pinned), set-asides and tied readings per softness, and on the real burst the agreement with `propagate_directions`, which is agreement between two inferences, not a score.

---

## 41. Confidences

Every confidence a sum of calibrated log-likelihood terms, the prior explicit and separate; the bottom-up terms first on section 37's graph. **The structural confidences of section 37** are calibrated on the harness as probabilities: P(join right) for two-arm joins from the pair's turn, width and density; P(bifurcation, not a through pair with a free end) at three-arm vertices from the far-side evidence; P(two vessels, not one lumen) at four-arm vertices from the `overlap` z now and the core z later. The histogram of section 27's margins is reported first (the share that is finite), since the margin alone is infinite at most crossings and answers none of these questions.

The graph handed on: `vessel_digraphs[n]`, per edge the spline, radii, line mass, blur, velocity ± σ, direction with `|q| / σ`, flux ± σ, the edge confidence and its terms, `drawn`, flags; per vertex position, kind, type, partition, structural and fit confidences, `impossible`, `set aside`, imbalance; per component with a directed cycle, one cycle.

---

## 42. The harness rows this adds

Rows: `plain ridges, joined, cyclic graph`; `cyclic graph, rendered, round 1 … final`; `flux as a test`; the λ sweep. Columns: the one per section (37 `told_apart`; 36 junction recall at matched precision; 38 core-z `told_apart` by radius and blur; 39 found-and-typed-right; 40 inferred directions right on trees), and as description: bridges right / wrong, lumens fused right, flux `z` `told_apart` with the share of readings failed, flips right / wrong, sign-constraining lumens, set-asides and tied readings per softness, directed cycles, windowless edges, per-edge RMS by radius, readings changed at step 6(d), the plexus (expected zero), the share of skeleton two-arm joins that are false. Flux columns upper bounds until the velocity profile is in.

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
| R5 1 | the plexus claims withdrawn: a crossing plexus is a no-op for signs and for inferred directions; a true anastomosis read as a crossing is a wrong structure the margin cannot see; the plexus column expected zero; the anastomotic-grid rows restated with the readings the pipeline makes; four-arm structural confidence as P(two vessels, not one lumen) from the `overlap` and core z |
| R5 2 | no 1 − ε where the structure can be wrong: two-arm joins calibrated from the pair's tests, 1 − ε only for one connected ridge before closing; three-arm vertices get the "through pair + free end" partition with the far-side evidence; an extension-made fork carries the extension's confidence whatever its arm count |
| R5 3 | the grouped set-aside by the pseudo-inverse (exact with a free-only arm); the bridge prior's fallback scale from the vessel it continues; both tested |
| R5 4 | a second solve after the readings of (d); blame, the handed-on graph and the column from it; a split vessel as separate measurements |
| R5 5 | wrong speeds not detected: stated; the studentised residuals reported with their base rate, sent nowhere |
| R5 6, R5 7 | one whitening by the noise's two-dimensional autocorrelation off the graph; per-edge RMS in the noise's along-line unit, cores by the core z, halos apart, never pooled; the core filter's reach as derived for the generator by radius and blur; the spectrum measured; capillaries added to the generator; the cluster modulation named |
| R5 8 | a tied reading has no balance and is reported; no tie is a free pass |
| R5 9 | the reach restated: lumens all of whose arms are measured or pinned; on the real pipeline, bifurcations and confluences |
| R5 10, R5 11 | the provenance wrappers named; the rule for choosing λ written |

## Open choices for you

1. The line mass for the flux: the kymograph's M or the profile's `π κ R²`. The plan uses M.
2. Composition where there is no reading: sum (the plan's choice) or max.
3. The real burst's base rate of true terminations against gaps and dives.
4. Whether the flow field (section 24) should be a second source of edges now or in the next version.
