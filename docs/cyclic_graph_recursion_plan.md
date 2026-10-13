# Plan: a clean cyclic graph from the plain ridges, rendered back and corrected by recursion

Version 4. This plan continues the burst notebook (`notebooks/development copy 2.ipynb`, sections 1–34) with sections 35–42. It has been through three adversarial reviews (`cyclic_graph_recursion_plan_review.md`, `_review_2.md`, `_review_3.md`); the last part of this document says how each finding was handled. The part of section 40 that was prototyped (`vascnet/fluxnet.py`, `tests/test_fluxnet.py`) is described as it behaves, with the numbers the tests measure, not as it was hoped to behave.

**What this plan claims, plainly.** On tree-like beds with measured arms, sections 37 and 40 should raise the junction and direction columns of the harness: the graph with confidences keeps what the consolidation dropped, and the network infers the unmeasured directions and turns lone wrong ones round (+3 to +7 points on binary trees with the thick level unmeasured, measured on the prototype). On a plexus, at the notebook's balance softness, the flux test types nothing and the sign step gains 0 to 4 points; there the plan's gains come from the graph, the core's matched filter and the bridges, which sections 38 and 39 build, and the plan says so where it applies. The parts of the recursion that the reviews showed to be unsound as first written (the residual along fitted edges, a single noise divisor, bridges that accepted or rejected gaps, a readings loop) are replaced, not patched, below.

What is already there, and the plan builds on rather than rebuilds:

- **The plain ridges**: `ridge_crests` → `centerlines` (section 29, `consolidate=False`), the "plain ridges" row of the harness. `centerlines` prunes side branches under `SPUR_LENGTH` = 12 px from the skeleton (a thinning artefact, not a piece of ridge) and takes no setting; the consolidation's *joining* steps (closing within `RIDGE_JOIN` = 2 px, `extend_ends` into a line within `RIDGE_REACH` = 16 px) and its *dropping* steps (`RIDGE_CHILD`, `LINE_LEAST`, `RIDGE_NETWORK`) are interleaved in `consolidate_lines`; section 37 separates them, which is new code.
- **The score section 29 saved**: `static_scores` is `orientation_scores`, the 16-layer score *before* the lateral interactions. The weaker line of a crossing survives in it. Section 36 reads it.
- **Arms read on a ring** (`read_arms`, section 26), from half the thickest width plus `RING_MARGIN` outwards; an edge that never leaves the core has its outer half read, inside the core. `contract_cores` already drops spurs shorter than the core from a node's ends.
- **The per-junction flux test** (`group_tests`, section 27): the net flux over `SIGMA_FLUX` × the sum of the arms' absolute fluxes. It can move a partition's score but never fail it (at most 2.2σ against `Z_PASS` = 3). A NaN in a test fails the partition (`partition_reading`), so a missing test must be left out, not passed as NaN.
- **The junction reading** (`choose_partition`): among the partitions none of whose tests is off by more than `Z_PASS`, the most specific. Groups of four or more carry no tests and fit by default; `geometric_reading` reads a planar four-way as a crossing.
- **`fuse_split_crossings`** (section 26): two three-way nodes within `FUSE_FACTOR` widths whose outer arms run straight across are one crossing already; it runs inside `vessel_graph`.
- **Whole vessels** (section 32): one kymograph and **one direction per whole vessel**, pieces joined through `join_pieces`. A wrong direction on the real burst reverses a chain of pieces, not one.
- **Flow as a likelihood ratio** (section 33) between the two best readings, and p measured without truth from the three-in junctions and the sides.
- **vascnet's renderer** (`vascnet.synth.render`): max within a tree, transmittance blurred, multiplied across trees. The notebook's generator blurs the density, moves clusters at one speed per segment, runs no background fit.

## Principles

- **Nothing is deleted below the thinning scale; everything carries a confidence; and the principle has a number.** The graph keeps every piece of ridge longer than a thinning artefact. Each stage states which confidence it works above, and that cutoff is swept. The notebook's own result (section 31) is that keeping pieces cost repeatability and found no more junctions; so the principle is a claim: on the held-out bursts, the edge confidence must tell true pieces from false ones with `told_apart` ≥ 0.75, and the confidence curve must pass the consolidated row's point. Below that, section 29's consolidation is kept and this plan's graph is not.
- **One objective, written down.** `J = χ²_w + λ Σ_elements k_e ln(n_e)`: the whitened squared residual over the valid pixels, plus a price per parameter of every drawn element, `n_e` the effective number of independent pixels the element touches (the squared norm of its whitened template over the squared norm of its raw template, times its pixel count; the whitening gives it), λ the one swept multiplier. The structure of the drawing (which edges are drawn, how they are grouped into lumens, where the centerlines lie, which halos exist) changes only by moves, and a move is taken when J falls. The parameters of a drawn element (κ, R, halo) are **measurements** read on the arms and windows, not optimised under J; they change only when the structure around them changes, and that change is itself the move that is priced.
- **A reading and the drawing never disagree.** A re-composition proposed by a junction reading is a move under J; if J vetoes it, the reading reverts to the previous one. Bridges are never drawn, so they are outside J by construction.
- **Neural layers are readings of the fields that exist.** The one new layer is the junction cell (section 36).
- **Flow decides what a still cannot, where it can.** Directions, net flux and one-vessel-one-speed are the evidence a projection of a 3-D bed needs; section 40 says exactly how far each reaches at the notebook's softness.
- **Each section has one column it has to move**, named in advance, on the held-out bursts.

---

## 35. Synthetic bursts with the things this plan is about

Extend the generator (section 28), keeping the old seeds bit-identical:

1. **Loops**: a daughter rejoins another branch of its tree with probability `SYNTH_LOOP`; its speed follows from flux conservation at both ends.
2. **A plexus**: `SYNTH_PLEXUS` thin vessels at three or more depths, with true four- and five-arm junctions and projection-only cycles. Section 40 is scored on it separately from the trees.
3. **Gaps**: a segment fades over `SYNTH_GAP_LENGTH` px to `SYNTH_GAP_DEPTH` of its density with probability `SYNTH_GAP`; recorded.
4. **Terminations** inside the field.
5. **Confluences**: both flow directions equally, as now.
6. **Halos of either sign** around thick vessels (scatter, or what a background fit leaves).
7. **Shared lumens and collinear overlaps**, recorded per junction; a few segments of two trees running along each other with nothing emerging, so that the miss section 38 accepts is counted.
8. **A velocity profile across the vessel**: clusters at radial positions, speed `v_max (1 − (r / R)^k)`, `k` = `SYNTH_BLUNT` ≈ 3. Until this is in, the harness's flux columns are upper bounds and the table says so.
9. **Blur in the transmittance domain**, as `vascnet.synth.render` and the camera do.
10. **The background fit** of section 12 run on the synthetic bursts, or a local mean of `BACKGROUND_SIGMA` subtracted with the harness marked approximate there.
11. **Per-junction `n`** in the truth (3 at a bifurcation, 4 at a crossing, the sum of arms at a fused "complex"), which section 36's column needs and `junction_metrics` does not give.

**Column:** none; per new seed, how many loops, gaps, terminations, plexus junctions and collinear overlaps, and the old seeds proven unchanged.

---

## 36. Junction cells

A junction cell at a pixel, for a ring radius ρ, counts the arms of a junction there and fires in proportion to how clearly it sees them.

**Input.** `static_scores[n]` (the pre-interaction score) and `half_width[n]`.

**Mechanism** (`junction_cells`): on a ring of radius ρ, at angles φ in steps of 360/`RING_STEPS` degrees, read the score of the neuron whose orientation is the radial direction φ (interpolated between the two nearest of the 16); peaks above `T_LOW` joined to one above `T_HIGH` are arms, two within `SAME_ARM` degrees one arm; `n` the count, `rate` the geometric mean of the peak scores in noise units times a core term. Rings `JUNCTION_RINGS` = (4, 6, 10, 16, 24) px, the smallest where `read_arms` reads a capillary junction. Non-maximum suppression within ρ/2 **per ring separately**. `n` = 1 an end cell; `n` = 2 on a line suppressed, with a turn a bend; `n` = 3 and 4 candidates for a bifurcation and a crossing, typed by section 27. **Plaid cells**: two orientations at least `PLAID_APART` = 45 degrees apart both above `T_HIGH` in the pre-interaction score. On a rendered capillary bifurcation the 4 px ring resolves daughters 40 degrees apart on the density (measured); whether it does on the oriented score is check 1.

**Checks before any cell map is built for a frame**: a thick vessel (R = 6) crossed by a thin one (R = 1.5), a thin daughter leaving a thick trunk, and a crossing broken into two T's 5–10 px apart.

**Result:** `junction_maps[n]`: `n`, `rate`, `ring`, `plaid`.

**Column:** junction recall at the plain ridges' junction precision, scored by each junction's `n` against the truth's `n` (section 35, item 11; new harness code).

---

## 37. The clean cyclic graph

**Input.** The ridges with the skeleton's own thinning (`ridge_crests` → close within `RIDGE_JOIN` → `centerlines`, which prunes thinning stubs under `SPUR_LENGTH` = 12 px and is kept as it is), then **`extend_ends` only from the free ends of pieces at least one field length long** (on an unpruned skeleton it extends several hundred 4–11 px stubs per frame into neighbouring lines; measured on a proxy), then `ridge_lines` with `least = 0`, `to_forks = True`. The dropping steps of `consolidate_lines` are not run. This is new code, since the joining and dropping steps are interleaved in `consolidate_lines`; the section says so and reports what it keeps that the consolidation dropped.

**Vertices.** Every fork and every free end; forks within one core are one vertex (`contract_cores`, `fuse_split_crossings` as in `vessel_graph`); a junction cell within a core adds its `rate`; a cell with no fork within its ring becomes a vertex with no edges yet. A free end carries `at_data_edge`.

**Edges.** Every piece between two vertices, as a spline. Twins kept and marked. Multi-edges and self-loops allowed. Zero-length pieces inside a core disappear into the vertex.

**Confidence per edge**, six terms, each calibrated on the harness on this graph alone before section 39 is written (section 41): length in field lengths; strength against `T_HIGH`; **repeatability per edge** (new code: `found_again` of section 22 applied per edge; `repeated` gives one number per fixation); `flow_share`; twin and stub (`tube_reach`); and from section 38 the rise of J without the edge.

**Confidence per vertex**: the cell's `rate`, times the reading's confidence (section 27), times the flux test's fit (section 40).

**Result:** `cyclic_graphs[n]`.

**Column:** `told_apart` of the edge confidence between true and false pieces (threshold 0.75), with the centerline and junction precision–recall curves against the plain and consolidated rows.

---

## 38. The renderer and the residual

**Per-edge measurements** (`edge_profiles`): at windows of `KNOT_SPACING` px along an edge, excluding every window whose centre lies within the core radius of either end vertex, the density profile across the edge fitted with the cylinder chord profile convolved with the fixation's measured blur (`picture_blur`): R and κ per window and an offset c; κ and R smoothed along the edge with the smoothing stopping at the core radius. For an edge thinner than the blur, R and κ are not separable: `R_od` only where `R_ridge` > σ; the line mass `π κ R²` elsewhere.

**Edges with no window** (shorter than two core radii plus a window: split crossings, plexus capillary pieces; their share is a column): κ and R come from the lumen the edge belongs to (the whole vessel of `join_pieces`, whose other pieces have windows); for an edge that is its own lumen, the line mass from `read_arm`'s outer half and R from `R_ridge`. The core filter at such an edge's vertices has reduced power, and the plan counts it.

**The renderer** (`render_graph`), following `vascnet.synth.render`: edges grouped into **lumens** (whole vessels, plus the groups of three of bifurcation readings); each lumen drawn unblurred into its own canvas, **max** within it; each canvas converted to transmittance, blurred by the fixation's σ, the lumens **multiplied** (a canvas is transmittance 1 outside its own extent, so a local canvas multiplies correctly), the picture `−log` of the product. A halo per thick edge, `h` of either sign, `σ_h` ≤ `BACKGROUND_SIGMA`, fitted thickest first.

**The whitened residual.** `residual = density_mean − rendered` on valid pixels. Its noise is the static texture plus registration, correlated over about 2 px off the graph; along existing edges it is also correlated at the knot scale by the fit's misfit. Two autocorrelations are measured per fixation, on one half of the quiet pixels off the graph and along the edges, and **each region is whitened by its own**: off-graph canvases and templates by the off-graph one, on-edge templates by the along-edge one. `χ²_w` is the squared whitened residual. The column's RMS is read on the *other* half of the quiet pixels, not on the half the whitening was measured on, where it is 1 by construction.

**Where the residual is read.**

1. **At every core**, as a matched filter: the template is the difference between the two compositions (one lumen, two vessels) rendered at that core from the arms' κ and R; its whitened inner product with the residual over its norm is a z for "two vessels" against "one lumen". A capillary crossing's core holds few independent pixels; the z is one test among the junction's others, and a change of composition is a priced move. **For a four-arm vertex this is the one test that can tell a crossing from one lumen of four**: geometry reads a planar four-way as a crossing, and the flux test at the notebook's softness passes both (section 40).
2. **Off the graph**: the hypercolumns of section 20 on the whitened residual, restricted to windows around residual ridges above `T_HIGH`.
3. **Along edges**, for what the fit could not absorb.

| residual | reading | move |
|---|---|---|
| the core filter says two vessels under one lumen, or the reverse | the composition is wrong | step 4 |
| off at both ends of an edge, good between | the vertices are mis-modelled | steps 3, 4 |
| off at one end only | a junction misread, or a free end that is a gap | steps 1, 2 |
| a bump on an edge continued by ridges off the graph on either side | a crossing vessel is missing: its tails, linked by the association field | step 5 |
| a ridge off the graph | a vessel nobody has | step 5 |

**The rule for new vessels, and the miss accepted**: evidence outside the pixels of existing edges adds a vessel; a vessel lying exactly along an existing edge with nothing emerging is not looked for, and the miss is counted (section 35, item 7).

**Result:** `rendered[n]`, `residual[n]`, `edge_profiles[n]`, per edge the rise of J without it, per core the matched-filter z, the share of windowless edges.

**Column:** `told_apart` of the core z between junctions whose true composition is one lumen and two, and the held-out whitened RMS.

---

## 39. The recursion

One round applies moves of ten kinds under `J`. λ is swept on the real burst by repeatability, loose ends, the share of impossible junctions (section 33's count) and the share of directed cycles; the held-out bursts score the λ chosen there. Rounds stop when no move lowers J, or after `ROUNDS_MOST` = 5. Since every structural change is a move and J falls at every taken move, the rounds end.

**Step 0, halos.** Thickest first; `h` of either sign.

**Step 1, bridging free ends.** Never accepted, rejected or drawn. Candidates: pairs of free ends within `BRIDGE_REACH` = 10 × the larger width (at least `RIDGE_REACH`), each end's direction in a cone of `SIGMA_BRANCH` around the other; the path a cubic Hermite spline. Confidence from similarity on the distal profiles: line mass, blur, the bend, and where there is flow, direction at reliability p, speed within `SPEED_AGREE` sigmas, pulse phase within `SIGMA_PHASE`. These terms separate thick ends and not capillary pairs (every capillary has nearly the same line mass); the flow terms need a clear speed on both sides, which short faint pieces lack; the column reports `told_apart` for thin and thick ends apart. Rivals are not resolved here. Recorded beside the confidence: the whitened residual's line mass along the path (clean, since nothing is drawn there) and whether the kymograph's pattern continues across the gap. **In the flux network** a bridge is an edge with a prior of zero flux scaled by its confidence (section 40), a free end's rival bridges are one lumen with the end's arm, and the flux lost at the free end ranks the candidates.

**Step 2, merging arms through a junction.** The partition reading of section 27 with the core filter and the flux test among its tests; arms that are one vessel joined into one lumen; the re-composition a move; the whole vessel re-read for its speed.

**Step 3, validating arms.** The junction cell's `n` against the degree: degree > n, a candidate for removal; degree < n, look for the arm in the residual's field along the cell's unexplained peak and add it as a move. Centerline shift outside cores: a move with zero parameters, taken only if J falls.

**Step 3b, undoing.** *Remove an edge from the rendering* (it stays in the graph, `drawn = False`, and can return). *Merge two vertices* within a core, proposed by the cell (`n` = 4 at the midpoint of two forks of degree 3; `fuse_split_crossings` finds the straight case already, the cell proposes the bent one too) and decided by the merged vertex's reading with the core filter and the flux test among its tests; J a veto, on which the reading reverts. *Split a vertex* of four or more arms whose groups' axes (`where_axes_meet` per group) meet apart by more than the core radius; decided the same way.

**Step 4, composition.** Where the core filter contradicts the composition by more than `Z_PASS`, the other composition is a move.

**Step 5, missing vessels.** Chains from the whitened residual's field, off the existing edges, as moves.

**Step 6, the flux network, once per round, no loop.** (a) readings from geometry and the core filter; (b) one network solve; (c) every vertex's candidate readings scored by the leave-one-vertex-out tests; (d) the readings taken once more, most specific whose tests fit, with the flux `z` able to fail a reading and never to choose one among readings that fit. No further iteration: a loop of readings and solves oscillates (measured on a grid: a two-cycle, with the flips growing and the directions worsening). The network's other findings go back up: studentised residuals lower the confidence of the speeds they belong to where the lumen is tight; a lumen set aside or reported impossible lowers its reading's confidence; a persistent imbalance marks a missing arm (step 3) or a misread (steps 2, 4); flux lost at a free end ranks the bridges (step 1).

**Result** per round: moves proposed and taken, J, the graph; the last is `graph_final[n]`.

**Column:** found-and-typed-right after each round; bridges right / wrong at the bridge confidence's cutoffs; the held-out whitened RMS.

---

## 40. Net flux = 0: a flux network (prototyped)

`vascnet/fluxnet.py`; `tests/test_fluxnet.py` on a toy graph (a bifurcation, a crossing, a confluence, a loop) and on grids with harmonic flow as a plexus. What follows is what it does and what it measured.

**Unknowns, measurements, balances.** One flux per edge; measured on edges with a clear speed, `M × v` with its sigma; free elsewhere. One balance per lumen at every vertex, allowed to miss by `SIGMA_FLUX × Σ|q|` over the lumen's arms, **the notebook's unit**. `SIGMA_FLUX` is re-measured in section 40's cell **on three-arm junctions whose three directions are measured and possible** (not under the readings it will then judge, which would widen it where they are wrong), and printed.

**The unit of a direction is a measurement, not an edge.** Section 32 reads one direction per whole vessel; the network takes a `measurement` id per edge (`join_pieces`' paths) and turns all of a measurement's edges round together at one cost. A trunk of six pieces read backwards is one move, where six independent flips were unreachable (measured: the one-at-a-time version flipped its daughters instead).

**The sizes of the balances** are taken twice: for the sign step from the measurements alone (sizing them from the solution tightened them around wrong neighbours on a plexus and turned right measurements round; measured), and for the sizes afterwards robustly (a measured arm counts the smaller of its measurement and what the rest predicts for it, so that a wrong speed cannot widen its own balance; measured with a trunk read five times too fast).

**The signs.** Within every lumen, every set of the measurements touching it is a move to turn those round, scored in closed form from the one solve's covariance, at `2 ln(p / (1 − p))` per measurement turned. **A lumen may instead be set aside** (its balance dropped) at the cost `ln(c / (1 − c))` of its reading's confidence c (section 27's; a join made by `extend_ends` or a bridge carries that extension's confidence; a three-arm vertex has no other reading and is nearly never set aside). The best move is taken while one lowers the total. **The pattern rule** (every arm measured, all in or all out) is reported per lumen and never acted on: acting on it turned a right measurement round where the lumen's reading was wrong (measured), and section 27's `in and out` test fails such a reading already.

What the tests measure:

| case | outcome |
|---|---|
| one wrong direction, contradicted at both ends | turned round |
| one wrong on a leaf, contradicted at one end (a confluence all in) | turned round at 0.45: the balance alone pays for it |
| trunk and one daughter wrong at a bifurcation | at 0.15 the two are turned together; at 0.3 the join the wrong daughter passes is set aside and reported instead (one join wrong is likelier a priori than two directions); at 0.45 one is turned and a 1.8σ imbalance or an impossible lumen is left and reported. In no case does a right measurement cascade |
| two vessels ending near each other, read as one passing through (a join at confidence 0.8) | the join is set aside and reported impossible; no right measurement is turned. At the default join confidence 0.99 the cheaper explanation is one wrong direction, and the network turns it |
| a whole vessel of two pieces through a crossing, read backwards | turned round as one at 0.3 and 0.15 (at 0.45 the two balances of 1.3σ do not quite pay for it) |
| grids 8×8 to 16×16, 70% measured, 15% wrong, three seeds | measured directions right +0 to +4 points (seed 3 of 8×8: 0, no flips); flips 2:0, 2:0, 0:0, 8:1, 7:1, 9:2, 11:1; at p = 0.6, 12:3; at p = 0.95, 5:0 |
| binary trees with the thick level unmeasured, 15% wrong | +3 to +7 points at 0.45, near-perfect at 0.15 (measured by the third review) |

**The sizes.** Huber weights on the balance residuals and the measurement misfits; per measurement the studentised residual against what the rest predicts for it. **Its reach**: it finds a wrong speed where the lumen around it is tight (a trunk against two measured daughters: 3.7σ); on a plexus with 70% of the edges measured it finds none of ×3 / ÷3 outliers (measured), and the plan does not claim it there.

**The tests.** Per lumen, its imbalance with its own balance not enforced. For a vertex, every candidate reading scored from one leave-one-vertex-out solve (a rank-k update; the scores do not depend on the reading the solve enforced, also with a wrong speed and a flipped daughter: measured). These `z` enter `group_tests` as the `flux` test, **left out where NaN** (a NaN fails a partition in `partition_reading`). Section 27 takes the most specific reading whose tests fit. The network's `z` can fail a partition where the notebook's test could not, but **at the notebook's softness it fails little**: on a single-depth plexus with every direction right, every four-arm vertex passes as a crossing by flux alone, at 0.45 and at 0.15 (measured). The flux test therefore only *fails* readings; it never chooses among readings that fit, and a four-arm vertex's type is the core filter's and geometry's to decide (section 38). No χ² ratio between readings (`compare_readings` refuses unequal balance counts).

**Doubtful edges.** A bridge with confidence c gets a prior of zero flux with sigma `c / (1 − c)` times the graph's typical flux: at c = 0.02 it carries under 0.5 of a flux of 4 and the mismatch it would have hidden is exposed at over 2σ; a softened balance instead kept it conducting and divided the mismatch away (measured). A free end's rival bridges are one lumen with the end's arm, so its flux is not counted twice and a worthless rival carries nothing. `flux_at_free_ends` reports what arrives at every free end.

**Directed cycles.** Strongly connected components over edges with `|q| / σ` ≥ 3 (unmeasured plexus edges sit near 0.7 and would make cycles of noise), one cycle per component; a report, never a measure of p, and neither are flips, impossible or set-aside lumens. p comes from section 33's counts only.

**Cost.** One dense solve per fixation per round: 3 s at a thousand edges, 20–27 s at two thousand (measured); the sign moves in closed form; the vertex tests k × k; a set-aside move costs one re-solve and is rare. Sparse factorisation beyond a few thousand edges.

**Column:** directions right among inferred edges against `propagate_directions`, on the tree bursts and on the plexus bursts separately; flips right / wrong; `told_apart` of the flux `z` for junctions typed right against wrong, reported with the share of readings it failed.

---

## 41. Confidences

Every confidence is a sum of log-likelihood terms, each calibrated apart from any prior on the development bursts (reliability diagrams, `told_apart`), the held-out bursts saying whether it held; the bottom-up terms of section 37 first, on section 37's graph. The prior (base rates of gaps, terminations, loops) is explicit and separate: the generator's setting on synthetic bursts, open choice 3 on the real one.

The graph handed on: `vessel_digraphs[n]`, per edge the spline, `R_ridge`, `R_od` where it exists, line mass, blur, velocity ± σ, direction (measured or inferred) with `|q| / σ`, flux ± σ, the edge confidence and its terms, `drawn`, flags; per vertex position, kind, type, partition, confidences by geometry, flow and flux, `impossible`, `set aside`, imbalance; per component with a directed cycle, one cycle.

---

## 42. The harness rows this adds

Rows: `plain ridges, joined, cyclic graph` (curves over edge confidence); `cyclic graph, rendered, round 1 … final`; `flux as a test`; the λ sweep on the real burst. Columns: bridges right / wrong at cutoffs, lumens fused right, directions right (measured, inferred; trees and plexus apart), flux `z` `told_apart` with the share of readings failed, flips right / wrong, directed cycles, the share of windowless edges, held-out whitened RMS. Flux columns marked upper bounds until section 35's velocity profile is in. One column per section is quoted; the rest is description.

---

## The order of work

1. Section 37 on the real burst and the harness; the first principle's number.
2. Section 40 on `whole_graphs` as they are: `SIGMA_FLUX` re-measured on possible three-arm junctions; inferred directions, studentised residuals, imbalances, set-aside and impossible lumens, cycles and flux at free ends on the real burst; inferred directions scored against `propagate_directions` on trees and plexus.
3. Section 36's three checks.
4. Sections 35 and 38 on one synthetic burst, then section 39 round by round, then the harness.

---

## How the reviews' findings were handled

| finding | handling |
|---|---|
| R1 A1, R2 4, R3 5 | whole-network residual; windows and smoothing stop at the core radius; windowless edges take their lumen's κ and R or `read_arm`'s outer half, and are counted; core read by a matched filter, priced |
| R1 A2, R2 3, R3 10 | whitening per region by its own measured autocorrelation; `n_e` from the whitened template's norm; the column's RMS on a held-out half of the quiet pixels |
| R1 A3, R2 10, R3 10 | removal, merge, split as moves; merge and split proposed by the cell and decided by the reading, J a veto on which the reading reverts; parameters are measurements, structure changes only by moves |
| R1 A4, R2 1, R3 2 | leave-one-vertex-out scores from one solve; the flux `z` fails readings and never chooses among fitting ones; no loop in step 6; NaN left out; four-arm vertices typed by the core filter and geometry |
| R1 A5, R2 5, R3 4 | bridges never accepted, rejected or drawn; in the network a prior of zero flux by confidence, rivals one lumen, flux lost at free ends reported |
| R2 2, R3 1, R3 3, R3 8 | whole-vessel flips; set-aside move at the reading's confidence; the pattern rule reported, not acted on; measurement-only sizing for the signs; the sign step's reach and the detector's reach stated; grid numbers as ranges over seeds and p |
| R2 6, R3 13 | the notebook's unit; `SIGMA_FLUX` re-measured on possible three-arm junctions |
| R2 7 | robust sizing; Huber on balances; studentised residuals with their reach stated |
| R2 8, C2, R3 11 | SCC cycles with cutoff 3 |
| R2 9 | transmittance blur in renderer and generator; local canvases multiply as transmittance 1 outside |
| R2 11, B7 | λ the swept multiplier; impossible junctions and cycles in the sweep |
| R2 12, R3 12 | per-edge repeatability named as new code; plaid cells on the pre-interaction score |
| R2 13, B8, R3 6 | the skeleton's thinning kept (`SPUR_LENGTH` = 12), `extend_ends` only from ends of pieces one field length long, the separation of joining from dropping named as new code; `told_apart` ≥ 0.75 |
| R2 14, R3 9 | tests for leaf, two-wrong (as the model's arithmetic), whole vessel, wrong join, bridge prior, rivals, free ends, plexus over seeds |
| R2 15, R3 12 | cost measured: 3 s at a thousand edges, 20–27 s at two thousand; pairs among the flip sets above six |
| R2 16, B9, R3 14 | rings from 4 px, NMS per ring; the 4 px ring resolves 40-degree daughters on the density, the score is check 1; per-junction `n` in the truth |
| R2 17, C5 | prior explicit and separate; bottom-up terms calibrated first |
| R2 18, B1 | `flux as a test`; velocity profile; flux columns upper bounds until then |
| R3 7 | the RMS column on a held-out half |
| B2, B3, B4, B6 | σ per fixation and line mass for thin vessels; per-lumen canvases before a transmittance blur; halo of either sign bounded by the background's sigma; flips not counted as p |
| C3 | the flow field as a second source of edges: not in this version |

## Open choices for you

1. The line mass for the flux: the kymograph's M or the profile's `π κ R²`. The plan uses M.
2. Composition where there is no reading: sum (the plan's choice) or max.
3. The real burst's base rate of true terminations against gaps and dives.
4. Whether the flow field (section 24) should be a second source of edges now or in the next version.
