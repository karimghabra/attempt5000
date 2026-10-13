# Plan: a clean cyclic graph from the plain ridges, rendered back and corrected by recursion

Version 3. This plan continues the burst notebook (`notebooks/development copy 2.ipynb`, sections 1–34) with sections 35–42. It has been through two adversarial reviews (`cyclic_graph_recursion_plan_review.md`, `cyclic_graph_recursion_plan_review_2.md`); the last part of this document says how each finding was handled. The part of section 40 that was prototyped (`vascnet/fluxnet.py`, `tests/test_fluxnet.py`) is described as it behaves, with numbers from the tests, not as it was hoped to behave.

What is already there, and the plan builds on rather than rebuilds:

- **The plain ridges**: `ridge_crests` → `centerlines` (section 29, `consolidate=False`), the "plain ridges" row of the harness. The consolidation's *joining* steps (closing within `RIDGE_JOIN` = 2 px, `extend_ends` into a line within `RIDGE_REACH` = 16 px) drop nothing and are kept; its *dropping* steps (side branches under `RIDGE_CHILD`, loose pieces under `LINE_LEAST`, networks under `RIDGE_NETWORK`) are what this plan replaces with confidences.
- **The score section 29 saved**: `static_scores` is `orientation_scores`, the 16-layer score *before* the lateral interactions of section 21. The weaker line of a crossing survives in it. Section 36 reads it.
- **Arms read on a ring** (`read_arms`, section 26): angle, width, density, strength, and with flow the line mass, velocity, direction, pulse phase, blur trace. The ring starts at half the thickest width plus `RING_MARGIN`; for an edge that never leaves the core, `read_arm` reads its outer half, inside the core. Section 38 says what that means for the fit.
- **The per-junction flux test** (`group_tests`, section 27): the net flux over `SIGMA_FLUX` × the sum of the arms' absolute fluxes. It can move a partition's score but never fail it (`|Σ| / Σ|·| ≤ 1` gives at most 2.2σ against `Z_PASS` = 3). Section 40 keeps its unit and replaces its reach.
- **The junction reading** (`choose_partition`, section 27): among the partitions none of whose tests is off by more than `Z_PASS`, the one with the most groups, the most specific. Groups of four or more carry no tests and fit by default. Section 40's test enters here.
- **`fuse_split_crossings`** (section 26): two three-way nodes within `FUSE_FACTOR` widths whose outer arms run straight across are one crossing already; it runs inside `vessel_graph`. Section 39's merge says what it adds.
- **Flow as a likelihood ratio between the two best readings** (section 33), the reliability p measured without truth from the three-in junctions and the sides of whole vessels, the whole-vessel speeds with widened sigmas (section 32), and the harness with held-out bursts (sections 30, 34).
- **vascnet's renderer** (`vascnet.synth`): `_branch_od` is the unblurred chord profile `OD(d) = 2 κ sqrt(R² − d²)`; `render` takes the max within a tree, converts to transmittance, blurs the transmittance, multiplies across trees. The notebook's own generator blurs the density instead (section 28). Section 38 follows `render`, and section 35 makes the generator do the same.

## Principles

- **Nothing is deleted; everything carries a confidence; and the principle has a number.** The graph keeps every piece of ridge. Each stage states which confidence it works above, and that cutoff is swept, so the numbers reported are curves over confidence. The notebook's own result (section 31) is that keeping pieces cost repeatability and found no more junctions; so the principle is a claim: on the held-out bursts, the edge confidence must tell true pieces from false ones with `told_apart` ≥ 0.75, and the confidence curve must pass the consolidated row's point (same junction recall at no lower precision, or the reverse). Below that, section 29's consolidation by length is kept and this plan's graph is not.
- **One objective, written down, and every change to the rendering is a move under it.** `J = χ²_w + λ Σ_elements k_e ln(n_e)`: the whitened squared residual over the valid pixels, plus a price per parameter of every drawn element, `n_e` its effective number of independent pixels, λ the one swept multiplier. Adding, removing, merging, splitting, shifting, re-composing and fitting a halo are all moves; a move is taken when J falls. Nothing that is drawn changes outside J. Bridges are never drawn, so they are outside it by construction.
- **Neural layers are readings of the fields that exist, not new fields.** The one new layer is the junction cell (section 36).
- **Flow decides what a still cannot.** Directions, net flux and one-vessel-one-speed are the evidence a projection of a 3-D bed needs.
- **Each section has one column it has to move**, named in advance, on the held-out bursts. Everything else in the table is description.

---

## 35. Synthetic bursts with the things this plan is about

The generator (section 28, `synthetic_network`, `synthetic_burst`) makes three trees that each split once, moves clusters along each segment at one speed, blurs the density per tree, and skips the background fit. Nothing in it has a cycle, a gap, a confluence, a shadow, a velocity profile or a background. Extend it, keeping the old seeds bit-identical (sections 30 and 34 must reproduce):

1. **Loops.** A daughter rejoins another branch of its tree with probability `SYNTH_LOOP`; its speed follows from flux conservation at both ends.
2. **A plexus.** A region of `SYNTH_PLEXUS` thin vessels at three or more depths, so that four- and five-arm junctions and projection-only cycles have a truth.
3. **Gaps.** A segment fades over `SYNTH_GAP_LENGTH` px to `SYNTH_GAP_DEPTH` of its density with probability `SYNTH_GAP`; recorded, so that a bridge can be right or wrong.
4. **Terminations** inside the field, so that a free end is not always a gap.
5. **Confluences**: trees run either way already (`rng.choice([-1, 1])`); keep both equally.
6. **Halos of either sign**: a wide shallow density `SYNTH_HALO_DEPTH × OD_peak` of sigma `SYNTH_HALO_WIDTH × R` around thick vessels, positive (scatter) or negative (what a background fit leaves), so that section 38's halo term has both to find.
7. **Shared lumens and collinear overlaps**: already the generator's physics (max within a tree, sum across); record per junction which it is, and make a few segments of two trees run along each other for a stretch with nothing emerging, so that the miss section 38 accepts is counted.
8. **A velocity profile across the vessel.** Clusters get a radial position; their speed is `v_max (1 − (r / R)^k)` with `k` = `SYNTH_BLUNT` (about 3, blunted as red cells are). The kymograph then reads something between the centerline speed and the mean, as on the real burst, and `M × v` is conserved only up to the size-dependent factor that `SIGMA_FLUX` has to absorb. Without this the harness's flux numbers are an upper bound, and the table must say so.
9. **Blur in the transmittance domain**, as `vascnet.synth.render` does and the camera does: the density is converted to transmittance, blurred, multiplied across depths, and converted back. The difference from blurring the density is a negative streak at every thick vessel's flank (Jensen's gap, about half the variance of the density across the kernel), which the real burst has and the generator did not.
10. **The background fit.** Run section 12's fit on the synthetic bursts as on the real one, so that a halo competing with the background (section 38) has a truth. If that is too slow for the harness, the generator applies a local-mean subtraction of `BACKGROUND_SIGMA` instead, and the plan says the harness is approximate there.

**Column:** none; the section lists per new seed how many loops, gaps, terminations, plexus junctions and collinear overlaps it has, and proves the old seeds unchanged.

---

## 36. Junction cells

A junction cell at a pixel, for a ring radius ρ, counts the arms of a junction there and fires in proportion to how clearly it sees them.

**Input.** `static_scores[n]` (the score before the interactions) and `half_width[n]`.

**Mechanism** (`junction_cells(scores, half_width, rings)`):

- On a ring of radius ρ, at angles φ in steps of 360/`RING_STEPS` degrees, read the score of the neuron whose orientation is the radial direction φ, interpolated between the two nearest of the 16. The scale competition has already chosen a width per pixel.
- Peaks of that angular profile above `T_LOW`, joined to one above `T_HIGH`, are arms; two peaks within `SAME_ARM` degrees are one arm.
- `n` = the number of peaks. `rate` = the geometric mean of the peak scores in noise units, times a core term (the core's density against the arms': a junction is at least as dark as its arms).
- Rings: `JUNCTION_RINGS` = (4, 6, 10, 16, 24) px. The smallest is where `read_arms` reads a capillary junction (half-width 1.5–2 px plus `RING_MARGIN` 2). Per pixel the ring with the largest rate is kept, with its `n`.
- Non-maximum suppression within ρ/2, **per ring separately**, so that a thin daughter's junction 6 px from a thick trunk's is not suppressed by the 24 px ring.
- `n = 1` is an end cell; `n = 2` within `STRAIGHT` degrees of a line is a line cell and is suppressed; `n = 2` with a turn is a bend; `n = 3` and `n = 4` are candidates for a bifurcation and a crossing; the type is read by section 27's partitions.
- **Plaid cells**: pixels where two orientations at least `PLAID_APART` = 45 degrees apart both exceed `T_HIGH` in `static_scores`. Being the pre-interaction score, the weaker line is there; how much of it survives the tuning is not the question, because the tuned field is not read here.

**Result:** `junction_maps[n]`: `n`, `rate`, `ring`, `plaid`.

**Checks, before any cell map is built for a whole frame:**

1. A synthetic crossing of a thick (R = 6) and a thin (R = 1.5) vessel, and a thin daughter leaving a thick trunk: does the thin arm appear as a peak on the ring that matches the thick vessel, at what rate, and on its own ring? If not, a second reading of the ring at the thin width is added, and the plan says so.
2. A crossing the skeleton broke into two T's 5–10 px apart: does the cell at the midpoint say `n` = 4?

**Column:** junction recall at the plain ridges' junction precision, scored through each junction's `n` and position (the harness's gate `max(6, 2R + 2)` matches a cell and two forks alike, so the per-junction `n` is what is scored).

---

## 37. The clean cyclic graph

**Input.** The plain ridges with the consolidation's joining steps and none of its dropping steps (`ridge_crests` → close within `RIDGE_JOIN` → `centerlines` with `SPUR_LENGTH` = 0 → `extend_ends`; `ridge_lines` with `least = 0`, `to_forks = True`), `junction_maps`, the other fixations, `flow_masks`.

**Vertices.** Every fork and every free end; then forks within one core are one vertex (`contract_cores`, with `fuse_split_crossings` as it runs in `vessel_graph`); a junction cell within a core adds its `rate` to the vertex; a cell with no fork within its ring becomes a vertex with no edges yet. A free end carries `at_data_edge`.

**Edges.** Every piece between two vertices, as a spline (`spline_vessel`). Twins (two edges leaving a vertex side by side within `SAME_ARM` degrees and a width) are kept, the shorter marked `twin` and not an arm. Zero-length pieces between forks of one core disappear into the vertex; that is the one thing not kept, and it carries no data. Multi-edges and self-loops are allowed.

**Confidence per edge**, bottom-up, each term a log-likelihood ratio calibrated on the harness *before section 39 is written* (section 41), then summed and squashed:

- length in field lengths (`ELONGATION` × half-width; zero at one, full by three);
- strength: the median score in noise units against `T_HIGH`;
- repeatability: the share of the edge's points found again in the other fixations above chance, **new code** (`repeated` of section 29 gives one number per fixation; `found_again` is adapted per edge);
- flow data: `flow_share`;
- twin and stub: lowered for a twin, and for a free-ended piece inside a parent's tube (`tube_reach`);
- from section 38: the rise of J when the edge is not drawn.

**Confidence per vertex**: the cell's `rate`, later multiplied by the reading's confidence (section 27) and the flux test's fit (section 40).

**Clean, as numbers**: no two vertices within one core; no edge under two pixels; every edge a spline with a radius profile; the cycle count (edges − vertices + components) with the share of cycles under `RING_LEAST` px in circumference.

**Result:** `cyclic_graphs[n]`: `vertices`, `edges` (with confidence and its terms), `cycles`.

**Column:** `told_apart` of the edge confidence between true and false pieces, with the threshold of the first principle (0.75); and the centerline and junction precision–recall curves against the plain and consolidated rows.

---

## 38. The renderer and the residual

**Per-edge measurements** (`edge_profiles`): at windows of `KNOT_SPACING` px along every edge, **excluding every window whose centre lies within the core radius of either end vertex**, the density profile across the edge over ±`PROFILE_REACH` half-widths (`profile_across`), fitted with the cylinder chord profile convolved with the fixation's blur: parameters R and κ per window, an offset c, and σ fixed to the fixation's measured blur (section 31's `picture_blur`). κ and R are smoothed along the edge with knots `RADIUS_KNOTS` × `KNOT_SPACING` apart, the smoothing **also stopping at the core radius**, so that no value inside a core is a fit. For an edge thinner than the blur, R and κ are not separately identifiable: `R_od` is reported only where `R_ridge` > σ, and the line mass `π κ R²`, which is identifiable, is what every later stage uses for thin vessels. Two radii, `R_ridge` from the scale competition and `R_od` from the profile, disagree only above the blur, and there a disagreement beyond `SIGMA_WIDTH` in log marks an overlap, a halo or a twin.

**The renderer** (`render_graph(graph, shape)`), following `vascnet.synth.render`:

1. The edges are grouped into **lumens**: the whole vessels of `join_pieces` plus the groups of three of every bifurcation reading. Each lumen is drawn unblurred into its own canvas as chord density along its splines, **max** within the lumen.
2. Each lumen canvas is converted to transmittance `exp(−OD)`, blurred by the fixation's σ, and the lumens are **multiplied**; the picture is `−log` of the product. Densities add across lumens, unite within one, and the blur is where the camera's is. Composing after blurring, as the first version did, is wrong exactly at the cores where the composition is judged.
3. A halo per thick edge: `h × Gaussian(σ_h)` along the spline, `h` of either sign, `σ_h` ≤ `BACKGROUND_SIGMA` (anything wider is in the background fit already), fitted first, thickest first.
4. Canvases are small and local, so a move that touches two lumens re-renders two canvases; the product is updated locally.

**The whitened residual.** `residual = density_mean − rendered` on valid pixels. Its noise is the static tissue texture (which does not average away) plus what registration added, correlated over about 2 px of grain, and along existing edges it is also correlated at the knot scale by the fit's own misfit. Its autocorrelation is measured twice per fixation, on the quiet pixels off the graph and along the edges, and **the residual is whitened by it** (one FFT per canvas): `χ²_w` is the squared whitened residual. Dividing a per-pixel χ² by one correlation area is not a substitute: the inflation of χ² by correlated noise depends on the proposal's shape (about 12× for a one-pixel line, about 50× for a halo), and one divisor throws away three quarters of a capillary's evidence.

**Where the residual is read.**

1. **At every core**, as a matched filter: the template is the difference between the two compositions (one lumen against two) rendered at that core; its inner product with the whitened residual, over its norm, is a z-score for "two vessels" against "one lumen". A core of a capillary crossing holds few independent pixels, so this z is one test among the junction's others (section 27), not a verdict, and a change of composition is a priced move under J like any other.
2. **Off the graph**: the hypercolumns of section 20 on the whitened residual; its ridges are candidate vessels. The thick vessels are gone from it, so a thin vessel at a thick one's flank, which the scale competition hands to the thick vessel in the data, stands alone here.
3. **Along edges**, for what the fit could not absorb: features shorter than the knot spacing, cross-profiles that one blurred cylinder cannot make, and the ends.

| residual | reading | move |
|---|---|---|
| the core's matched filter says two vessels under a one-lumen composition, or the reverse | the composition is wrong | step 4 |
| off at both ends of an edge, good between | the vertices are mis-modelled: composition, core radius, or a missing arm | steps 3, 4 |
| off at one end only | a junction misread, or a free end that is a gap | steps 1, 2 |
| a bump on an edge continued by ridges off the graph on either side | a crossing vessel is missing: its two tails, linked through by the association field | step 5 |
| a ridge off the graph | a vessel nobody has | step 5 |

**The rule for new vessels, and the miss accepted.** A new vessel is proposed only on evidence outside the pixels of existing edges: the residual's ridges off the graph, including the tails of a shallow crossing. A vessel lying exactly along an existing edge with nothing emerging leaves no trace (the fit absorbed it into κ) and is not looked for; the miss is a stated limit of the still, counted on the harness (section 35, item 7). The kymograph could see it (two families of streaks where one vessel has one); that stays a later, separate test that adds no vessel by itself.

**Result:** `rendered[n]`, `residual[n]` (whitened and raw), `edge_profiles[n]`, per edge the rise of J without it, per core the matched-filter z.

**Column:** the whitened residual's RMS off the graph (1 where the model is right), and `told_apart` of the core z between junctions whose true composition is one lumen and two.

---

## 39. The recursion

One round applies moves of ten kinds under the objective

`J = χ²_w + λ Σ_e k_e ln(n_e)`,

`k_e` the parameters of a drawn element and `n_e` the effective number of independent pixels it touches (its pixels over the correlation area of the noise there, which the whitening gives). A move is taken when J falls. λ is the one swept setting; section 42 sweeps it on the real burst by repeatability, loose ends, the share of impossible junctions (section 33's count) and the share of directed cycles, the last two because they rise when something systematic is added and repeatability does not see it. Rounds stop when none of the moves lowers J, or after `ROUNDS_MOST` = 5. J falls at every taken move, every move is one of the ten, and nothing drawn changes otherwise, so the rounds end; the readings' loop of step 6 is bounded on its own.

**Step 0, halos.** Thickest first; `h` of either sign.

**Step 1, bridging free ends.** A bridge is never accepted or rejected, and **never drawn**. It is a proposed continuation of a vessel across a gap, with a confidence that says only how alike the vessel looks on the two sides:

- *Candidates*: every pair of free ends not at the data edge, within `BRIDGE_REACH` = 10 × the larger width (at least `RIDGE_REACH`), each end's direction pointing into a cone of `SIGMA_BRANCH` around the other. The path is a cubic Hermite spline between the two ends' directions, radius and line mass interpolated.
- *Confidence from similarity*, read on the distal profiles outside the cores: line mass within `SIGMA_DENSITY` in log (not radius, which thin vessels do not have), blur alike, the bend each end makes within `SIGMA_TURN`, and where both sides have flow, direction agreeing along the bridge at reliability p, speeds within `SPEED_AGREE` sigmas, pulse phases within `SIGMA_PHASE`. These terms do not separate capillary pairs well, since every capillary has nearly the same line mass; the flow terms do, where there is flow, and section 42 reports the bridge confidence's `told_apart` for thin and thick ends apart. Distance enters through the bend and the field-length term.
- *Rivals* are not resolved: an end with two candidates has two bridges. The flux network weighs them (section 40: a bridge's lumens are softened by 1 / confidence), and the whole-vessel joining of section 32 takes the stronger.
- *Recorded beside the confidence*: the whitened residual's line mass along the path, clean because nothing is drawn there, and whether the kymograph's pattern continues across the gap. Separate fields until the harness says what they are worth.

**Step 2, merging arms through a junction.** The partition reading of section 27 says which arms are one vessel; its tests gain the core's matched filter (section 38) and the network's flux test (section 40), and the arms are compared by their distal profiles. Arms that are one vessel are joined (`join_pieces`) into one lumen; the re-composition is a move under J; the whole vessel is re-read for its speed.

**Step 3, validating arms.** At every vertex compare the junction cell's `n` with the graph's degree:

- degree > n: an arm the cell does not see; its rise of J without it is already known; if small, confidence down, and it is a candidate for removal (step 3b).
- degree < n: an arm the cell sees that the skeleton lacks; look in the residual's field along the cell's unexplained peak for a ridge within `RIDGE_REACH`; string it; add it as a move.
- Centerline shift: outside every core, an edge whose rendered crest is off the data's crest by more than half a pixel is moved half the way; **a move under J with zero parameters added**, taken only if J falls.

**Step 3b, undoing: removal, merge and split.**

- *Remove an edge from the rendering*: a move; the edge stays in the graph with `drawn = False` and its confidence down, and can be drawn again later.
- *Merge two vertices*: proposed by the junction cell (`n` = 4 at the midpoint of two forks of degree 3 within a core, which `fuse_split_crossings` also finds when the outer arms run straight within `FUSE_STRAIGHT`; the cell proposes it also where they bend) and decided by the partition reading of the merged vertex with the core's matched filter and the flux test among its tests; the rendering under the new reading is then a move under J, which can veto but not propose. The core's few pixels are too little to decide a merge by themselves; the reading decides, J checks.
- *Split a vertex*: proposed where a vertex of four or more arms is read with a partition whose two groups' axes (`where_axes_meet` applied per group, not per vertex) meet at two places apart by more than the core radius; decided as the merge is.

**Step 4, composition.** At every vertex whose matched-filter z contradicts its composition by more than `Z_PASS`, the other composition is a move under J; where a halo makes the density weak, the radius continuity and the flow decide through the reading.

**Step 5, missing vessels.** The hypercolumns on the whitened residual, restricted to windows around residual ridges above `T_HIGH` (not the whole frame every round); chains longer than one field length off the existing edges are moves under J; the rule of section 38 holds.

**Step 6, the flux network and the readings.** A coordinate ascent, stated: (a) readings from geometry and the core filter at every vertex; (b) one network solve (section 40); (c) every vertex's candidate readings scored by the network's leave-one-vertex-out tests; (d) the readings taken again, most specific that fits; repeat (b)–(d) until no reading changes or three times. Its findings go back up: studentised residuals lower the confidence of the speeds they belong to; a persistent imbalance under every reading marks a missing arm (step 3) or a misread (steps 2, 4); flux lost at a free end ranks the bridges (step 1).

**Result** per round: moves proposed and taken, J, and the graph; the last is `graph_final[n]`.

**Column:** found-and-typed-right on the held-out bursts after each round; bridges right / wrong at the bridge confidence's cutoffs; the whitened RMS.

---

## 40. Net flux = 0: a flux network (prototyped)

The per-junction version exists (`group_tests`); the network version is weighted least squares on the whole graph, and is what gives the recursion something to send back up. `vascnet/fluxnet.py` implements it; `tests/test_fluxnet.py` runs it on a toy graph (a bifurcation, a crossing, a confluence, a loop) and on grids with harmonic flow as a plexus. What follows is what it does.

**Unknowns and measurements.** One flux `q_e` per edge, signed along its points; measured on edges with a clear speed, `q̂_e = M_e × v_e` with `σ_e` from the widened velocity sigma and the spread of M; free elsewhere.

**Balances.** One per lumen at every vertex: `Σ s_e q_e = 0`, `s_e` = +1 out, −1 in (the sign of `velocity_out`). A balance may miss by `SIGMA_FLUX × Σ|q|` over the lumen's arms, **the notebook's unit** (a share of the absolute sum, as `group_tests` divides). The number 0.45 is quoted as measured on the real burst's bifurcations; the notebook stores no outputs, so section 40's cell re-measures it on the whole-vessel graphs and prints it. A bridge's lumens are softened by 1 / its confidence.

**The sizes of the balances** are taken twice. For the sign step, from the measurements alone (an unmeasured arm at its first-solve flux): the sign step must not feed on its own solution, which on a plexus tightens the balances around a wrong neighbour and turns right measurements round (measured: it did). For the sizes afterwards, robustly: a measured arm counts the smaller of its measurement and what the rest of the network predicts for it, so that a wrong speed cannot widen its own lumen's balance to fit itself (measured: a trunk read five times too fast kept weight 1 and hid, before this).

**The signs.** A wrong direction is a flipped sign, not noise, with probability 1 − p (section 33's p; the prototype takes p as given). Within every lumen, every set of its measured arms is one move to turn those measurements round; the change of the minimised χ² is in closed form from the one solve's covariance (no re-solve), the cost is `2 ln(p / (1 − p))` per turned measurement, and the **pattern rule** of section 33 is a hard constraint: a lumen with every arm measured and all in or all out cannot be the truth, so a move that leaves one is never the end and a move that makes one is never taken. The best move is applied while one lowers the total; the flips a lumen's pattern forced are reported per lumen as evidence against the reading that made the lumen. Measured on the tests:

| case | one measurement at a time (before) | sets within a lumen with the pattern rule |
|---|---|---|
| one wrong direction, contradicted at both ends | turned round | turned round |
| one wrong on a leaf, contradicted at one end | not tested | turned round |
| trunk and one daughter wrong at a bifurcation | the three right measurements turned, tree reversed | at `SIGMA_FLUX` ≤ 0.3 the two turned together; at 0.45 one is turned and a 1.8σ imbalance is left and reported (one wrong direction is six times likelier a priori than two, and the balances are too soft to pay for the second) |
| grids 8×8 to 16×16, 70% measured, 15% wrong | as many wrong flips as right, directions no better or worse | directions right +2 to +4 points, flips at least 7 : 1 right |

**The sizes.** Huber weights on the balance residuals (a lumen with a missing arm must not bend its neighbours) and on the measurement misfits. Per measurement, the studentised residual against what the rest of the network predicts for it (the leave-one-out identity) is the wrong-speed detector (measured: the five-times trunk at 4σ, the rest under 2σ, the bifurcation's balance weighted down rather than the daughters bent).

**The tests.** Per lumen, its imbalance with its own balance not enforced (the balance row's studentised residual; NaN where nothing but its own balance says what its edges carry). For a vertex, **every candidate reading is scored from one leave-one-vertex-out solve**: its rows are taken out of the covariance by a rank-k update (Woodbury, k × k per vertex), and every candidate lumen gets `z = bᵀq₋ᵥ / sqrt(bᵀC₋ᵥb + s²)`. The scores do not depend on which reading the solve enforced at the vertex (tested). These `z` enter `group_tests` as the `flux` test, with the network's spread; section 27 then takes the most specific reading whose tests all fit. **No χ² ratio between readings**: a reading with fewer balances always fits at least as well (one lumen of four arms against a crossing), so `compare_readings` refuses unequal balance counts. The network's `z` can fail a partition where the notebook's test could not; that is a behaviour change, and section 42 reports both.

**What the test can see.** With `SIGMA_FLUX` = 0.45, two through-vessels of 6 and 2 give 2.2σ against the wrong pairing and 6 and 5 give nothing; at 0.15, 6.7σ. The flux's evidence at a crossing is worth something only where the two vessels differ by more than about the balance's softness. If the line mass improves (section 38's `π κ R²` against the kymograph's M), the softness can come down and the evidence grows directly.

**Directed cycles.** Blood does not go round a loop; a directed cycle holds a wrong direction or a crossing read as one lumen. Found as strongly connected components over edges whose `|q| / σ` is above a cutoff (Tarjan, linear; one cycle per component), never by enumeration (exponential on a plexus). **Reported, never counted as a measure of p**, and neither are the flips: both are the network's decisions, and counting them as evidence would assume the network right. p comes from section 33's three-in junctions and sides only.

**What comes out, per edge**: `q`, `σ_q`, direction (measured or inferred) with `|q| / σ_q` as its confidence, `measured`, `flipped`, `studentised`; per lumen `imbalance`, `forced`; per vertex the candidates' `z`; the components with directed cycles; flux lost at every free end inside the data. The digraph's thickest vessels are above the trackable limit on the real burst, so their directions are inferred, not measured; the picture draws measured and inferred differently (`measured`).

**Cost.** One dense solve per fixation per reading iteration (a few seconds at a thousand edges; the covariance is E × E), the sign moves in closed form, the vertex tests k × k. Sparse factorisation with selected-inverse diagonals when a graph passes a few thousand edges.

**Column:** directions right among inferred edges against `propagate_directions`; `told_apart` of the flux `z` for junctions typed right against wrong; flips right / wrong on the plexus bursts.

---

## 41. Confidences: what they mean, and the digraph that is handed on

Every confidence is a sum of log-likelihood terms. Each term is **calibrated apart from any prior**: on the development bursts, binned, the share right against the bin's value (a reliability diagram), with `told_apart` as the one number; the held-out bursts say whether it held. The bottom-up terms of section 37 are calibrated on section 37's graph alone, before section 39 is written. The prior, the base rate of gaps, terminations and loops, is kept explicit and separate: on the synthetic bursts it is the generator's setting, on the real burst it is open choice 3, and a bridge confidence of 0.8 means "the similarity is 4 : 1 for a continuation", not "right 80% of the time", until the prior is known.

The graph handed on: `vessel_digraphs[n]` with, per edge, the spline, `R_ridge`, `R_od` where it exists, the line mass, blur, velocity ± σ, direction (measured or inferred) with its confidence, flux ± σ, the edge confidence and its terms, `drawn`, and `bridge` / `twin` / `stub` / `found in residual`; per vertex, position, kind, type, the partition, confidence by geometry, with flow, with flux, the forced flips and the imbalance; per component with a directed cycle, one cycle. Written as `vessel_digraph_fixation_<n>.json`.

---

## 42. The harness rows this adds

To `HARNESS_WAYS`, on the old and the new seeds:

- `plain ridges, joined, cyclic graph` (section 37), as curves over edge confidence;
- `cyclic graph, rendered, round 1`, `… round 2`, `… final`;
- `… flux as a test` (section 40's `z` in `group_tests`), beside section 33's `likelihood ratio` row;
- the λ sweep (section 39) on the real burst, by repeatability, loose ends, impossible junctions and directed cycles; the held-out bursts only score the λ chosen there.

Columns added: bridges right / wrong at cutoffs, lumens fused right, directions right (measured, inferred), flux `z` `told_apart`, flips right / wrong, directed cycles, whitened RMS on and off the graph. Flux columns are marked as upper bounds until section 35's velocity profile is in. Each section's one column is the one quoted; the rest is description, and with sixteen held-out bursts something among sixty cells will look like an effect by chance.

---

## The order of work

1. Section 37 on the real burst and the harness: the joined plain ridges with the six confidence terms, and the first principle's number. Most of the code exists; the per-edge repeatability is new.
2. Section 40 on `whole_graphs` as they are: balances from the geometric readings, fluxes from the whole-vessel speeds, `SIGMA_FLUX` re-measured; inferred directions, studentised residuals, imbalances, directed cycles and forced flips on the real burst; inferred directions scored on the harness against `propagate_directions`. No renderer needed.
3. Section 36's two checks on synthetic crossings and a thin daughter before any cell map is built.
4. Sections 35 (the generator) and 38 (renderer, whitening, core filter) on one synthetic burst, then the recursion of section 39 round by round, then the harness.

---

## How the reviews' findings were handled

| finding | handling |
|---|---|
| R1 A1, R2 4: residual empty along fitted edges; cores fitted after all | whole-network residual; windows and smoothing stop at the core radius; new vessels only off existing edges; collinear miss accepted and counted; core read by a matched filter and priced |
| R1 A2, R2 3: correlated noise; area division wrong for lines | whitening by the measured autocorrelation (off-graph and along edges), `n_e` per element from it; no single divisor |
| R1 A3, R2 10: no undo; merge exists; objective leaky | removal, merge, split as moves; merge proposed by the cell and decided by the reading, J a veto; `fuse_split_crossings` cited; centerline shift a priced move; J written out with nothing drawn outside it |
| R1 A4, R2 1: flux ratio favours looser readings; LOO test costs a solve per candidate; circular readings | leave-one-vertex-out candidate scores from one solve; flux as a test in `group_tests`; most specific that fits; `compare_readings` refuses unequal counts; the readings loop stated with a stop |
| R1 A5, R2 5: bridges reject gaps; rendered bridges break termination | never accepted, rejected or drawn; confidence from similarity on line mass and flow; rivals left to the network; recorded fields clean |
| R2 2: greedy sign step fails on a plexus; flips and cycles counted as p | lumen-wise sets, pattern rule, measurement-only sizing for signs; measured on grids; flips and cycles reported only |
| R2 6: `SIGMA_FLUX` unit; unverifiable | the notebook's unit in the prototype; re-measured and printed in section 40's cell |
| R2 7: Huber vacuous | robust sizing; Huber on balances; studentised residuals output |
| R2 8, C2: cycle enumeration exponential; directions of no confidence | SCC with a confidence cutoff |
| R2 9: blur in the wrong domain | transmittance blur in renderer and generator |
| R2 11, B7: `PARAMETER_COST` computed and swept; sweep blind to systematic additions | λ the swept multiplier; impossible junctions and directed cycles in the sweep |
| R2 12, B5: `repeated` per fixation; `static_scores` pre-interaction | per-edge repeatability named as new code; plaid cells read the pre-interaction score |
| R2 13, B8: junction recall bounded; no number for the principle | the consolidation's joining steps kept; `told_apart` ≥ 0.75 and the curve passing the consolidated point |
| R2 14: tests weaker than the text | leaf, two-wrong, monotonicity, plexus, vertex tests, wrong speed, forced flips tested |
| R2 15, C4: cost; inferred thick directions | stated; measured flag drawn |
| R2 16, B9: ring geometry; NMS | rings from 4 px; NMS per ring; thin-daughter check first |
| R2 17, C5: calibration against hand-set priors | prior explicit and separate; bottom-up terms calibrated first |
| R2 18, B1: flux row naming; plug flow | `flux as a test`; velocity profile in the generator; flux columns marked upper bounds until then |
| B2, B3, B4 | σ per fixation and line mass for thin vessels; per-lumen canvases composed before blur; halo of either sign bounded by the background's sigma, background fit on synthetic bursts |
| B6 | flips not counted as p |
| C3 (flow field unused) | open: the flow field's ridges as a second edge source and its orientation agreement as a confidence term are not in this version; listed for the next |
| C6, C7 | the order of work; one column per section |

## Open choices for you

1. The line mass for the flux: the kymograph's M (what the speed was read on) or the profile's `π κ R²`? The plan uses M and keeps the other as the cross-check.
2. Composition where there is no reading: sum (the plan's choice) or max.
3. The real burst's base rate of true terminations against gaps and dives, which sets the bridges' prior.
4. Whether the flow field (section 24) should be a second source of edges now or in the next version.
