# Plan: a clean cyclic graph from the plain ridges, rendered back and corrected by recursion

This plan continues the burst notebook (`notebooks/development copy 2.ipynb`, sections 1–34) with sections 35–42. It starts from your proposal (junction cells, a graph that throws nothing away, a renderer and a residual, recursion through bridging, through-junction merging, arm validation and lumen fusion, and a net-flux check) and turns it into steps that each end with a number. Where I would do something differently from the proposal, the section says so and why; the open choices are collected at the end.

What is already there, and the plan builds on rather than rebuilds:

- **The plain ridges**: `ridge_crests` → `centerlines` (section 29, `consolidate=False`), which is the "plain ridges" row of the harness. The attached picture is the consolidated version; this plan works from the plain one.
- **Arms read on a ring outside the core** (`read_arms`, section 26): angle, width, density, strength, and with flow the line mass, velocity, direction, pulse phase, blur trace. "Outside the junction" is already how arms are measured: the ring starts at half the thickest width plus `RING_MARGIN`.
- **A per-junction flux test** (`flux`, `group_tests`, section 27): per group of arms, the sum of velocity × line mass over their absolute sum, against `SIGMA_FLUX`. This is net flux = 0 at one junction under one hypothesis. Section 40 makes it a network.
- **Flow as a likelihood ratio between two readings** (section 33), the reliability p measured without truth, the whole-vessel speeds with widened sigmas (section 32), and the harness with held-out bursts (sections 30, 34).
- **vascnet's renderer** (`vascnet.synth._branch_od`): the cylinder chord profile `OD(d) = 2 κ sqrt(R² − d²)`, blurred, with transmittances multiplied between depth layers (OD adds at a crossing) and lumens united within a tree (max at a bifurcation). The forward model of section 38 is this function on our graph.

## Principles

- **Nothing is deleted; everything carries a confidence.** The graph keeps every ridge pixel's piece. Each stage states which confidence it works above, and that cutoff is swept in the harness, so that the numbers reported are curves over confidence, not one point. A consumer that ignores confidence will drown in stubs; that is the cost of the principle, and the plan pays it everywhere.
- **A proposal is accepted by the residual, with a price for its parameters.** Every change to the graph (a bridge, a merge, a fusion, a halo, a new vessel) is rendered and accepted when it lowers the weighted squared residual by more than its parameters cost. The objective only goes down, so the recursion ends.
- **Neural layers are readings of the fields that exist, not new fields.** The hypercolumns, scale competition and lateral interactions already give line cells, radius cells and orientation cells. The one genuinely new layer is the junction cell (section 36), because it runs everywhere and not only where the skeleton has a fork. End cells are junction cells with one arm; plaid cells are the two-orientation case of the hypercolumn.
- **Flow decides what a still cannot.** Direction, net flux and one-vessel-one-speed are the evidence a projection of a 3-D bed needs: a crossing and a confluence can look alike in a still.
- **Every section ends with a check on the harness.** The synthetic generator is extended first (section 35) so that cycles, gaps, shared lumens and shadows have a truth.

---

## 35. Synthetic bursts with the things this plan is about

The generator (section 28, `synthetic_network`) makes three trees that each split once. Nothing in it has a cycle, a gap, a confluence or a shadow, so nothing below could be scored. Extend it before anything else, keeping the old bursts as they are (the seeds of sections 30 and 34 must still give the same bursts):

1. **Loops.** A daughter may rejoin another branch of the same tree (an anastomosis) with probability `SYNTH_LOOP`. Its speed is then set by flux conservation at both ends (the generator already conserves flux at a split). A loop is a true cycle in the 2-D graph that also exists in 3-D.
2. **Crossings at several depths.** Already there (two trees at two depths cross). Add a region with `SYNTH_PLEXUS` thin vessels at three or more depths, so that four-arm and five-arm junctions, and cycles that exist only in the projection, have a truth.
3. **Gaps.** With probability `SYNTH_GAP` a segment fades over `SYNTH_GAP_LENGTH` px (its density goes to a share `SYNTH_GAP_DEPTH` of itself, as when a vessel dives through the focal plane). The truth records the gap, so a bridge can be right or wrong.
4. **Terminations inside the field.** A few capillary ends stop inside the field. A free end is then not always a gap, and the bridging of section 39 has something to get wrong.
5. **Confluences.** A tree whose flow runs from the leaves to the root is already a confluence; make sure both directions appear equally (they do: `rng.choice([-1, 1])`).
6. **Shadows.** Add a halo to thick vessels: a wide, shallow density `SYNTH_HALO_DEPTH × OD_peak` with sigma `SYNTH_HALO_WIDTH × R`, as the light scattered around a big vessel appears in the real burst. Section 38 fits a halo, and this is its truth.
7. **Shared lumens.** Where two segments of one tree overlap (a tight bifurcation seen almost end on) their lumens unite (max); where two trees overlap, densities add. Both are already the generator's physics; record per junction which it is, so that the fusion of section 39 is scored.

**Check:** the old seeds reproduce sections 30 and 34 to the digit. The new seeds (`HARNESS_SEEDS_2`, `HELD_OUT_SEEDS_2`) are listed with how many loops, gaps, terminations and plexus junctions each has.

---

## 36. Junction cells

A junction cell at a pixel, for a ring radius ρ, counts the arms of a junction there and fires in proportion to how clearly it sees them.

**Input.** The static score of section 20 (`static_scores[n]`, 16 orientation layers) and `half_width[n]`.

**Mechanism** (`junction_cells(scores, half_width, rings)`):

- On a ring of radius ρ around the pixel, at angles φ in steps of 360/`RING_STEPS` degrees, read the score of the neuron whose orientation is the radial direction φ (interpolated between the two nearest of the 16). This is "line energy pointing at the centre", one number per angle. The scale competition has already chosen a width per pixel, so arms of different widths are read at their own width without a separate kernel per size.
- The angular profile's peaks above `T_LOW` (joined to a peak above `T_HIGH`, as the ridges are) are the arms. Two peaks closer than `SAME_ARM` degrees are one arm (a vessel traced twice).
- `n` = the number of peaks. `rate` = the geometric mean of the peak scores in noise units, times a core term: the centre's density against the arms' (a junction's core is at least as dark as its arms; two vessels crossing are darker, section 27's overlap test). The rate is the cell's confidence; it is not thresholded.
- Rings: ρ in `JUNCTION_RINGS` = (6, 10, 16, 24) px. The ring that matches a junction is the one whose radius is about the thickest arm's half-width plus `RING_MARGIN` (the same place `read_arms` reads). A cell reports `n` and `rate` per ring; the map keeps, per pixel, the ring with the largest rate and the `n` of that ring.
- `n = 1` is an end cell. `n = 2` with the two arms within `STRAIGHT` degrees of a line is a line cell, and is suppressed (it fires everywhere along a vessel). `n = 2` with a turn is a bend. `n = 3` and `n = 4` are candidates for a bifurcation and a crossing; the type is still read by the partitions of section 27, the cell only says where and how many.
- **Plaid cells**: pixels where the hypercolumn fires at two orientations at least `PLAID_APART` = 45 degrees apart, both above `T_HIGH`, are crossings seen from the inside. They are kept as a separate map because they fire at the core, where the ring cell's core term is weakest.
- Non-maximum suppression within a radius ρ/2 on `rate`, so that one junction is one cell.

**Where it differs from the proposal.** You asked for kernels of several sizes that handle arms of several sizes. The ring radius is the kernel size; the arm size is handled by the scale competition that already happened in the hypercolumn. A separate kernel per (ring, arm width) would be 4 × 8 maps per n, and nothing in the check below would show it earning that.

**Result:** `junction_maps[n]`: `n`, `rate`, `ring`, and `plaid`, at the frame's size.

**Check** (synthetic): precision and recall of junction cells against the true junctions, as a curve over `rate`, next to the forks of the plain ridges (which are what section 29 uses as junctions). The claim to test: cells find crossings that the skeleton broke into two T's, and bifurcations hidden in a blob, at the same false-alarm rate. Also the share of true n matched (3 read as 3, 4 as 4) per ring.

---

## 37. The clean cyclic graph

**Input.** The plain ridges (`ridge_crests` → `centerlines`, no consolidation; `least = 0` in `ridge_lines`, `to_forks = True`), `junction_maps`, the repeatability across fixations (`repeated`), `flow_masks`.

**Vertices.** Every fork and every free end of the skeleton; then:

1. Forks within one junction core are one vertex: `contract_cores` (section 26) with the core radius from the thickest arm, as now. A junction cell within the core adds its `rate` to the vertex; a junction cell with no fork within its ring becomes a vertex of its own with no edges yet (section 39 looks for its arms).
2. A free end is a vertex with degree one. It carries `at_data_edge` (within `EDGE_MARGIN` of where the data ends; not a free end in any later sense).

**Edges.** Every piece of skeleton between two vertices, as a spline (`spline_vessel`: knots `KNOT_SPACING` apart, radius with knots `RADIUS_KNOTS` times farther, points spaced along the spline). Then:

3. **Twins.** Two edges that leave a vertex side by side within `SAME_ARM` degrees and closer than a width are one vessel traced twice (as `read_arms` already folds them). The shorter is kept as an edge with confidence lowered by the twin test, not removed, and it is not an arm.
4. **Zero-length edges** (a piece of one or two pixels between two forks of one core) disappear into the vertex when the cores are contracted; they are the only thing that is not kept, and they carry no data.
5. **Multi-edges and self-loops are allowed.** Two edges between the same two vertices are a plexus; an edge from a vertex to itself is a loop. Nothing merges them.

**Confidence per edge**, bottom-up, each a log-likelihood ratio added up and squashed to (0, 1) at the end:

- *Length in field lengths*: its length over `ELONGATION` × its half-width (the `STUB_FIELDS` argument of section 31: a response no longer than the field that gave it says nothing). Zero at one field length, rising to full by three.
- *Strength*: median score along it in noise units, against `T_HIGH`.
- *Repeatability*: the share of it found again in the other fixations above chance (`repeated`, section 29), where the other fixations cover it.
- *Flow data*: the share of its points with flow data (`flow_share`), where the question is whether blood moves there.
- *Twin* and *stub*: lowered for a twin, and for a free-ended piece that lies inside a parent's tube (`tube_reach`, section 31). Section 31 did not adopt the tube rule for dropping pieces; as a confidence it costs nothing.
- A seventh term comes from section 38: how much the residual rises when the edge is left out of the rendering.

**Confidence per vertex**: the junction cell's `rate` (0 where no cell fires), later multiplied by the reading's confidence (section 27) and by the flux balance (section 40).

**What "clean" means, as numbers**: no two vertices within one core; no edge shorter than two pixels; every edge a spline with a radius profile; every edge in exactly one place in the adjacency; the graph's cycle count (edges − vertices + components) reported with the share of cycles whose circumference is under `RING_LEAST` px (these are almost always one junction read as two).

**Result:** `cyclic_graphs[n]`: `vertices` (x, y, kind: fork / end / cell, rate, confidence), `edges` (spline, radius, strength, density, the two vertices, confidence and its terms), `cycles` (a cycle basis).

**Check** (synthetic): centerline precision and recall as a curve over edge confidence, next to the single points of the plain and consolidated rows of section 34. The claim: the curve passes above the consolidated point (same recall at higher precision, or the other way round), so that keeping everything with a confidence loses nothing against dropping by length. Also the number of cycles found against the true number of loops plus projection cycles.

---

## 38. The renderer and the residual

**Per-edge measurements first** (`edge_profiles`): at windows of `KNOT_SPACING` px along every edge, the density profile across the edge over ±`PROFILE_REACH` half-widths (`profile_across`, section 21), fitted with the cylinder chord profile convolved with a Gaussian: parameters R (radius), κ (density per pixel of chord: the darkness), σ (the blur: the sharpness), and an offset c (the halo level under it). Smoothed along the edge with the radius spline's knots. From the fit, per window: `R_od`, `kappa`, `blur`, and from the fields: `R_ridge` = half-width / `CYLINDER`, orientation, and from section 32: velocity with sigma, direction, line mass M. Flux per edge: `q = M × v` (as `flux` does per arm), with σ_q from σ_v and the spread of M.

Two radii, on purpose: `R_ridge` from the scale competition and `R_od` from the profile. Where they disagree by more than `SIGMA_WIDTH` in log, the profile is not one cylinder: an overlap, a halo or a twin. That disagreement is a feature of the residual analysis, not an error to average away.

**The renderer** (`render_graph(graph, shape)`): every edge drawn as its cylinder's density along its spline (`_branch_od` with our R(s) and κ(s)), blurred by its σ; a halo term `h × Gaussian(σ_h)` per edge where fitted (section 39, step 0). Composition at every pixel:

- edges that the junction readings say are one vessel (a group of two or three in a vertex's partition, joined through `join_pieces`): **max** (one lumen);
- otherwise: **sum** (Beer–Lambert: densities add where two vessels lie over each other).

Edges are drawn into their own small canvases, so that a proposal that touches two edges re-renders two canvases and the residual is updated locally. That is what makes several rounds of section 39 affordable.

**The residual**: `residual = density_mean − rendered`, in units of the local noise (section 19's contrast layer gives it), on valid pixels. Three readings of it:

1. **Along every edge**, sampled on the centerline and on two lines at ±2R: a profile `residual_along(edge)`. Positive = darkness the model does not explain; negative = darkness the model drew that is not there.
2. **At every vertex**, within its core: the sign and size of the residual at the core says whether the composition was right. Positive core under a max composition: the arms were two vessels after all (they add). Negative core under a sum: one lumen (they do not add).
3. **Off the graph**: the residual's own ridges (the hypercolumns of section 20 run on the residual) are candidate vessels nobody has yet.

The interpretation table you gave becomes the residual profile's classifier:

| residual along an edge | reading |
|---|---|
| positive along the whole edge | the edge is underdrawn: κ or R too small (refit), or a second vessel runs along it (twin, or a deeper vessel) |
| negative along the whole edge | overdrawn: a stub drawn as a vessel, or a halo counted as a lumen |
| good in the middle, off at both ends | the vertices are mis-modelled: wrong composition, wrong core radius, or a missing arm |
| off at one end only | that end: a junction misread, or a free end that is a gap |
| a bump in the middle | a crossing vessel is missing there: look for a ridge through the bump in the residual's field |

**Shadows and big vessels.** Thick vessels in the real burst carry a broad halo. It is fitted first, biggest vessels first, as the offset c and the halo (h, σ_h) of their profiles, before anything thin is judged: a thin vessel inside a big one's halo is otherwise read as overdrawn.

**Result:** `rendered[n]`, `residual[n]`, `edge_profiles[n]`, and per edge the seventh confidence term: `Δχ²_without` = how much the weighted squared residual rises when the edge is not drawn (an edge that explains nothing has it near zero; it stays in the graph with its confidence down).

**Check** (synthetic): the residual's RMS in noise units on the graph against off it (should be near 1 where the model is right); `R_od` and κ against the true radius and absorption; the classifier's rows against the truth (how often "bump in the middle" marks a true missing crossing).

---

## 39. The recursion

One round applies proposals of six kinds, each local, each rendered, each accepted when

`Δχ² − PARAMETER_COST × (parameters added) > 0`,

with `PARAMETER_COST` = ln(number of valid pixels the proposal touches) (the BIC price; it is the one setting of the recursion, and section 42 sweeps it on the real burst's repeatability, not on the harness). The objective only falls, so the rounds end: stop when a round accepts nothing, or after `ROUNDS_MOST` = 5. The order within a round is from what explains most density to what explains least.

**Step 0, halos.** Biggest vessels first (by R × length): fit (h, σ_h); accept by Δχ².

**Step 1, bridging free ends.** For every pair of free ends not at the data edge, within `BRIDGE_REACH` = 3 × the larger width (and at least `RIDGE_REACH` px): the connecting line must leave each end within `SIGMA_BRANCH` of the end's own direction (both ends point at each other); widths within `SIGMA_WIDTH`; densities within `SIGMA_DENSITY`; where both have a direction of flow, the directions agree along the bridge (one in, one out); and the density along the line has no dip deeper than `SIGMA_DENSITY` below the two ends' (a dip is two vessels ending, not one vessel fading). The bridge is an edge with `bridge = True`, its radius and κ interpolated, drawn and accepted by Δχ²: a bridge over a true gap finds unexplained darkness under it, a bridge over background does not. Its confidence is the sum of these terms, so a bridge that is accepted by the residual but turns the flow against itself keeps a low confidence rather than being refused. Rivals (an end with two candidates) are read like a junction's partitions: the best pairing against the second best. Free ends that section 40 says carry flux rank first: flux that stops inside the field has to go somewhere.

**Step 2, merging arms through a junction.** At every vertex, the partition reading of section 27 says which arms are one vessel. Add to its tests: the straight connection between the two arms, across the core, must not overlie a density dip against the arms' distal densities (as above); and the arms are compared by their distal profiles (`edge_profiles` outside the core: R_od, κ, blur), not by anything inside it. The arms that pass are joined (`join_pieces`) and composed by max; the rendering is accepted by Δχ²; the vertex's confidence is updated by the reading's. Where a reading is accepted, the whole vessel is re-read for its speed (section 32), so that the next round's flux is the whole vessel's.

**Step 3, validating arms, top-down and bottom-up.** At every vertex compare the junction cell's `n` (at the ring that matches the vertex) with the graph's degree there:

- degree > n: an arm the cell does not see. It is rendered without; if Δχ² barely moves it is a stub or a twin: confidence down, it stays.
- degree < n: an arm the cell sees that the skeleton has not. Look in the residual's field along the cell's unexplained peak direction for a ridge within `RIDGE_REACH`; string it (section 22's stringing on the residual's crests) and add it as an edge, accepted by Δχ². This is the cell telling the orientation cells where to look (inside → out), and the orientation cells answering (bottom-up).
- Centerlines shift: outside every core, each edge's spline is refitted to the crest of the data profile where the rendered profile's crest is off by more than half a pixel, damped by half per round. Inside the cores nothing moves; the core is the reading's business.

**Step 4, fusing lumens.** At every vertex whose composition is in doubt (the core residual has the wrong sign for its composition by more than `Z_PASS`): render the other way (sum ↔ max) and keep the better. Where a halo makes the density test weak (a thin vessel inside a thick one's halo), the decision is left to the radius continuity (one lumen has one R_od across the core; two overlapping vessels have two) and to flow (one speed and direction through the core, or two). The reading's confidence takes the Δχ² of the two renderings as one more likelihood ratio.

**Step 5, missing vessels.** The residual's own field (section 20's hypercolumns on the residual) is strung into chains (section 22); every chain longer than one field length and above `T_HIGH` is proposed as an edge, joined to the graph where it meets it, accepted by Δχ². This is where a vessel that the static score never saw (a faint one under a thick one's halo, or one the consolidation would have dropped) comes in, with the lowest confidence of all, and is then subject to every step above in the next round.

**Step 6, the flux network** (section 40), run once per round, which sends its findings back up: inconsistent fluxes lower the confidence of the measurements, persistent imbalances at a vertex flag a missing arm (back to step 3) or a misread (back to step 2 or 4), and flux at a free end ranks the bridges (step 1).

**Result** per round: `rounds[n][r]` with what was proposed, what was accepted, Δχ² total, and the graph after it. The last is `graph_final[n]`.

**Check** (synthetic, held out): after each round, centerline and junction precision and recall as curves over confidence, type right, bridges right (a bridge over a true gap) against bridges wrong (over a termination or background), fused lumens right, and the residual's RMS. The claim: each round improves the curves or leaves them, and the improvements come mostly in round 1 and 2.

---

## 40. Net flux = 0: a flux network

This is the stage you were unsure how to implement. The per-junction version exists (`group_tests`'s `flux` test: the signed sum of velocity × line mass over the arms of one group, against `SIGMA_FLUX` = 0.45 of the absolute sum). The network version is a weighted least-squares problem on the whole graph, and it is what gives the recursion something to send back up.

**Unknowns.** One flux `q_e` per edge, signed along the edge's points. Measured on edges with a clear speed: `q̂_e = M_e × v_e` with sigma `σ_e` from the widened velocity sigma and the spread of M along the edge (`edge_profiles`). Free on the rest.

**Constraints.** One per lumen at every vertex. A vertex's reading partitions its arms into groups; each group is one lumen, and what comes into it leaves it:

`Σ_{e in group} s_{v,e} q_e = 0`,

with `s_{v,e}` = +1 where the edge's points run out of the vertex and −1 where they run in (that is `velocity_out`'s sign). A bifurcation (one group of three) gives one equation, a crossing (two groups of two) two, a through-vertex one, a vertex with no reading none. Free ends give no constraint, and a vertex at the data edge none.

**Solve.** Minimise `Σ_measured (q_e − q̂_e)² / σ_e²` subject to `A q = 0`: a linear KKT system, or in penalty form `‖W(q − q̂)‖² + (1/σ_flux²) ‖A q‖²` with `σ_flux` the share of the mean absolute flux at the vertex that a balance may miss (`SIGMA_FLUX` is this number; the proposal's "soft" is right, because OD × velocity is not red-cell flow exactly: the hematocrit differs between daughters, and the line mass is a mean along the vessel). Weights are made robust by iteratively reweighting (a measurement far from the solution gets a lower weight), so that one wrong speed does not spoil a whole tree.

**What comes out.**

1. **A flux on every edge**, measured or inferred, with its sigma from the solve. Inferred directions replace `propagate_directions` (which passes a direction from one junction to the next without weights). Every edge now has a direction with a confidence `|q_e| / σ_q`, and the graph is a **cyclic digraph**.
2. **An imbalance per constraint**, `r_v = (A q̂)_v / σ`, in sigmas: the net flux at a lumen, measured. It multiplies the vertex's confidence through `exp(−r_v² / 2)`.
3. **A likelihood ratio between readings**: solve under the best partition and under the second best; the difference in χ² is the flux's evidence between the two, added to the directions' likelihood ratio of section 33 in `read_junction`. A crossing whose two through-vessels each balance, against a bifurcation that does not, is told apart here even when the directions fit both.
4. **Directed cycles.** Blood does not go round a loop. A directed cycle in the digraph means a wrong direction on it or a crossing read as one lumen; it is reported per cycle and counted as a third way of measuring p without truth (next to the three-in junctions and the sides of section 33).
5. **Flux lost at free ends**: at every free end inside the data, the flux that arrives there. A large one says the vessel continues: it ranks the bridging candidates of step 1, and where no bridge is possible it is a vessel diving out of focus, which is a kind of vertex (`dive`) worth marking.
6. **Persistent imbalance** at a vertex under every reading: a missing arm (the flux has somewhere to go that the graph does not have), a missing crossing vessel, or a wrong speed. It is sent back to step 3 with the sign of the missing flux.

**Mass as a cross-check.** The line mass from the kymograph (`M`, the mean density along the vessel times its width, section 23) should match `κ π R²`-shaped line mass from the profile fit. Two vessels overlapping double M against what one R² predicts, and that is one more overlap test that does not need a junction.

**Result:** `flux_networks[n]`: per edge `q`, `σ_q`, `direction`, `measured`; per vertex `imbalance`, per cycle `directed`, per free end `flux lost`.

**Check** (synthetic): inferred directions right (curve over `|q|/σ_q`), against `propagate_directions`; imbalance at true junctions near zero, at false junctions not; the flux likelihood ratio's `told_apart` for type right against wrong, next to the directions' (section 34's table gets a column); directed cycles found against the count that the true directions give (zero), with the p they imply.

---

## 41. Confidences: what they mean, and the digraph that is handed on

Every confidence above is a sum of log-likelihood terms or a product of probabilities. Before they are used by anyone, each is **calibrated** on the development bursts: binned, and the share right in each bin plotted against the bin's confidence (a reliability diagram) with `told_apart` as the one number. A confidence of 0.8 should be right about 80% of the time; where it is not, the terms' sigmas are what to adjust, and the held-out bursts say whether the adjustment held.

The graph handed to whatever comes next: `vessel_digraphs[n]` with, per edge, the spline, R (both), κ, blur, velocity ± sigma, direction (measured or inferred, with its confidence), flux ± sigma, the edge confidence and its terms, and `bridge` / `twin` / `stub` / `found in residual` flags; per vertex, position, kind, type (bifurcation, crossing, confluence, through, bend, end, dive), the partition, confidence by geometry, with flow, with flux, and the imbalance; per cycle, whether it is directed. Written as `vessel_digraph_fixation_<n>.json` next to the whole-vessel files.

---

## 42. The harness rows this adds

To `HARNESS_WAYS`, each on the old and the new seeds:

- `plain ridges, cyclic graph` (section 37, before any recursion), scored as curves over edge confidence;
- `cyclic graph, rendered, round 1`, `… round 2`, `… final`;
- `… likelihood ratio with flux` (section 40's ratio added to section 33's);
- the `PARAMETER_COST` sweep, on the real burst's repeatability and loose ends (as section 31 swept `STUB_FIELDS`), with the held-out bursts only scoring the setting chosen there.

Columns added to the ablation table: bridges right / wrong, lumens fused right, directions right (measured, inferred), flux imbalance at true junctions, directed cycles, and the residual's RMS on and off the graph. The numbers to quote stay the held-out ones.

---

## What I changed from the proposal, and why

- **One junction-cell layer, not a kernel per arm size.** The scale competition already picks a width per pixel; the ring radius is the only size the cell needs (section 36). If the check shows crossings between a thick and a thin vessel are missed, a second reading of the ring at the thin width is the fix, not a bank.
- **The renderer composes by the junction reading, not by a global rule.** Max within a lumen, sum across lumens; the reading says which (section 38). The "OD adds at overlap, not at a shared lumen" test is then per pixel in the residual, and the per-junction overlap test of section 27 becomes a special case of it.
- **Accept by Δχ² with a price per parameter**, rather than "keep if the residual improved". Without a price, every bridge and every stub improves the residual a little, and the recursion grows the graph until it draws the noise. The price is one setting, swept on the real burst (section 39).
- **Net flux as a network solve** rather than a test per junction (section 40). The per-junction test exists; the network is what infers the unmeasured directions with weights, finds the wrong speed among right ones, and reports directed cycles, which a junction cannot see.
- **The recursion's order**: halos first, then readings, then bridges, then arms, then fusion, then the residual's new vessels, then flux, each round. You proposed bridges first; a bridge judged before the big vessels' halos are drawn is judged against the wrong background.
- **Nothing is deleted, but every stage names its working cutoff**, and the harness reports curves over confidence. "Only assign confidence" is right; it is also unscorable unless every number becomes a curve.

## Open choices for you

1. **Line mass for the flux**: the kymograph's M (mean density × width along the vessel, section 23) or the profile's `κ π R²`? I would use M for the solve and `κ π R²` as the cross-check, because M is what the speed was read on. The other way round is defensible.
2. **Composition when there is no reading** (a vertex with five arms, or one left unresolved): sum or max? I would sum (two vessels is the safer assumption for the residual, since underdrawing is what the residual finds most easily) and let step 4 flip it.
3. **Terminations.** The real burst's free ends inside the data: are true terminations expected (capillary ends), or is nearly every free end a dive or a gap? That decides the prior on bridging and whether `dive` vertices are worth their own kind.
4. **How many rounds the real burst should get** before the harness says it is enough: five is a guess; the per-round check of section 39 will show where it flattens.
5. **The cutoff for the digraph handed on.** A graph with everything at confidence > 0 is right by principle and unusable by a consumer that does not weight. Which consumer comes next decides what cutoff is stated with it.
