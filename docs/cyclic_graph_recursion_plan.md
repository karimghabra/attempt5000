# Plan: a clean cyclic graph from the plain ridges, rendered back and corrected by recursion

Version 7. This plan continues the burst notebook (`notebooks/development copy 2.ipynb`, sections 1–34) with sections 35–42. It has been through five adversarial reviews (`cyclic_graph_recursion_plan_review.md` to `_review_5.md`); the last part of this document says how each finding was handled, and the fifth review's verdict is that sections 35, 36, 37, the moves of 39 and section 40 on trees have converged, with what remains being for the harness to measure. Version 6 made the three design decisions the fifth review left open (the structural confidences, the plexus claims, the residual column) and closed its two code holes. **Version 7 moves the residual for topology out of the optical density and into the orientation layer** (the render is run through section 20's hypercolumns and compared with the data's score, channel by channel), makes the set of splines the model and the graph a derived view of it (`spline_to_graph`), and keeps the optical density only for what the orientation layer cannot see: composition at cores and line mass. Section 40 was prototyped (`vascnet/fluxnet.py`, `tests/test_fluxnet.py`) and is described as it behaves, with the numbers the tests and the reviews measured.

**What this plan claims, plainly.**

- **Sections 37 and 41** (the graph with confidences, calibrated) should raise the harness's centerline and junction columns over the consolidated row, because they keep what the consolidation dropped and say how much each piece is worth. This is the cheapest section and the first to run; its one number (the first principle's) decides whether the rest is built on it.
- **Section 40** (the flux network) should raise the direction columns on tree-like beds: it infers the unmeasured directions along whole vessels, turns lone wrong directions round where a lumen's arms are all measured or pinned by other lumens, which on the real pipeline means bifurcations and confluences (+3 to +7 points on binary trees with the thick level unmeasured, measured), and reports what it cannot settle. **On a plexus it does nothing, and the plan claims nothing there.** `join_pieces` joins every vessel through every crossing, so a vessel of a projected plexus is one measurement end to end and every balance pairs arms of one measurement: no sign is constrained, and a vessel with no measured piece is in the null space and gets no direction either (measured). A true anastomosis is read as a crossing by `geometric_reading`, and the one-lumen reading of four arms has no tests and is never a rival, so the network cannot set the wrong structure aside; the only thing that can read an anastomosis is section 38's core filter, which is decisive for vessels of R ≥ 3 and marginal for capillaries. The plexus column of section 42 is therefore expected to be zero until section 27 can read a thick anastomosis as one lumen.
- **Sections 38 and 39** (renderer, residual, recursion) should raise centerline and junction recall through the moves that add, remove, extend and bridge splines, because their evidence is now the difference between the data's orientation score and the render's, channel by channel, which ignores what wrecked the optical-density residual (halos, the background fit, the blur's domain, the modulation of clusters along a vessel, κ's freedom) and has a number computable on the real burst without truth: the agreement between the render's ridges and the data's. They should raise found-and-typed-right where the core's matched filter has power, which stays in the optical density: for R ≥ 3 at every blur the generator makes (z 9 to 135, derived by the fifth review), and for capillaries only in focus. The recursion's moves are edits to splines; the flux network feeds the readings once, at the end, with a second solve after it and no loop.
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

**The model is the set of splines; the graph is a view of it** (`spline_to_graph`, new code). Each spline carries its place and half-width splines (`spline_vessel`), κ or line mass, its confidence and terms, its provenance and flags. The graph is derived: a vertex wherever splines end, meet or cross, an edge for every piece of spline between two vertices, and the lumens from the readings. The rules:

- an endpoint within `RIDGE_JOIN` of another spline's endpoint is a join; within `RIDGE_JOIN` of another spline's body, a T (the body is split there);
- two bodies that intersect are a crossing vertex at the intersection; two bodies that run within the sum of their half-widths for longer than one field length without intersecting are a twin or an overlap, marked, not a vertex; two bodies that meet at under `SAME_ARM` degrees are a join candidate, not a crossing;
- vertices within one core radius are one vertex (`contract_cores`); a spline that intersects itself makes a loop vertex; multi-edges and self-loops are allowed; a free end carries `at_data_edge`.

The initial spline set is the one section 29 fits to the ridges above; every structural move of section 39 is an edit to the spline set (add a spline, remove one from the rendering, extend an end, join two end to end, split one at a point, move a centerline), and the graph is recomputed where the edit reaches. Intersections are found by bounding boxes and a Newton step on the two splines; a twin and a crossing at a glancing angle are told apart by the angle and the length of contact.

**Vertices and edges** then as in version 4 (cores contracted, cells added, twins kept and marked, multi-edges and loops allowed).

**Initialisation: how the first splines are placed, and when the first network is done.**

1. *Ridges.* `ridge_crests` on the static strength after the interactions: a pixel is on a ridge where its strength is the largest across its own orientation, above `T_LOW` and joined to a pixel above `T_HIGH`. The crest neurons of section 21 carry their place to a fraction of a pixel; the skeleton's pixels are integers. Section 29 fits to the skeleton's staircase; the initialisation carries each ridge pixel's sub-pixel offset across the line (new: `ridge_lines` keeps it), which halves the fit residual.
2. *Pieces.* Closing within `RIDGE_JOIN`, the skeleton with its `SPUR_LENGTH` pruning, `extend_ends`, then `ridge_lines` with `least = 0` and `to_forks = True`: every piece runs fork to fork or fork to free end with the fork pixel at each end. Section 29 leaves pieces under `SPLINE_LEAST` = 5 px out; here they are kept as straight segments, since `fit_spline` fits a line to two points already.
3. *Splines.* Each piece through `spline_vessel`: a least-squares cubic B-spline for the place with knots `KNOT_SPACING` = 12 px apart, one for the half-width with knots three times farther, points re-sampled a pixel apart along the spline, and the fit residual (the rms distance of the ridge points from the spline) recorded. *Knot refinement* (new): where a piece's residual exceeds the ridge's own precision (`RIDGE_PRECISION`, about 0.5 px on the staircase, half that with the sub-pixel offsets), the knot spacing on that piece is halved and the fit repeated while the residual's fall pays for the knots at section 39's price; a piece that still bends faster than a 6 px spacing is split at its point of largest residual, since a corner is usually a junction the fork detection missed. A spline is done when no further knot lowers the priced residual.
4. *Darkness.* κ (or the line mass) per window outside the cores from the chord-profile fit of section 38, the thickest vessels first and the thinner ones on the density with the thicker ones' render subtracted, repeated until no κ changes by more than its own error: a Gauss–Seidel pass over the splines, two or three times.
5. *The graph.* `spline_to_graph`; every vertex moved to where its arms' axes meet (`where_axes_meet`, read on the ring); cores contracted; the junction cells' rates added; readings by geometry.

The first network is done when three things hold, none of which needs truth:

- every spline's priced residual is at the ridge's precision, every vertex is at its arms' meeting point, and no two vertices lie within one core;
- κ has converged;
- **self-consistency along the model's own support**: the splines are rendered, scored and their ridges extracted, and within the tubes of the existing splines (the radius plus `TUBE_BLURS` sigmas of the blur) the render's ridges agree with the data's, within `REPEAT_WITHIN`, at least as well as the data's ridges agree with the other fixation's (`repeated`'s share above chance, section 29). That is the most the render can be asked to match, since the data does not agree with itself better. A disagreement inside a tube is a measurement problem (a width, a κ, the blur, a spline placed off the crest) and is fixed here; a disagreement outside the tubes, a ridge the data has and the model lacks or the reverse, is the recursion's input, not the initialisation's failure.

Then the self-consistency number over the whole frame, with no restriction to the tubes, is recorded as round 0, and the recursion begins. "Done" does not mean every ridge pixel has a spline above a cutoff (stubs and short pieces are splines with low confidence), and it does not mean the topology residual off the splines is empty; that residual is what the rounds are for.

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

**Two residuals, for two questions.**

**(a) Topology, in the orientation layer.** The render is run through section 20's hypercolumns, with the data's noise normalisation (the render has no noise; its answers are put in the data's noise units with the same constants, never re-measured on the render), and compared with the data's answers **channel by channel, at the stage before the scale competition**: 16 orientations × 8 widths. The topology residual is `score(data) − score(render)` per channel. Its ridges are the moves' evidence: a positive ridge is a line the data has and the render lacks (add a spline, extend an end, bridge a gap where it continues a spline), a negative ridge a line the render has and the data lacks (remove from the rendering). The comparison is per channel and before the competition because that is where a thin vessel at a thick one's flank is still visible: after the competition the thick vessel wins the pixel in the data and in the render alike, and the thin one vanishes from the difference. The junction cells of section 36, read on the render and on the data, give the junction residual in the same way (an `n` the data has and the render lacks).

What this removes, against the optical-density residual the earlier versions used for topology: the halos and the background fit (the filters are band-pass and ignore both), the blur's domain (the render goes through the camera's blur and then the same filter as the data), the cluster modulation along a vessel (slow, and ignored by a line filter), and κ's freedom (a missed vessel answers in its own channel whatever κ its neighbours were given). What it costs: a hypercolumn run per evaluation (FFT-based over the frame in section 20; a move re-scores a window of its own extent plus one footprint, with the same padding), and a residual correlated over the filter's footprint (`ELONGATION` × the width along the line, the width across), so the per-move price counts independent samples per channel as pixels over the footprint, which is known from the filter and needs no measurement.

**The self-consistency number.** The full pipeline (competition, interactions, `ridge_crests`) run on the render gives the render's ridges; their precision and recall against the data's ridges (the harness's centerline metric, through `found_again`) is a number on the real burst that needs no truth. A graph is a fixed point of extract → fit → render → score → extract when it is 1; at round 0 it says how far section 29's splines are from one, and it is section 39's column on the real burst.

**(b) Composition and mass, in the optical density.** What the orientation layer cannot see: one lumen against two at a core (a darkness difference at the same orientation and width), the line mass, κ. These stay in the optical density, at cores and along edges only, never for topology: the core's matched filter below, κ per window for the flux's line mass, and the per-edge residual along the spline in the noise's along-line unit as description. The noise for these is the quiet pixels' two-dimensional autocorrelation and spectrum off the graph, as before, since the core z rides on the white floor.

**The core's matched filter.** The template is the difference between the two compositions rendered at the core from the arms' κ and R; its whitened inner product with the residual over its norm is a z for two vessels against one lumen. What it adds to section 27's `overlap` test: every pixel of the core instead of a 5 × 5 maximum, the right spatial template, and the noise's correlation. **Its reach, derived for the generator's noise** (the fifth review): two vessels crossing at 90 degrees, R = 1.5 each, give z = 18.6, 2.4 and 0.8 at blurs of 0.8, 2 and 3.5 px; R = 3 each give 44, 17.5 and 9.2; R = 5 and 3 give 135. Decisive for R ≥ 3 at every blur the generator makes, and for capillaries only in focus. The z rides on the white floor of the noise, so on the real burst the quiet pixels' spectrum is what sets it. A **check before section 39**, after section 35 has capillaries: the z by radius and blur on rendered crossings against recorded one-lumen junctions. At a blurred capillary four-way geometry keeps deciding, and the plan does not claim the filter there.

**Where the residual is read, the table, and the rule for new vessels** as in version 4 (cores, off the graph, along edges for what the fit cannot absorb; evidence off existing edges adds a vessel; the collinear miss counted).

**Column:** `told_apart` of the core z between one-lumen and two-vessel junctions, by radius and blur; the self-consistency number (ridge precision and recall of the render against the data) at round 0, and the per-edge RMS by radius, as description.

---

## 39. The recursion

Moves under J, λ chosen on the real burst by the rule of the principles (the largest λ at which impossible junctions and directed cycles stay at or below the consolidated graph's shares; repeatability and loose ends reported beside); at most `ROUNDS_MOST` = 5 rounds; the rounds end because J falls at every taken move.

Every move is an edit to the spline set, the graph recomputed where it reaches (`spline_to_graph`), and the evidence for a structural move is the topology residual of section 38(a): `J = χ²_topo + λ Σ_splines k ln(n)`, the channel residual squared over its footprint-counted samples, plus the price. Halos are no longer fitted: the orientation layer does not see them, and they return only in section 38(b)'s core test, where the background around a core is the quiet pixels' level.

**Rounds 1 to 4: structure, with the readings from geometry and the core filter only.** Step 1 bridges (never accepted, rejected or drawn; confidence from similarity; rivals unresolved; a positive ridge of the topology residual that continues a spline across the gap is recorded beside the confidence); step 2 merging arms through a junction by the partition reading with the core filter among its tests; step 3 arms against the cell's `n` on the data and on the render (an arm the data's cell sees and the render's does not is a positive ridge to string), centerline shifts as zero-parameter moves judged by the topology residual; step 3b removal from the rendering (a negative ridge), merge, split (proposed by the cell, decided by the reading, J a veto on which the reading reverts); step 4 composition, by the core z of section 38(b); step 5 missing vessels: the positive ridges of the topology residual off the existing splines, strung and fitted, each a move.

**Step 6, the flux network, in the last round: two solves, no loop.** (a) The readings as they stand; (b) one solve on the whole graph, with a measurement id per whole vessel, structural confidences per join, bridges as doubtful edges in their end's unit; (c) every vertex's candidate readings scored by the leave-one-vertex-out tests; (d) the readings taken once more, the flux test able to fail a reading and never to choose among readings that fit; (e) **a second solve under the readings of (d)**, with a vessel that (d) split carried as separate measurements with the old direction, and **no re-reading after it**: a sequence of readings and solves oscillates (measured on a grid), and two solves are not a sequence. `blame`, the imbalances, the impossible and set-aside lumens, the inferred directions, the cycles and the handed-on graph all come from the second solve. Its findings are applied in **one closing round of structural moves with the readings frozen**, by the rule of `blame`: an imbalance at a two-arm lumen, a lumen set aside, or an impossible lumen goes to the graph (a missing arm or a wrong join: step 3, step 1 for the rival bridges, ranked by the flux lost at the free ends). **Wrong speeds are not detected**: the rule that would blame a speed (its edge off in wide lumens at both ends) never fires on trees with a tenth of the speeds three times too fast, at any softness (measured), so the studentised residuals are reported with the base rate of |z| > 3 among right speeds beside them and are sent nowhere.

**Watching the fit.** Every fit in this plan is a sequence of candidates, some taken and some refused: the knot refinement of a spline, the κ passes, every move of every round, the sign moves of the flux network. A recorder (`vascnet.watch.Recorder`) is handed every candidate as a small state (the splines with their confidences, the vertices with their types and rates, the move's kind and gain, and for the rounds the topology residual's ridges and the render) and keeps one candidate in `WATCH_EVERY`, every taken one, and the last. Recording costs the fit nothing but memory; the frames are drawn afterwards (`vascnet.watch.animate`, with a drawing function per kind of fit) and written as a GIF, one per fit: `vessel_pictures/fit/initialisation.gif` (the splines settling and κ converging), `vessel_pictures/fit/round_<r>.gif` (each round's moves over the density, the proposed move highlighted, refused moves shown and faded, J and the self-consistency number in the caption) and `vessel_pictures/fit/flux.gif` (the sign moves and set-asides over the graph, directions as arrows, the move highlighted). The flux network's film exists already (`scripts/watch_flux_demo.py`, `docs/flux_fit_demo.gif`); the others follow their fits.

**Column:** found-and-typed-right after each round on the harness; on the real burst the self-consistency number after each round. As description, bridges right / wrong at the bridge confidence's cutoffs, the per-edge RMS, and the readings changed at (d) with how many of `blame`'s lumens survive the second solve.

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
4. `spline_to_graph` on section 29's splines, and the render-to-score pipeline: the self-consistency number of the initial graph on the real burst, which says how far section 29's splines are from a fixed point before any move is made.
5. Section 38's core-z check by radius; then sections 35 and 38 on one synthetic burst, section 39 round by round, the harness.

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
| the author's proposal after R5 | the topology residual moved into the orientation layer, channel by channel before the scale competition, with the self-consistency number of the render's ridges against the data's; the splines as the model and `spline_to_graph` as the view; the optical density kept for composition at cores and for line mass only; halos dropped from the fit |

## Open choices for you

1. The line mass for the flux: the kymograph's M or the profile's `π κ R²`. The plan uses M.
2. Composition where there is no reading: sum (the plan's choice) or max.
3. The real burst's base rate of true terminations against gaps and dives.
4. Whether the flow field (section 24) should be a second source of edges now or in the next version.
