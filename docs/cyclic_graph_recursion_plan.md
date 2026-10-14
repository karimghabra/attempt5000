# Plan: a clean cyclic graph from the plain ridges, rendered back and corrected by recursion

Version 9, the handoff version. This document is self-contained: it continues the burst notebook (`notebooks/development copy 2.ipynb`, sections 1–34) with sections 35–42, and an implementer needs the notebook, `vascnet/`, and this file. It went through five adversarial reviews (`docs/cyclic_graph_recursion_plan_review.md` to `_review_5.md`); the fifth's verdict was that sections 35, 36, 37, the moves of 39 and section 40 on trees had converged, with what remained being for the harness to measure. Versions 6 and 7 made the remaining design decisions and moved the residual for topology into the orientation layer; version 8 consolidated everything into one text, added the code units with their interfaces and tests, turned the order of work into acceptance steps, and wrote defaults for the open choices; a sixth review (`_review_6.md`) found that sections 37 and 38(a) were not buildable from that text as written and that four one-paragraph fixes would make them so, and this version makes them, with its smaller findings. Appendix A keeps the record of how every review finding was handled. Section 40 is prototyped (`vascnet/fluxnet.py`, `tests/test_fluxnet.py`; 39 tests of this plan's with `tests/test_watch.py`, 17 of the package's) and is described as it behaves.

**How to read this.** Part I says what the plan claims and what it builds on. Part II is the plan proper, sections 35–42. Part III is for the implementer: the code units, the acceptance steps, the defaults, the known limits. Part IV is the appendix.

---

# Part I. Claims and ground

## What this plan claims, plainly

- **Sections 37 and 41** (the graph with confidences, calibrated) should raise the harness's centerline and junction columns over the consolidated row, because they keep what the consolidation dropped and say how much each piece is worth. This is the cheapest section and the first to run; its one number decides whether the rest is built on it.
- **Section 40** (the flux network) should raise the direction columns on tree-like beds: it infers the unmeasured directions along whole vessels, turns lone wrong directions round where a lumen's arms are all measured or pinned by other lumens, which on the real pipeline means bifurcations and confluences (+3 to +7 points on binary trees with the thick level unmeasured, measured), and reports what it cannot settle. **On a plexus it does nothing, and the plan claims nothing there.** `join_pieces` joins every vessel through every crossing, so a vessel of a projected plexus is one measurement end to end and every balance pairs arms of one measurement: no sign is constrained, and a vessel with no measured piece is in the null space and gets no direction (measured). A true anastomosis is read as a crossing by `geometric_reading`, and the one-lumen reading of four arms has no tests and is never a rival, so the network cannot set the wrong structure aside; only section 38's core filter can read an anastomosis, and it is decisive for vessels of R ≥ 3 and marginal for capillaries. The plexus column of section 42 is expected to be zero until section 27 can read a thick anastomosis as one lumen.
- **Sections 38 and 39** (renderer, residual, recursion) should raise centerline and junction recall through the moves that add, remove, extend and bridge splines, because their evidence is the difference between the data's orientation score and the render's, channel by channel, which ignores what wrecked an optical-density residual (halos, the background fit, the blur's domain, the modulation of clusters along a vessel, κ's freedom) and has a number computable on the real burst without truth: the agreement between the render's ridges and the data's. They should raise found-and-typed-right where the core's matched filter has power, which stays in the optical density: for R ≥ 3 at every blur the generator makes (z 9 to 135, derived by the fifth review), and for capillaries only in focus (z 18.6 at a blur of 0.8 px, 2.4 at 2 px, 0.8 at 3.5 px). The recursion's moves are edits to splines; the flux network feeds the readings once, at the end, with a second solve after it and no loop.
- **Section 36** (junction cells) should raise junction recall at thick junctions; for capillaries it hangs on its first check.

## What exists in the notebook, and what the plan relies on

| name (section) | what it is | how the plan uses it |
|---|---|---|
| `orientation_scores` (20) | `(16, rows, columns)`, in noise units, before the lateral interactions: per orientation the winner over the eight widths (the widths compete inside `one_orientation`; no 16 × 8 array exists); saved by section 29 as `static_scores`. The noise constants are eight numbers, the robust sd of the even field per width on the quiet sample; the floor is defined per winning width on the quiet pixels. The hypercolumns take section 19's contrast, the density over the background fit's local noise map and `contrast_gains`, not the density | section 36 reads it; section 38 compares the render's against the data's per channel, which needs the 16 × 8 stack as new code |
| `static_strength`, `static_orientation`, `half_width` (29) | the winner over orientations of the pre-interaction score, `winning_neurons(orientation_scores, radius_maps)`, as section 29 saves it and `ridge_front` uses it. Section 21's field after the interactions (`static_fields`) is used by section 22's chains only; its facilitation raises chance rows of noise into ridges (measured: 73% noise ridge pixels against 6%) | the ridges come from these, and from nothing after the interactions |
| `ridge_crests`, `centerlines`, `extend_ends`, `consolidate_lines`, `ridge_lines` (18, 29) | ridges; a skeleton with spurs under `SPUR_LENGTH` = 12 px pruned (no setting); free ends extended into lines within `RIDGE_REACH` = 16 px where the median score along the extension reaches `T_LOW` / 2; the consolidation, whose joining and dropping steps are interleaved; pieces fork to fork as chains, each ending at its fork pixel (the fitted splines' ends lie within 0.63 px of it) | section 37 keeps the thinning and the joining, drops none of the pieces |
| `fit_spline`, `spline_vessel`, `picture_blur` (29) | least-squares cubic B-splines with knots `KNOT_SPACING` = 12 px apart for the place, three times farther for the half-width; points a pixel apart along the spline; the fit residual (0.28 px median, 0.44 max on the generator's smooth vessels); the picture's blur as one quantile per fixation | the initialisation; the model's splines; the renderer's blur |
| `read_arms`, `read_arm`, `contract_cores`, `fuse_split_crossings`, `vessel_graph` (26) | arms read on a ring from half the thickest width plus `RING_MARGIN` outwards (an edge that never leaves the core has its outer half read, inside the core); forks within a core fused; two three-way nodes within `FUSE_FACTOR` widths with straight outer arms fused into a crossing; `contract_cores` already drops spurs shorter than the core from a node's ends | `spline_to_graph` reuses the arm reading and the core rules |
| `set_partitions`, `group_tests`, `partition_reading`, `geometric_reading`, `choose_partition`, `read_junction`, `reading_confidence`, `where_axes_meet`, `propagate_directions` (27) | every partition of a vertex's arms scored by tests (turn, width, Murray, density, `overlap` of the core's 5 × 5 maximum against the arms', flux, pulse, blur, fingerprint, `in and out`); the most specific partition none of whose tests is off by more than `Z_PASS` = 3; `reading_confidence` = goodness of fit × (1 − exp(−margin)); a planar four-way read as a crossing; no group of one; a NaN test fails a partition | the junction readings; section 40's flux test enters `group_tests`; the `overlap` test is what the core filter generalises |
| `SIGMA_FLUX` = 0.45 (27) | the share of the arms' absolute flux sum a balance may miss | section 40's unit; re-measured per lumen kind |
| `join_pieces`, `whole_vessel`, `whole_vessel_flow`, `speed_of` (32) | pieces joined into whole vessels through the geometric partition; one kymograph and one direction per whole vessel; a join whose sides read against each other undone, only the count kept | the measurement ids of section 40; the provenance wrapper for undone joins |
| `direction_counts`, `reliability` (33) | p, the share of measured directions that are right, from three-in junctions and the sides of whole vessels | section 40's p |
| `synthetic_network`, `synthetic_burst`, `score_reading`, `harness_scores`, `HARNESS_WAYS`, `told_apart`, `with_interval` (28, 30, 34) | the generator (three trees, one split each, clusters at one speed per segment, density blurred per tree, no background fit, no vessel thinner than about 2.5 px), the scoring, the ablation table with held-out seeds | section 35 extends the generator; section 42 adds rows and columns |
| `found_again` (18), `repeated` (29) | how much of one ridge map is found again in another within `REPEAT_WITHIN` = 3 px, above chance; one number per fixation. The harness's own centerline metric is a different function, within 2 px | the self-consistency number (defined through `found_again`, within `REPEAT_WITHIN`); per-edge repeatability (new code) |
| `vascnet.synth.render`, `_branch_od` (package) | the unblurred chord density `OD(d) = 2 κ sqrt(R² − d²)`; max within a tree; transmittance blurred; multiplied across trees | the renderer follows it |
| `vascnet.fluxnet`, `vascnet.watch` (package, this plan) | the flux network; the recorder and the film | section 40; watching every fit |

## Principles

- **Nothing is deleted below the thinning scale; everything carries a confidence; the principle has a number.** On the held-out bursts the edge confidence must tell true pieces from false ones with `told_apart` ≥ 0.75 on the five bottom-up terms (the sixth comes from section 38, which is built last), and the confidence curve must pass the consolidated row's point. Below that, section 29's consolidation is kept.
- **One objective, written down.** `J = χ²_topo + λ Σ_splines k ln(n)`: the topology residual of section 38(a) squared over its footprint-counted samples, plus a price per spline parameter, λ the one swept multiplier. The *structure* of the model (which splines exist and are drawn, how they are grouped into lumens, where the centerlines lie) changes only by moves, taken when J falls. κ and the radii are measurements on windows and arms, not optimised under J; they jump when a move changes the windows (a merge moves the core radius), and the jump is in ΔJ. The price counts measured parameters as it counts fitted ones; λ absorbs the difference, which is why λ is swept. λ prices structural moves only (the initialisation's knot refinement has its own price, in the ridge's unit, below). λ is chosen on the real burst as the largest value at which the shares of impossible junctions and of directed cycles, from the flux network run on the current model, do not exceed the consolidated graph's, with repeatability and loose ends reported beside it, never combined. The consolidated graph's shares, the rule's reference, are measured at acceptance step 3. λ itself can only be swept once structural moves exist (acceptance step 7): before the recursion no move depends on λ, so every λ gives the same model.
- **A reading and the drawing never disagree.** A re-composition proposed by a reading is a move; if J vetoes it, the reading reverts. Bridges are never drawn.
- **The orientation layer decides what lines exist and where; the optical density decides how they are composed and how much blood they carry.** Section 38 has one residual for each.
- **Flow decides what a still cannot, where it can**, and section 40 states exactly where it cannot.
- **Each section has one column it has to move**, named in advance, on the held-out bursts. Everything else in the table is description.
- **Every fit is filmed.** A recorder keeps every few candidates and every taken one; the film is drawn afterwards.

---

# Part II. The plan

## 35. Synthetic bursts with the things this plan is about

Extend the generator (section 28), keeping the old seeds bit-identical (sections 30 and 34 must reproduce to the digit):

1. **Capillaries**: trees with R of 1 to 2 px. The generator's thinnest vessel today is about 2.5 px, so no claim about capillaries can be checked on it.
2. **Loops**: a daughter rejoins another branch of its tree with probability `SYNTH_LOOP`; its speed follows from flux conservation at both ends.
3. **A plexus of two kinds**, recorded apart: projected crossings at several depths, and true anastomoses (mixing nodes of four arms in one lumen), with the readings and measurement ids the pipeline makes for them (crossings from `geometric_reading`, one measurement per `join_pieces` path), since that is what section 40 sees.
4. **Gaps**: a segment fades over `SYNTH_GAP_LENGTH` px to `SYNTH_GAP_DEPTH` of its density with probability `SYNTH_GAP`; recorded.
5. **Terminations** inside the field.
6. **Confluences**: both flow directions equally, as now.
7. **Halos of either sign** around thick vessels (scatter, or what a background fit leaves).
8. **Shared lumens and collinear overlaps**, recorded per junction; a few segments of two trees running along each other with nothing emerging, so that the miss section 38 accepts is counted.
9. **A velocity profile across the vessel**: clusters at radial positions with speed `v_max (1 − (r / R)^k)`, `k` = `SYNTH_BLUNT` ≈ 3. Until this is in, the harness's flux columns are upper bounds and the table says so.
10. **Blur in the transmittance domain**, as `vascnet.synth.render` and the camera do.
11. **The background fit** of section 12 run on the synthetic bursts, or a local mean of `BACKGROUND_SIGMA` subtracted with the harness marked approximate there.
12. **Per-junction `n`** in the truth (3 at a bifurcation, 4 at a crossing, the sum of arms at a fused "complex"), which section 36's column needs and `junction_metrics` does not give.
13. **A second fixation per seed**: the same truth imaged again with fresh noise and a known shift, with the map between the two as the real burst's fixations have it, so that `repeated`, `found_again` and the per-edge repeatability have a value on the harness (the harness today makes one burst per seed, and section 37's gate counts a term it could not produce). The data's own repeatability between two noise realisations of the generator is 0.995–1.0.

**Column:** none; per new seed the counts of loops, gaps, terminations, plexus junctions of each kind and collinear overlaps, and the old seeds proven unchanged.

---

## 36. Junction cells

A junction cell at a pixel, for a ring radius ρ, counts the arms of a junction there and fires in proportion to how clearly it sees them. It runs everywhere, not only where the skeleton has a fork.

**Input.** `static_scores[n]` (the pre-interaction score, 16 layers) and `half_width[n]`.

**Mechanism** (`junction_cells(scores, half_width, rings)`):

- On a ring of radius ρ, at angles φ in steps of 360/`RING_STEPS` degrees, read the score of the neuron whose orientation is the radial direction φ, interpolated between the two nearest of the 16. The scale competition has already chosen a width per pixel.
- Peaks of that angular profile above `T_LOW`, joined to one above `T_HIGH`, are arms; two peaks within `SAME_ARM` degrees are one arm.
- `n` = the number of peaks. `rate` = the geometric mean of the peak scores in noise units, times a core term (the core's density against the arms': a junction is at least as dark as its arms).
- Rings: `JUNCTION_RINGS` = (4, 6, 10, 16, 24) px. The smallest is where `read_arms` reads a capillary junction (half-width 1.5–2 px plus `RING_MARGIN` 2). Per pixel the ring with the largest rate is kept, with its `n`.
- Non-maximum suppression within ρ/2, **per ring separately**, so that a thin daughter's junction 6 px from a thick trunk's is not suppressed by the 24 px ring.
- `n` = 1 is an end cell; `n` = 2 within `STRAIGHT` degrees of a line is a line cell and is suppressed; `n` = 2 with a turn is a bend; `n` = 3 and `n` = 4 are candidates for a bifurcation and a crossing; the type is read by section 27's partitions.
- **Plaid cells**: pixels where two orientations at least `PLAID_APART` = 45 degrees apart both exceed `T_HIGH` in the pre-interaction score. The weaker line of a crossing is there; the tuned field is not read here.
- Read on the render as well as on the data (section 38): an `n` the data's cell has and the render's lacks is a junction residual.

On a rendered capillary bifurcation the 4 px ring resolves daughters 40 degrees apart on the density (measured); whether it does on the oriented score is check 1.

**Checks, before any cell map is built for a whole frame**: a thick vessel (R = 6) crossed by a thin one (R = 1.5), on the oriented score (does the thin arm appear as a peak on the ring that matches the thick vessel, at what rate, and on its own ring? If not, a second reading of the ring at the thin width is added, and the plan says so); a thin daughter leaving a thick trunk; a crossing the skeleton broke into two T's 5–10 px apart (does the cell at the midpoint say `n` = 4?).

**Measured (acceptance step 4, eight noise realisations per configuration and blur):**

- *Thin vessel crossing a thick one.* At blurs of 0.8 and 1.5 px the thin arm is an arm on every ring, its own 4 px ring included, in 8 of 8 pictures; densities add across lumens, so the thin line stays visible over the thick vessel. At 3 px it is found on its own ring in 5 of 8, and on the 16 and 24 px rings in 7 of 8. The second reading at the thin width (`width_channels`) sees it exactly where the first does at 0.8 and 1.5 px, and never at 3 px: a thin vessel blurred by 3 px is no longer a thin line. The second reading is therefore built but not used.
- *Thin daughter of a thick trunk, one lumen.* The daughter is an arm only on the rings outside the trunk (16 and 24 px), at 0.8 and 1.5 px blur, and on none at 3 px. The ring with the largest rate is always the two thick arms' (rate about 98 against 45), so the cell reads a line and misses the junction. The ring with the most arms among those whose rate is above `T_HIGH` reads three arms at 0.8 and 1.5 px.
- *A crossing the skeleton breaks into two forks.* Two R = 3 vessels crossing at 45 degrees give forks 8–11 px apart. The cell at their midpoint reads four arms in 8 of 8 pictures at 0.8 and 1.5 px and in 7 of 8 at 3 px.

**Result:** `junction_maps[n]`: `n`, `rate`, `ring`, `plaid`.

**Column:** junction recall at the plain ridges' junction precision, scored by each junction's `n` against the truth's `n` (section 35, item 12; the harness's gate `max(6, 2R + 2)` matches a cell and two forks alike, so the per-junction `n` is what is scored).

---

## 37. The clean cyclic graph

**Input.** `ridge_crests` on the **pre-interaction winner** (`static_strength` as section 29 saves it; never section 21's field, whose facilitation makes most of its ridge pixels noise) → close within `RIDGE_JOIN` → `centerlines` (its `SPUR_LENGTH` = 12 px pruning kept: a 1–11 px stub is a thinning artefact, and on an unpruned skeleton `extend_ends` would extend several hundred of them per frame into neighbouring lines, measured on a proxy) → `extend_ends` → `ridge_lines` with `least = 0`, `to_forks = True`. None of `consolidate_lines`' dropping steps (`RIDGE_CHILD`, `LINE_LEAST`, `RIDGE_NETWORK`) is run. This is new code, since the joining and dropping steps are interleaved in `consolidate_lines`; the section reports what it keeps that the consolidation dropped.

**The model is the set of splines; the graph is a view of it** (`spline_to_graph`, new code). Each spline carries its place and half-width splines (`spline_vessel`), κ or line mass, its confidence and terms, its provenance and flags (`drawn`, `bridge`, `twin`, `stub`, `found in residual`). The graph is derived: a vertex wherever splines end, meet or cross, an edge for every piece of spline between two vertices, and the lumens from the readings. The rules:

- k endpoints within `RIDGE_JOIN` of each other are one vertex of k arms (two: a join; three: a fork); an endpoint within `RIDGE_JOIN` of another spline's body is a T, and the body is split there: both halves inherit the body's κ windows, confidence terms and lumen, and the lumen is re-read. The initial pieces end at their fork pixels (within 0.63 px), so the T rule is reached only by moves;
- two bodies that intersect are a crossing vertex at the intersection, moved to where the arms' axes meet (`where_axes_meet`); two bodies that cross twice within one core fuse by the core rule, and outside one are two crossings and a multi-edge; two bodies that run side by side are a twin by `read_arms`' one rule (within `SAME_ARM` degrees, centres within the larger width plus `RING_MARGIN`), marked, not a vertex, and the pair's overlap is read for one line mass, not per spline; two bodies that meet at under `SAME_ARM` degrees are a join candidate, not a crossing;
- vertices within one core radius are one vertex (`contract_cores`); a spline that intersects itself makes a loop vertex; multi-edges and self-loops are allowed; a free end carries `at_data_edge`; zero-length pieces inside a core disappear into the vertex;
- intersections are found by bounding boxes and a Newton step on the two splines; a twin and a crossing at a glancing angle are told apart by the angle and the length of contact.

The initial spline set is the one the initialisation below fits; every structural move of section 39 is an edit to the spline set (add a spline, remove one from the rendering, extend an end, join two end to end, split one at a point, move a centerline), and the graph is recomputed where the edit reaches.

**Vertices.** Every fork and every free end; forks within one core are one vertex (`contract_cores`, with `fuse_split_crossings` as it runs in `vessel_graph`); a junction cell within a core adds its `rate` to the vertex; a cell with no fork within its ring becomes a vertex with no edges yet (section 39, step 3, looks for its arms). A free end carries `at_data_edge` (within `EDGE_MARGIN` of where the data ends; not a free end in any later sense).

**Edges.** Every piece between two vertices, as a spline. Twins (by `read_arms`' rule above) are kept, the shorter marked `twin` and not an arm. Multi-edges and self-loops allowed.

**Initialisation: how the first splines are placed, and when the first network is done.**

1. *Ridges.* `ridge_crests` on the pre-interaction winner: a pixel is on a ridge where its strength is the largest across its own orientation, above `T_LOW` and joined to a pixel above `T_HIGH`. The skeleton's pixels are integers; crest neurons carry a place to a fraction of a pixel, across their own orientation. The initialisation runs `crest_neurons` on the pre-interaction score itself (`static_scores`, the score the ridges come from; section 29's saved inputs hold no interaction field), takes the neuron of the pixel's winning orientation (the nearest in k, within one step) and falls back to the pixel's centre (new: `fine_lines` keeps the offset). Measured on synthetic bursts 1 and 2 (step 2's report): a crest on 84–95% of the points, against 78–89% with section 21's field. The fit residual at 12 px knots falls from 0.31 and 0.26 px median to 0.22 and 0.14, against 0.25 and 0.16 with section 21's crests. The points' distance from the true centerline falls less, from 0.46 to 0.38 px rms (median over 120 pieces of eight bursts).
2. *Pieces.* Closing within `RIDGE_JOIN`, the skeleton with its `SPUR_LENGTH` pruning, `extend_ends`, then `ridge_lines` with `least = 0` and `to_forks = True`: every piece runs fork to fork or fork to free end with the fork pixel at each end. Section 29 leaves pieces under `SPLINE_LEAST` = 5 px out; here they are kept as straight segments, since `fit_spline` fits a line to two points already.
3. *Splines.* Each piece through `spline_vessel`: a least-squares cubic B-spline for the place with knots `KNOT_SPACING` = 12 px apart, one for the half-width with knots three times farther, points re-sampled a pixel apart along the spline, and the fit residual (the rms distance of the ridge points from the spline) recorded. *Knot refinement* (new), **priced in the ridge's own unit and needing nothing from section 39**. The piece's χ² is its summed squared distance from the spline over `RIDGE_PRECISION`², counted in independent samples:

- `RIDGE_PRECISION` is 0.38 px with the offsets and 0.46 px at the pixels' centres (measured);
- `RIDGE_CORRELATION` = 1.9 points make one sample (measured: the deviations are correlated along the line). Counted as independent points at 0.25 px, the generator's smooth vessels take 6 px knots on 8% of their pieces.

Two moves are priced as a BIC, at ln(samples) per added coefficient, with no λ:

- *Halving.* The piece's knot spacing is halved, down to `KNOT_LEAST` = 6 px.
- *Split.* The piece is cut at its point of largest residual, and both halves are refined again; the split makes a two-arm vertex that section 36 calls a bend. It is taken only where the halves meet at a bend of more than `FUSE_STRAIGHT`.

The split is a move of its own, not what the halving ends in, for two reasons (both measured):

- A halving over the whole piece adds coefficients along all of it, so it never pays for a local corner: a 90-degree corner on a 79-point piece gains less than its price at every precision tried.
- Priced by residual alone, a split cuts a ridge pulled 1–1.8 px aside near a crossing as readily as a corner.

On the generator's vessels (16 bursts):

- of those that lie on one segment away from junctions, 24 of 25 keep 12 px knots and one is split;
- of the pieces that pass a junction, 10 of 201 are split.

A spline is done when no move lowers the priced χ².
4. *Darkness.* κ (or the line mass) per window outside the cores from the chord-profile fit of section 38, the thickest vessels first and the thinner ones on the density with the thicker ones' render subtracted (exact under the renderer, since densities add across lumens after each one's blur), repeated until no κ changes by more than its own error: a Gauss–Seidel pass over the splines, two or three times. A marked twin or overlap is read as one line mass for the pair, not per spline, since the two would otherwise share one density in an order-dependent split.
5. *The graph.* `spline_to_graph`; every vertex moved to where its arms' axes meet (`where_axes_meet`, read on the ring); cores contracted; the junction cells' rates added; readings by geometry.

The first network is done when three things hold, none of which needs truth:

- every spline's priced residual is at the ridge's precision, every vertex is at its arms' meeting point, and no two vertices lie within one core;
- κ has converged;
- **self-consistency along the model's own support**: the splines are rendered, scored and their ridges extracted, and within the tubes of the existing splines (the radius plus `TUBE_BLURS` sigmas of the blur) the render's ridges agree with the data's, within `REPEAT_WITHIN`, at least as well as the data's ridges agree with the other fixation's (`repeated`'s share above chance, section 29; on the harness, the second fixation of section 35). That is the most the render can be asked to match, since the data does not agree with itself better. A disagreement inside a tube is a measurement problem and is given the one mechanism that exists: the centerline shift of section 39's step 3, run within the tubes before the rounds, with R re-measured from the half-width map and κ from the profile where the spline moved; what that does not fix is reported, and the condition is met as a share, not pixel by pixel (measured on the generator with the exact truth rendered: 0.991–0.998 within the tubes against the data's own 0.995–1.0). A disagreement outside the tubes, a ridge the data has and the model lacks or the reverse, is the recursion's input, not the initialisation's failure.

Then the self-consistency number over the whole frame, with no restriction to the tubes, is recorded as round 0, and the recursion begins. "Done" does not mean every ridge pixel has a spline above a cutoff (stubs and short pieces are splines with low confidence), and it does not mean the topology residual off the splines is empty; that residual is what the rounds are for.

**Confidence per edge**, each term a log-likelihood ratio calibrated on the harness on this graph alone before section 39 is written (section 41), then summed and squashed:

- length in field lengths (`ELONGATION` × half-width; zero at one, full by three);
- strength: the median score in noise units against `T_HIGH`;
- **repeatability per edge** (new code: `found_again` of section 18 applied per edge against the other fixation's ridges; `repeated` gives one number per fixation; on the harness the other fixation is section 35's item 13);
- flow data: `flow_share`;
- twin and stub: lowered for a twin, and for a free-ended piece inside a parent's tube (`tube_reach`);
- from section 38, later: the rise of J without the edge.

**Structural confidence per join**, which section 40 consumes, and which is never 1 − ε where the new synthetic cases can make the structure wrong:

- a two-arm join: P(join right | the pair's turn, width ratio and density ratio), calibrated on the harness as a probability; 1 − ε only for two pieces that were one connected ridge before the closing within `RIDGE_JOIN`, since a closing or thinning can fuse two vessels ending near each other or a collinear overlap, which section 35 now makes;
- a join `extend_ends` made, whatever the arm count of the fork it made: the extension's confidence;
- a join section 32 undid: 0.5, no balance;
- a bridge: its confidence;
- a three-arm vertex: not one structure. `set_partitions` allows no group of one, so every three-arm vertex is one lumen of three today, and a T whose stem dives (a through pair plus a free end) is read as a bifurcation and is silent in the network (measured: −0.2 to −1.9σ at 0.45). Section 27 gains the partition "through pair + free end" at three-arm vertices, with the far-side check of `geometric_reading` (which already reads a thin-stemmed T as a crossing and then discards it as a baseline) as its evidence; the bifurcation's structural confidence is then its margin over that rival;
- a four-arm vertex: P(two vessels, not one lumen), from section 27's `overlap` z now and section 38's core z later, calibrated; the margin of the best pairing over the second pairing does not ask this question (it is infinite above a 45-degree acute angle and never compares a crossing with one lumen).

**Provenance**, as wrappers that change no section 29 or 32 code: the pixels `extend_ends` added are `extended & ~lines`, and a fork whose incident pixels include them is extension-made; the joins section 32 undid are the partner pairs of `join_pieces(graph)` run without `undone` whose two edges carry different `vessel` numbers in `graph["readings"]`.

**Confidence per vertex**: the cell's `rate`, times the reading's confidence (section 27), times the flux test's fit (section 40).

**Clean, as numbers**: no two vertices within one core; no edge under two pixels; every edge a spline with a radius profile; the cycle count (edges − vertices + components) with the share of cycles under `RING_LEAST` px in circumference.

**Result:** `cyclic_graphs[n]`: `vertices`, `edges` (with confidence and its terms), `cycles`, and the spline set they are derived from.

**Column:** `told_apart` ≥ 0.75 of the edge confidence between true and false pieces on the five bottom-up terms (the repeatability term from the harness's second fixation), with the centerline and junction precision–recall curves against the plain and consolidated rows, and the share of skeleton two-arm joins that are false.

---

## 38. The renderer and the residual

**Per-edge measurements** (`edge_profiles`): at windows of `KNOT_SPACING` px along an edge, **excluding every window whose centre lies within the core radius of either end vertex**, the density profile across the edge over ±`PROFILE_REACH` half-widths (`profile_across`, section 21), fitted with the cylinder chord profile convolved with the fixation's measured blur (`picture_blur`, section 29): R and κ per window and an offset c; κ and R smoothed along the edge with the smoothing stopping at the core radius. For an edge thinner than the blur, R and κ are not separable: `R_od` only where `R_ridge` > σ; the line mass `π κ R²`, which is identifiable, elsewhere. Two radii, `R_ridge` from the scale competition and `R_od` from the profile, disagree only above the blur, and there a disagreement beyond `SIGMA_WIDTH` in log marks an overlap, a halo or a twin.

**Edges with no window** (shorter than two core radii plus a window: split crossings, plexus capillary pieces; their share is reported): κ and R come from the lumen the edge belongs to (the whole vessel of `join_pieces`, whose other pieces have windows); for an edge that is its own lumen, the line mass from `read_arm`'s outer half and R from `R_ridge`. The core filter at such an edge's vertices has reduced power, and the plan counts it.

**The renderer** (`render_graph`), following `vascnet.synth.render`: the edges are grouped into **lumens** (the whole vessels of `join_pieces` plus the groups of three of every bifurcation reading); each lumen is drawn unblurred into its own canvas as chord density along its splines, **max** within the lumen; each canvas is converted to transmittance, blurred by the fixation's σ, the lumens **multiplied** (a canvas is transmittance 1 outside its own extent, so a local canvas multiplies correctly), and the picture is `−log` of the product. Densities add across lumens, unite within one, and the blur is where the camera's is. Canvases are small and local: a move that touches two lumens re-renders two canvases. Halos are not fitted (the orientation layer does not see them; the core test takes the background around a core as the quiet pixels' level).

**Two residuals, for two questions.**

**(a) Topology, in the orientation layer.** The render is first put through section 19's `contrast_layer` with the data's own background-fit noise map and `contrast_gains` (the hypercolumns take the contrast, not the density), then through section 20's hypercolumns with the data's eight noise constants (the robust sd of the even field per width on the quiet sample), never re-measured on the render, which has no noise; the floor, which section 20 defines per winning width, is measured per channel on the data's quiet pixels or left out. The comparison is **per channel, 16 orientations × 8 widths, before the width competition**. That stack does not exist in the notebook (the widths compete inside `one_orientation` and only the winner is kept); it is new code of about twenty-five lines, 512 MB per stack in float32 at 1000 × 1000, so it is kept in float16 or computed per window. The topology residual is `score(data) − score(render)` per channel, **against a zero point measured per channel on the initial model's own splines at round 0**: on a correctly drawn line the data's rectified answer carries noise with a positive expectation and the render's carries none, so the difference on a right spline is about −1 noise unit (measured: −0.8 to −1.2 at the centre, −0.3 to −2.0 across the widths), and a "negative ridge on a spline" means something only above that. Off the vessels most channel-pixels are exactly zero (white noise through the bank), so χ²_topo is a working objective whose differences under λ mean something, not a χ² statistic. Its ridges, after the zero point, are the moves' evidence: a positive ridge is a line the data has and the render lacks (add a spline, extend an end, bridge a gap where it continues a spline), a negative ridge a line the render has and the data lacks (remove from the rendering). The junction cells of section 36, read on the render and on the data, give the junction residual the same way.

Why per channel: a thin vessel running beside a thick one **with two to four pixels of clear gap** answers in its own width's channel while the competition still picks the thick vessel's width at that pixel (measured: 1.9 then 7.2 in the width-1.5 channel against 17.6 and 18.2 for the winner), and from five pixels on the winner map shows it anyway. **A thin vessel touching the thick one's wall is a stated limit**, beside the collinear overlap below: in the data itself its thin channels are exactly zero (the thick vessel's edge drives the odd answer above the even one and the energy clips), and its only trace is an asymmetry in the width-12 and 17 channels that a half-pixel shift of the thick spline also leaves. A thin vessel crossing a thick one is fine: its channel is zero within 8 px of the thick centre and carries its tails from two pixels beyond the wall.

What this removes, against an optical-density residual for topology: the halos and the background fit (the filters are band-pass and ignore both), the blur's domain (the render goes through the camera's blur and then the same filter as the data), the cluster modulation along a vessel (slow, and ignored by a line filter), and κ's freedom (a missed vessel answers in its own channel whatever κ its neighbours were given). What it costs: a hypercolumn run per evaluation (FFT-based over the frame; 3.7 s on 1000 × 1000 with four threads, 11 s with one; no interactions are needed, which saves their 6.6 s), and a residual correlated over the filter's footprint, so the per-move price counts independent samples per channel as pixels over the rectified channel's correlation area, which is 2.5 × width² within 10–40% (measured) and needs no measurement on the data. **Windowed re-scoring** equals the full frame's only when the window's origin and size are multiples of 20 px (widths 6 and up are computed on grids reduced by 2, 2, 4 and 5) and with a margin of 110 px around the move (at 70 px the width-17 channel is off by 0.075), against the frame's own 32 px mirror padding: a 320 px window costs 2.0 s against 3.7 s, so a round of a thousand candidates is half an hour to an hour of scoring, and the candidates of a round are batched by region.

**The self-consistency number.** The competition and `ridge_crests` run on the render's score (no interactions: they raise chance rows of noise into ridges) give the render's ridges; their precision and recall against the data's ridges, through `found_again` within `REPEAT_WITHIN` (the harness's own centerline metric is a different function, within 2 px), is a number on the real burst that needs no truth. A model is a fixed point of extract → fit → render → score → extract when it is 1: the generator's exact truth, rendered without noise and scored with the data's constants, reaches 0.996–0.999 in both directions (measured on two seeds), so the fixed point is reachable. At round 0 it says how far section 29's splines are from one, and it is section 39's column on the real burst.

**(b) Composition and mass, in the optical density.** What the orientation layer cannot see: one lumen against two at a core (a darkness difference at the same orientation and width), the line mass, κ. These stay in the optical density, at cores and along edges only, never for topology. The noise for these is the quiet pixels' two-dimensional autocorrelation and spectrum off the graph; the core z rides on the white floor (shot and read noise over the kept frames), not on the texture, so the spectrum is measured, not only the short-lag autocorrelation. The cluster modulation that the frames' mean leaves on a vessel (a tenth to a quarter of the chord, correlated over 30 to 40 px along it) is absorbed by the per-window κ and is named as the dominant on-vessel density residual.

**The core's matched filter.** The template is the difference between the two compositions (one lumen, two vessels) rendered at the core from the arms' κ and R; its whitened inner product with the density residual over its norm is a z for two vessels against one lumen. What it adds to section 27's `overlap` test: every pixel of the core instead of a 5 × 5 maximum, the right spatial template, and the noise's correlation. **Its reach, derived for the generator's noise** (the fifth review): two vessels crossing at 90 degrees, R = 1.5 each, give z = 18.6, 2.4 and 0.8 at blurs of 0.8, 2 and 3.5 px; R = 3 each give 44, 17.5 and 9.2; R = 5 and 3 give 135. Decisive for R ≥ 3 at every blur the generator makes, and for capillaries only in focus. **For a four-arm vertex this is the one test that can tell a crossing from one lumen of four**: geometry reads a planar four-way as a crossing, and the flux test at the notebook's softness passes both (section 40). At a blurred capillary four-way geometry keeps deciding, and the plan does not claim the filter there. A **check before section 39**, after section 35 has capillaries: the z by radius and blur on rendered crossings against recorded one-lumen junctions.

**Where the residuals are read, and what each reading means.**

1. **The topology residual's ridges**, off the existing splines and continuing them: the structural moves' evidence (steps 1, 3, 5 of section 39).
2. **The core filter's z** at every vertex: the composition's evidence (steps 2, 3b, 4).
3. **The density residual along an edge**, for what the per-window fit could not absorb: features shorter than the knot spacing, cross-profiles one blurred cylinder cannot make, and the ends; as description, by radius.

| residual | reading | move |
|---|---|---|
| the core filter says two vessels under one lumen, or the reverse | the composition is wrong | step 4 |
| a positive ridge of the topology residual continuing a spline across a gap | a gap to bridge, or an end to extend | steps 1, 3 |
| a positive ridge off the splines, meeting them at a vertex the data's cell sees and the render's does not | a missing arm | step 3 |
| a positive ridge off the splines, through a spline, with tails on both sides | a crossing vessel is missing: its tails, linked by the association field | step 5 |
| a positive ridge off the splines, meeting nothing | a vessel nobody has | step 5 |
| a negative ridge on a spline | a line the data does not have: a stub, a twin, a bridge that should not have been drawn | step 3b |

**The rule for new vessels, and the miss accepted.** A new spline is proposed only on evidence outside the pixels of existing splines: the topology residual's ridges off them, including the tails of a shallow crossing. A vessel lying exactly along an existing spline with nothing emerging leaves no trace in either residual (the per-window κ absorbed it in the density, and the orientation layer sees one line) and is not looked for; the miss is a stated limit of the still, counted on the harness (section 35, item 8). The kymograph could see it (two families of streaks where one vessel has one); that stays a later, separate test that adds no spline by itself.

**Result:** `rendered[n]`, `topology_residual[n]` (per channel, after its zero point) and its ridges, the self-consistency number, `edge_profiles[n]`, per spline the rise of J without it, per core the matched-filter z, the share of windowless edges.

**Column:** `told_apart` of the core z between one-lumen and two-vessel junctions, by radius and blur; the self-consistency number at round 0 and the per-edge density RMS by radius as description.

---

## 39. The recursion

Moves under J; λ chosen on the real burst by the rule of the principles; at most `ROUNDS_MOST` = 5 rounds; the rounds end because J falls at every taken move and every structural change is a move.

Every move is an edit to the spline set, the graph recomputed where it reaches (`spline_to_graph`), and the evidence for a structural move is the topology residual of section 38(a); for a composition move, the core z of section 38(b).

**Rounds 1 to 4: structure, with the readings from geometry and the core filter only.**

- **Step 1, bridging free ends.** A bridge is never accepted, rejected or drawn. Candidates: pairs of free ends not at the data edge, within `BRIDGE_REACH` = 10 × the larger width (at least `RIDGE_REACH`), each end's direction in a cone of `SIGMA_BRANCH` around the other; the path a cubic Hermite spline between the two ends' directions. Confidence from similarity on the distal profiles outside the cores: line mass within `SIGMA_DENSITY` in log (not radius, which thin vessels do not have), blur alike, the bend each end makes within `SIGMA_TURN`, and where there is flow, direction agreeing at reliability p, speed within `SPEED_AGREE` sigmas, pulse phase within `SIGMA_PHASE`. These terms separate thick ends and not capillary pairs; the flow terms need a clear speed on both sides; the column reports `told_apart` for thin and thick ends apart. Rivals are not resolved here. Recorded beside the confidence: whether a positive ridge of the topology residual continues a spline across the gap, and whether the kymograph's pattern continues across it. In the flux network a bridge is an edge with a prior of zero flux scaled by the flux of the vessel it continues, a free end's rival bridges are one lumen with the end's arm, and the flux lost at the free end ranks the candidates.
- **Step 2, merging arms through a junction.** The partition reading of section 27 with the core filter among its tests, the arms compared by their distal profiles; arms that are one vessel joined into one lumen (`join_pieces`); the re-composition a move; the whole vessel re-read for its speed.
- **Step 3, validating arms.** The junction cell's `n` on the data against the render's: an arm the data's cell sees and the render's does not is a positive ridge to string from the cell's unexplained peak within `RIDGE_REACH`, added as a move; an arm the render has and the data's cell does not is a candidate for removal. Centerline shifts outside cores: a move with zero parameters, judged by the topology residual, taken only if J falls.
- **Step 3b, undoing.** *Remove a spline from the rendering* (a negative ridge; it stays in the model with `drawn = False` and its confidence down, and can return). *Merge two vertices* within a core, proposed by the cell (`n` = 4 at the midpoint of two forks of degree 3; `fuse_split_crossings` finds the straight case already, the cell proposes the bent one too) and decided by the merged vertex's reading with the core filter and the flux test among its tests; J a veto on which the reading reverts. *Split a vertex* of four or more arms whose groups' axes (`where_axes_meet` per group) meet apart by more than the core radius; decided the same way.
- **Step 4, composition.** Where the core z contradicts the composition by more than `Z_PASS`, the other composition is a move.
- **Step 5, missing vessels.** The positive ridges of the topology residual off the existing splines, strung (section 22's stringing on the residual's crests) and fitted, each a move; a tail on either side of a spline linked through by the association field.

**Step 6, the flux network, in the last round: two solves, no loop.** (a) The readings as they stand; (b) one solve on the whole graph, with a measurement id per whole vessel, structural confidences per join, bridges as doubtful edges in their end's unit; (c) every vertex's candidate readings scored by the leave-one-vertex-out tests; (d) the readings taken once more, the flux test able to fail a reading and never to choose among readings that fit; (e) **a second solve under the readings of (d)**, with a vessel that (d) split carried as separate measurements with the old direction, and **no re-reading after it**: a sequence of readings and solves oscillates (measured on a grid), and two solves are not a sequence. `blame`, the imbalances, the impossible and set-aside lumens, the inferred directions, the cycles and the handed-on graph all come from the second solve. Its findings are applied in **one closing round of structural moves with the readings frozen**, by the rule of `blame`: an imbalance at a two-arm lumen, a lumen set aside, or an impossible lumen goes to the graph (a missing arm or a wrong join: step 3, step 1 for the rival bridges, ranked by the flux lost at the free ends). **Wrong speeds are not detected**: the rule that would blame a speed (its edge off in wide lumens at both ends) never fires on trees with a tenth of the speeds three times too fast, at any softness (measured), so the studentised residuals are reported with the base rate of |z| > 3 among right speeds beside them and are sent nowhere.

**Watching the fit.** Every fit in this plan is a sequence of candidates, some taken and some refused: the knot refinement of a spline, the κ passes, every move of every round, the sign moves of the flux network. A recorder (`vascnet.watch.Recorder`) is handed every candidate as a small state (the splines with their confidences, the vertices with their types and rates, the move's kind and gain, and for the rounds the changed lumen canvases and the window's residual, never the frame; the frame-size render only on taken states) and keeps one candidate in `WATCH_EVERY` (`Recorder.every`, default 5), every taken one, and the last. The recorder stores what it is handed, so the fit copies the arrays it hands over, as `fluxnet.py`'s `report` does. Recording then costs the fit nothing but a few megabytes per kept state; the frames are drawn afterwards (`vascnet.watch.animate`, with a drawing function per kind of fit) and written as a GIF, one per fit: `vessel_pictures/fit/initialisation.gif` (the splines settling and κ converging), `vessel_pictures/fit/round_<r>.gif` (each round's moves over the density, the proposed move highlighted, refused moves shown and faded, J and the self-consistency number in the caption) and `vessel_pictures/fit/flux.gif` (the sign moves and set-asides over the graph, directions as arrows, the move highlighted). The flux network's film exists (`scripts/watch_flux_demo.py`, `docs/flux_fit_demo.gif`); the others follow their fits.

**Result** per round: moves proposed and taken, J, the self-consistency number, the model; the last is `graph_final[n]`.

**Column:** found-and-typed-right after each round on the harness; on the real burst the self-consistency number after each round. As description, bridges right / wrong at the bridge confidence's cutoffs, the per-edge density RMS, and the readings changed at step 6(d) with how many of `blame`'s lumens survive the second solve.

---

## 40. Net flux = 0: a flux network (prototyped)

`vascnet/fluxnet.py`; `tests/test_fluxnet.py` on a toy graph, on grids with harmonic flow as a true anastomotic net, and on a plexus of projected crossings. What follows is what it does and what was measured.

**Unknowns, measurements, balances.** One flux per edge; measured on edges with a clear speed, `M × v` (the kymograph's line mass times the whole vessel's velocity); free elsewhere. One balance per lumen, allowed to miss by `SIGMA_FLUX × Σ|q|` over the lumen's arms, the notebook's unit. **`SIGMA_FLUX` is measured per lumen kind** (through-lumens of one vessel, where the velocity-profile factor cancels; bifurcations, where it does not), on junctions whose directions are measured and possible, with the counts printed; the population is capillaries, since the thick vessels are above the trackable limit, and the plan says so; where the real burst has too few, the harness's value stands in.

**The unit of a direction is a measurement**: one whole vessel (`join_pieces`' path), all its edges turned round together at one cost `2 ln(p / (1 − p))` (3.5 at p = 0.85). A join section 32 undid gets no balance (structural confidence 0.5, cost 0) and is reported; it is not left as a through-lumen for the network to flip one of its sides.

**What the balances know.** A through-lumen inside one whole vessel pairs two pieces that share one velocity: its balance is the two pieces' mass ratio over the softness, section 27's `density` test in a looser unit, and it carries no sign information. A projected crossing's two through-lumens pair arms of one vessel each: the same. **The sign step's reach is the lumens all of whose arms are measured or pinned by other lumens**, which on the real pipeline means bifurcations and confluences: two measured daughters off an unmeasured trunk give nothing, since the trunk takes whatever balances them (measured), and "joins between two whole vessels" is empty by construction, because `join_pieces` joins every through group into one vessel and the only joins between two vessels are the ones section 32 undid, which get no balance. **On a plexus of projected crossings the network does nothing**: no sign is constrained (measured: twelve vessels, two reversed, no flips, no imbalance) and a vessel with no measured piece is in the null space and gets no direction (measured: q = 0, σ 672). The plexus column is expected to be zero, and nothing in this section claims otherwise.

**The signs.** Within every lumen, every set of the measurements touching it is a move to turn those round, in closed form from the one solve's covariance (exact also with Huber weights, priors and a measurement spanning several lumens, verified). **A reading may instead be set aside**, both lumens of a crossing together (the identity for deleting a set of rows, exact also where an arm nothing else reaches makes it singular), at the cost `ln(c / (1 − c))` of its **structural** confidence c, the probability that the structure that made the lumen is right, as section 37 defines it per join and per vertex: never a goodness of fit (section 27's `reading_confidence` has a median of about 0.65 under calibrated tests and would set a fifth of the lumens aside for nothing), never 1 − ε where the structure can be wrong, and never a tie's zero: a reading whose margin is below the notebook's `TIE` has no balance and is reported as tied, so that a tie is not a free pass. The best move is taken while one lowers the total; `flip_margin` is a precision knob for the harness. **The pattern rule** (every arm measured, all in or all out) is reported and never acted on: acting on it turned a right measurement round where the lumen's reading was wrong (measured), and section 27's `in and out` test fails such a reading already.

What was measured:

| case | outcome |
|---|---|
| one wrong direction, contradicted at both ends | turned round |
| one wrong on a leaf, contradicted at one end (a confluence all in) | turned round at 0.45 |
| trunk and one daughter wrong at a bifurcation | at 0.15 the two are turned together; at 0.3 with the joins at 0.99 the join the wrong daughter passes is set aside and reported; at 0.45 one is turned and the rest left and reported; no right measurement cascades |
| two vessels ending near each other, read as one, the join at 0.8 | set aside and reported impossible; nothing turned; `blame` sends it to the graph |
| a whole vessel of two pieces through a crossing, read backwards | turned round as one at 0.3 and 0.15; at 0.45 the two 1.3σ balances do not quite pay for it |
| a crossing read with margin 1.0 against a 4-against-2 imbalance | set aside as one reading, also when a free-only arm makes the deletion identity singular |
| a daughter missing from the graph | the two-arm lumen goes to the graph; the trunk's right speed is not blamed (at a tight balance the raw residual had blamed it at 5σ) |
| a bridge from a free end whose arm is unmeasured, no scale given | the vessel it continues sets the prior's unit from a first solve; nothing crashes or pins |
| a crossing plexus, whole-vessel errors | inert by construction (no flips, no inferred directions) |
| an anastomotic grid read as the pipeline reads it (crossings), whole-vessel errors in chains, six seeds | active only at chain boundaries; mixed: −1.7 to +4.6 points at 0.45 with vessel flips 1:2 to 3:1, −8.7 to +9.4 at 0.15 (the fifth review). This is a wrong structure the network cannot set aside, and the plan does not claim it |
| an anastomotic grid read as one lumen of four (a reading section 27 never produces today), per-edge or chain errors | +0 to +4 points at 0.45, flips 2:0 to 11:1; +1 to +13 at 0.15, flips 16:2. What section 27 would have to produce for the network to help a true plexus |
| binary trees, thick level unmeasured, 15% wrong, structural confidences at their defaults | +3 to +7 points at 0.45, flips 18:5 over four seeds; near-perfect at 0.15 |
| the same trees with goodness-of-fit confidences wrongly passed as structural | 9–28 of 127 lumens set aside, gain 0 to 1 point (the fourth review); the reason the two are separated above |

**The sizes.** Huber weights on balances and misfits; the balances sized by the measurements alone for the sign step (sizing them from the solution tightened them around wrong neighbours on a plexus and turned right measurements round; measured), and robustly afterwards (a measured arm counts the smaller of its measurement and what the rest predicts for it, so that a wrong speed cannot widen its own balance; measured with a trunk read five times too fast). The studentised residual per measurement is **reported, not a detector**: on trees with a tenth of the speeds three times too fast it flags none at 0.45 and as many right speeds as wrong at 0.15 (measured). `blame` gives the rule for which of a missing arm and a wrong join an imbalance is, sends both to the graph, and lists the speeds in doubt as a report.

**The tests.** Per lumen the leave-one-out imbalance (the balance row's studentised residual; NaN where nothing but its own balance says what its edges carry); per vertex the candidate readings from one leave-one-vertex-out solve (a rank-k update of the covariance; the scores do not depend on the reading the solve enforced, verified); they enter `group_tests` as the `flux` test, **left out where NaN** (a NaN fails a partition in `partition_reading`), able to fail a reading, never choosing among readings that fit; at the notebook's softness they fail little (every four-arm vertex of a single-depth plexus passes as a crossing by flux alone, measured). No χ² ratio between readings: `compare_readings` refuses unequal balance counts, since the reading with fewer balances always fits at least as well.

**What the test can see.** With `SIGMA_FLUX` = 0.45, two through-vessels of 6 and 2 give 2.2σ against the wrong pairing and 6 and 5 give nothing; at 0.15, 6.7σ. The flux's evidence at a crossing is worth something only where the two vessels differ by more than about the balance's softness.

**Doubtful edges.** A bridge with confidence c has a prior of zero flux with sigma `c / (1 − c)` times **the flux of the vessel it continues** (the end arm's measurement; where the end is unmeasured, what a first solve without priors makes that vessel carry; else the graph's mean): scaled by the graph's mean alone, a right thick bridge was held back and a wrong capillary bridge was not (measured). A free end's rival bridges are one lumen with the end's arm, so its flux is not counted twice and a worthless rival carries nothing; `flux_at_free_ends` reports what arrives at every free end.

**Directed cycles**: strongly connected components over edges with `|q| / σ` ≥ 3 (unmeasured plexus edges sit near 0.7 and would make cycles of noise), one cycle per component, reported only. Flips, set-asides, impossible lumens and cycles are never measures of p; p comes from section 33's counts only.

**Cost.** One dense solve per fixation, twice: 3–10 s at a thousand edges and 20–180 s at two thousand across the two machines that measured it; sparse factorisation from a thousand edges on. The sign moves are in closed form; a set-aside costs one re-solve.

**Column:** directions right among inferred edges on the tree bursts. As description: the plexus (expected zero), the count of sign-constraining lumens (lumens all of whose arms are measured or pinned), set-asides and tied readings per softness, flips right / wrong, and on the real burst the agreement with `propagate_directions`, which is agreement between two inferences, not a score.

---

## 41. Confidences

Every confidence is a sum of calibrated log-likelihood terms, the prior explicit and separate: on the development bursts each term is binned and the share right in each bin plotted against the bin's value (a reliability diagram), with `told_apart` as the one number; the held-out bursts say whether it held. The bottom-up terms of section 37 are calibrated first, on section 37's graph alone. **The structural confidences of section 37** are calibrated as probabilities: P(join right) for two-arm joins from the pair's turn, width and density; P(bifurcation, not a through pair with a free end) at three-arm vertices from the far-side evidence; P(two vessels, not one lumen) at four-arm vertices from the `overlap` z now and the core z later. The histogram of section 27's margins is reported first (the share that is finite), since the margin alone is infinite at most crossings and answers none of these questions. The prior (the base rate of gaps, terminations and loops) is the generator's setting on synthetic bursts and the default of Part III on the real one; a bridge confidence of 0.8 means "the similarity is 4 : 1 for a continuation", not "right 80% of the time", until the prior is known.

The graph handed on: `vessel_digraphs[n]`, per edge the spline, `R_ridge`, `R_od` where it exists, line mass, blur, velocity ± σ, direction (measured or inferred) with `|q| / σ`, flux ± σ, the edge confidence and its terms, `drawn`, and the flags `bridge` / `twin` / `stub` / `found in residual`; per vertex position, kind, type (bifurcation, crossing, confluence, through, bend, end, dive), the partition, structural and fit confidences, `impossible`, `set aside`, the imbalance; per component with a directed cycle, one cycle. Written as `vessel_digraph_fixation_<n>.json` next to the whole-vessel files.

---

## 42. The harness rows this adds

Rows: `plain ridges, joined, cyclic graph` (curves over edge confidence); `cyclic graph, rendered, round 1 … final`; `flux as a test` (section 40's `z` in `group_tests`, beside section 33's `likelihood ratio` row); the λ sweep on the real burst. Columns: the one per section (37 `told_apart`; 36 junction recall at matched precision; 38 core-z `told_apart` by radius and blur; 39 found-and-typed-right; 40 inferred directions right on trees), and as description: bridges right / wrong, lumens fused right, flux `z` `told_apart` with the share of readings failed, flips right / wrong, sign-constraining lumens, set-asides and tied readings per softness, directed cycles, windowless edges, per-edge density RMS by radius, readings changed at step 6(d), the plexus (expected zero), the share of skeleton two-arm joins that are false, the self-consistency number per round. Flux columns upper bounds until the velocity profile is in. With sixteen held-out bursts something among sixty cells will look like an effect by chance; the one column per section is what is quoted.

---

# Part III. For the implementer

## Code units

Each unit names what it takes, what it gives, what existing code it wraps or replaces, and the test that shows it works. "Notebook" means a cell of the notebook with its own definition-cell id; "package" means `vascnet/`.

| unit | where | takes | gives | wraps / replaces | test that shows it works |
|---|---|---|---|---|---|
| `joined_ridges(strength, orientation, region)` | notebook, 37 | the saved maps | the skeleton after closing, thinning and `extend_ends`, no dropping; the pixels `extend_ends` added | `ridge_skeleton`; the joining half of `consolidate_lines` | on a synthetic burst, every piece the consolidation drops is present; the added pixels equal `extended & ~lines` |
| `ridge_lines` with sub-pixel offsets (`crest_places`, `fine_lines`) | notebook, 37 | the skeleton, strength, orientation, half-width, and the crest neurons of the pre-interaction score | chains with a fractional offset across the line per point (the neuron of the pixel's winning orientation, else the pixel's centre) | `ridge_lines` (29), `crest_neurons` (21) | the fit residual on straight synthetic vessels falls against the staircase's 0.28 px median |
| `refine_knots(chain)` | notebook, 37 | a chain | the pieces it is fitted as, each with its knot spacing: halved where the χ² in `RIDGE_PRECISION` units over `RIDGE_CORRELATION` falls by more than ln(samples) per coefficient; split at a bend of more than `FUSE_STRAIGHT` where the split pays | `spline_vessel` (29) | a synthetic corner is split at the corner; the generator's smooth vessels keep 12 px knots (at least 90% of them over eight bursts) |
| `kappa_passes(splines, density, blur)` | notebook, 37/38 | the splines, the mean density, the fixation's blur | κ or line mass per window outside the cores, converged | `profile_across` (21), new fit | on a synthetic crossing the thin vessel's κ is recovered within its error after the thick one is subtracted |
| `spline_to_graph(splines)` | notebook, 37 | the spline set | vertices, edges, lumens, cycles; the graph recomputed locally after an edit | `vessel_graph`, `contract_cores`, `fuse_split_crossings` (26) | the rules of section 37 on a join, a T, a crossing, a twin, a glancing crossing, a self-loop; the graph of section 29's splines on a synthetic burst equals `vessel_graph`'s where no rule differs |
| `edge_repeatability(edges, other fixations)` | notebook, 37 | an edge's points, the other fixations' ridges and the maps to the anchor (on the harness, section 35's second fixation) | the share found again above chance, per edge | `found_again` (18), `repeated` (29) | per-edge values average to `repeated`'s number |
| `join_provenance(graph)` | notebook, 37 | the graph, the added pixels, `graph["readings"]` | per two-arm join: skeleton / extension / undone / bridge | wrappers over `extend_ends`' output and `join_pieces` | the undone joins equal the ones `whole_vessel_flow` counted |
| `junction_cells(scores, half_width, rings)` | notebook, 36 | the pre-interaction score and the half-width | `n`, `rate`, `ring`, `plaid` maps | new | section 36's three checks |
| the "through pair + free end" partition | notebook, 27 | a three-arm vertex | one more partition with its far-side evidence | `set_partitions`, `geometric_reading` | a T whose stem dives is read as through + free end on a synthetic burst; bifurcations are unchanged |
| `render_graph(splines, lumens, blur, shape)` | notebook, 38 | the splines with κ, the lumens, the fixation's blur | the rendered density, per-lumen canvases | `vascnet.synth.render`, `_branch_od` | a synthetic burst rendered from its truth reproduces the generator's clean frame within the blur's error |
| `score_render(rendered, noise map, gains, noise constants)` | notebook, 38 | a render, the data's background-fit noise map and `contrast_gains`, the eight per-width noise constants | its 16 × 8 channels before the width competition (new code: the stack does not exist in section 20), and its ridges after the competition and `ridge_crests`, without the interactions | `contrast_layer` (19), the hypercolumns of section 20 run with fixed constants | the self-consistency number of the generator's own truth is 0.996 or better in both directions within `REPEAT_WITHIN` |
| `topology_residual(data channels, render channels, zero point)` and `rescore_window` | notebook, 38 | the two channel stacks and the zero point per channel; a window aligned to 20 px with a 110 px margin | the per-channel difference, its ridges; the difference re-scored in a window after an edit | new | a missing synthetic vessel appears as a positive ridge; a stub drawn appears as a negative one below the zero point; the windowed re-score equals the full one inside the window within 0.005 |
| `core_z(vertex, splines, density, noise)` | notebook, 38 | a vertex, its arms' κ and R, the density residual, the noise's spectrum | the matched-filter z | the `overlap` test of 27, generalised | the z-by-radius-and-blur table on rendered crossings against recorded one-lumen junctions |
| `recursion(model, data, λ, rounds)` with its moves | notebook, 39 | the initial model and the data | the model per round, J, the self-consistency number, the film | new, over the units above and `vascnet.fluxnet` | each move kind on a synthetic burst made to need it (a gap, a stub, a missing crossing, a wrong composition); J falls at every taken move |
| `solve_flux_network`, `vertex_tests`, `blame`, `flux_at_free_ends`, `directed_cycles`, `lumens_of`, `compare_readings` | package, `vascnet/fluxnet.py` | fluxes with sigmas, lumens, measurement ids, structural confidences, bridge confidences and scales | the solution (fluxes, sigmas, directions, flips, set-asides, impossible lumens, imbalances, studentised residuals), the candidate readings' z, the blame, the flux at free ends, the cycles | exists; 36 tests | `tests/test_fluxnet.py` |
| `sigma_flux_by_kind(whole_graphs)` | notebook, 40 | the whole-vessel graphs | `SIGMA_FLUX` per lumen kind with counts | new; replaces the one number of 27 | printed on the real burst; on synthetic bursts near the generator's spread |
| `Recorder`, `animate`, drawing functions per fit | package, `vascnet/watch.py` | candidate states; a drawing function | the film | exists for the flux network; drawing functions for the initialisation and the rounds are new | `tests/test_watch.py`; `scripts/watch_flux_demo.py` |
| the generator's extensions | notebook, 35 | seeds | bursts with the twelve things of section 35 and their truth | `synthetic_network`, `synthetic_burst` (28) | old seeds bit-identical; per-seed counts |
| the harness rows and columns | notebook, 42 | the readings | the table | `HARNESS_WAYS`, `score_reading`, `ablation_row` (30, 34) | the old rows reproduce section 34's numbers |

## Acceptance steps

Each step has the number it must produce and what that number decides. Nothing in a later step is built until its gate is passed.

1. **The generator** (section 35), first, because step 2's gate needs its second fixation and step 6's check needs its capillaries. Gate: old seeds bit-identical; per-seed counts.
2. **The graph with confidences** (section 37, its units; the sixth term later). Number: `told_apart` of the edge confidence between true and false pieces on the held-out bursts, on the five bottom-up terms, and the share of skeleton two-arm joins that are false. Gate: `told_apart` ≥ 0.75 and the confidence curve passing the consolidated row's point; else section 29's consolidation is kept and the plan stops here for the graph.
3. **The flux network on the existing whole-vessel graphs** (section 40, with `join_provenance` and `sigma_flux_by_kind`). Numbers: `SIGMA_FLUX` per lumen kind with counts; on the real burst the count of sign-constraining lumens, set-asides, impossible lumens, cycles and flux at free ends; on the tree bursts directions right among inferred edges against `propagate_directions`. Gate: the tree column improves; the plexus is reported as zero. This run measures the consolidated graph's shares of impossible junctions and cycles, the reference of the principles' rule for λ. λ is swept against them in step 7, when the moves it prices exist.
4. **The junction cells' three checks** (section 36) on the oriented score. Gate: the thin arm appears on its own ring at a rate above `T_LOW`, or the second reading at the thin width is added and the plan says so.
5. **`spline_to_graph` and the render-to-score pipeline** (sections 37, 38a). Numbers: the self-consistency number of the initial model on the real burst, whole-frame and within the tubes; the zero point per channel; the cost of a full re-score and of a windowed one. Gate: within the tubes the agreement reaches the data's own repeatability on the real burst (the initialisation is done); the windowed re-score equals the full one inside the window within 0.005; a move's evaluation is under a few seconds.
6. **The core z by radius and blur** (section 38b), on the generator's capillaries from step 1. Number: the z table and its `told_apart`. Gate: decisive (z ≥ 3) for R ≥ 3 at the real burst's blur; capillaries reported as they come out.
7. **The recursion round by round** on one synthetic burst made to need each move, then the harness (sections 39, 42). Numbers: J and the self-consistency number per round; found-and-typed-right per round on the held-out bursts; the films. Gate: J falls at every taken move; found-and-typed-right does not fall from one round to the next; the first principle's number does not fall below its gate after the rounds.

## Defaults for the open choices

1. **The line mass for the flux**: the kymograph's M, since it is what the speed was read on; the profile's `π κ R²` is the cross-check, and the two are reported side by side on the real burst. Revisit if they disagree by more than `SIGMA_DENSITY` on vessels above the blur.
2. **Composition where there is no reading** (a vertex of five or more arms, or one left unresolved): sum, since two vessels is the safer assumption for the residual, and step 4 can flip it.
3. **The base rate of true terminations against gaps and dives on the real burst**: 0.2 terminations per free end inside the data. This is a guess with no source in the notebook, to be replaced by the count the flux at free ends gives (a free end with flux arriving is a gap or a dive, not a termination).
4. **The flow field as a second source of edges**: not in this version; its orientation's agreement with the static one is recorded as description for the next.

## Known limits, stated once

- On a plexus of projected crossings the flux network does nothing; on a true anastomosis read as a crossing it cannot set the structure aside; the plexus column is expected to be zero.
- Wrong speeds are not detected.
- A vessel lying exactly along an existing spline with nothing emerging is not found by either residual, and neither is a thin vessel touching a thick one's wall: both are limits of the still.
- The topology residual is a working objective, not a χ² statistic: its per-channel difference is censored and its zero point on a right spline is about −1 noise unit, measured and subtracted at round 0.
- The core filter is marginal at blurred capillary crossings; geometry decides there.
- The flux's evidence between two through-vessels is worth something only where their fluxes differ by more than about the balance's softness.
- The harness's flux columns are upper bounds until the generator has a velocity profile; its halo and background columns are approximate until it runs the background fit.
- Section 36's capillary claim hangs on its first check. That check passes at blurs up to 1.5 px; at 3 px a thin crossing vessel is lost on its own ring in 3 of 8 pictures. A thin daughter leaving a thick trunk is read as a line under the largest-rate rule.
- A full hypercolumn run per move evaluation is the recursion's cost; the windowed re-score is what makes it affordable, and step 4 measures it before anything depends on it.

---

# Part IV. Appendix A: how the reviews' findings were handled

| finding | handling |
|---|---|
| R1 A1, R2 4, R3 5, R4 7 | whole-network residual; windows stop at the core; windowless edges counted and given their lumen's parameters; the core filter's reach stated by radius and blur, what it adds over `overlap` said, a check before section 39; capillary four-ways left to geometry |
| R1 A2, R2 3, R3 10, R4 6, R4 16 | the topology residual moved to the orientation layer (its correlation is the filter's footprint, known); the density residual kept for cores and mass with the noise's two-dimensional autocorrelation; measured parameters' jumps in ΔJ, λ absorbs the rest |
| R1 A3, R2 10 | removal, merge, split as moves; reading reverts on a veto |
| R1 A4, R2 1, R3 2, R4 3, R5 4 | flux tests from one leave-one-vertex-out solve, failing only; step 6 once, in the last round, two solves, readings then frozen; a closing structural round; no loop |
| R1 A5, R2 5, R3 4, R4 5, R5 3 | bridges never drawn; a prior in the unit of the vessel the bridge continues, with a fallback from a first solve; rivals one lumen; flux at free ends |
| R2 2, R3 1, R3 3, R3 8, R4 1, R4 2, R4 10, R4 12, R5 2, R5 8, R5 9 | whole-vessel measurements; structural set-aside cost, never goodness of fit, never 1 − ε where the structure can be wrong, a tie no balance; the pattern rule reported; the sign step's reach stated as lumens all of whose arms are measured or pinned, and measured on a crossing plexus (inert) and an anastomotic grid; set-asides per softness a column; a flip margin for the harness |
| R2 6, R3 13, R4 11 | the notebook's unit; `SIGMA_FLUX` per lumen kind, counts printed, population named |
| R2 7, R4 9, R5 5 | robust sizing; `blame` for one signal; wrong speeds stated as not detected |
| R2 8, R3 11 | SCC cycles at cutoff 3 |
| R2 9 | transmittance blur in renderer and generator; local canvases |
| R2 11, B7, R5 11 | λ the swept multiplier with the rule for choosing it |
| R2 12, R4 13, R5 10 | per-edge repeatability and join provenance named as new code with the wrappers given; plaid cells on the pre-interaction score |
| R2 13, B8, R3 6, R4 8 | the skeleton's pruning kept and credited; the field-length clause dropped; `told_apart` ≥ 0.75 on five terms |
| R2 14, R3 9, R4 2 | tests for whole vessels, set-asides, the prior across two orders, a crossing plexus, a missing daughter, a free-only arm, an unmeasured free end |
| R2 15, R3 12, R4 15 | cost as measured on two machines; sparse from a thousand edges |
| R2 16, B9, R3 14 | rings from 4 px, NMS per ring; the oriented-score check; per-junction `n` |
| R2 17, C5, R4 14 | prior explicit; the structural confidences calibrated as probabilities; set-aside per reading |
| R2 18, B1 | `flux as a test`; velocity profile; upper bounds |
| R3 7, R4 6, R5 6 | the density column per edge in the noise's along-line unit, cores by the core z, never pooled |
| R4 4 | undone joins recorded and given no balance; through-lumens of one vessel carry the mass ratio only |
| R5 1 | the plexus claims withdrawn; the anastomotic-grid rows restated with the readings the pipeline makes; four-arm structural confidence as P(two vessels, not one lumen) |
| R5 7 | the core filter's reach as derived by radius and blur; the spectrum measured; capillaries added to the generator; the cluster modulation named |
| B2, B3, B4, B6 | σ per fixation and line mass for thin vessels; per-lumen canvases before a transmittance blur; halos no longer fitted; flips not p |
| C3 | the flow field as a second source of edges: not in this version |
| the author's proposals after R5 | the topology residual in the orientation layer with the self-consistency number; splines as the model and `spline_to_graph`; the initialisation and its done conditions; every fit filmed |
| R6 1 | the ridges from the pre-interaction winner, as section 29 saves it; no interactions anywhere in the render's scoring |
| R6 2, R6 5, R6 13 | the 16 × 8 stack named as new code with its memory; the contrast layer and the per-width noise constants on the render; the per-channel rationale restated as the two-to-four-pixel gap and the touching thin vessel named as a limit; the zero point measured at round 0; χ²_topo a working objective |
| R6 3 | the knot refinement priced in the ridge's unit with no λ; λ chosen at acceptance step 3 |
| R6 4, R6 7 | the generator's second fixation per seed; the generator first among the steps; the core z after it |
| R6 6 | done condition 3 given the centerline shift within the tubes as its mechanism, met as a share |
| R6 8 | the crest neuron of the winning orientation, else the pixel's centre |
| R6 9 | the recorder's state per round as the changed canvases and the window's residual; the fit copies |
| R6 10 | windowed re-scoring aligned to 20 px with a 110 px margin against the frame's padding; costs measured |
| R6 11, R6 12, R6 14, R6 16 | one twin rule, `read_arms`'; k endpoints one vertex; a split body's halves inherit; the corner rationale dropped; an overlap read as one line mass |
| R6 15 | the table slips corrected; `WATCH_EVERY` defined; the terminations default marked a guess |
