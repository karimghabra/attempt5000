# Plan: from the density burst to typed vessel junctions

This plan continues the burst notebook (sections 1–18) with sections 19–27. It follows the notes at the end of section 18:
- centerlines come from orientation-sensing neurons, not binary masks;
- lateral inhibition and a contrast layer do the thinning;
- a second, flow-sensitive field runs on frame differences;
- pruning is parametric;
- static and flow centerlines are compared and combined.

It then carries the result through to junction classification.

## Principles

- **No binary masks for structure.** Centerlines are crests of an orientation field. They are thinned by lateral inhibition and strung together in position and orientation. Masks are only for display.
- **Nothing is trained.** Kernels are fixed. Thresholds are multiples of the background noise measured on this burst. Sizes are in pixels.
- **Two fields, one mechanism.** The static field sees density; the flow field sees how density changes. Both use the same hypercolumns, the same lateral inhibition and the same stringing.
- **Every step ends with a check.** That means a picture and a number: repeatability across fixations, loose ends per 1000 px, or truth on a synthetic burst.
- **Reuse vascnet where it already does the job.** That covers kernels, orientation score, association field, orientation-space thinning, junction rules, synthetic data and metrics.

---

## 0. Small changes upstream

1. **Frame times.** `gate_burst_frames` computes `keep_indices` but does not return it. Return it as `gated_indices`, and add `frame_times = gated_indices / fps` and `fixation["times"]`. The flow field divides by the real time between kept frames, and gating leaves gaps.
2. **Data area per fixation.** `fixation_has_data[n]` is where `density_means[n]` is finite and `background_fits[n]["ok"]` holds, shrunk by `EDGE_MARGIN`.
3. **One temporal frame.** `temporal_frame(n, i) = density_fixations[n][i] - enlarge(frame_shadings[n][i])`. This is a frame's density with its moving shading taken out, positive in vessels. All flow work starts from it.
4. **vascnet accepts this notebook's density.**
   - Use `VesselPipeline.from_od(od)` with `od` = this notebook's density. The conventions match: `B - ln I`, positive in vessels.
   - Add a `valid` mask to `cortex.feedforward`. The noise level and firing threshold are measured only on valid pixels, and responses within one kernel radius of no-data are zeroed.
   - Set `radii ≈ RIDGE_WIDTHS / 0.836`. vascnet's radius is the cylinder radius; section 18's widths are Gaussian σ.

---

## Phase A — the static orientation field

### 19. Contrast layer

These are the contrast-sensing neurons the section-18 notes ask for. Their output is "how many times the local noise this pixel stands above the background".

- **Input:** `background_fits[n]["OD"]` (alternative: `density_means[n]`).
- **No data:** continue the picture smoothly from around the hole, as `ridge_strength` does, with 0 (background density) far from data.
- **Normalize:** `contrast = OD / background_fits[n]["noise"]`. Density has already divided out the illumination; this divides out the local noise. A faint vessel in a dim corner and a dark vessel in the centre are then judged on the same scale.
- **Result:** `contrast_maps[n]`, `valid_maps[n]`.
- **Check:** on quiet pixels, `contrast` is roughly zero-mean with a robust sd within 10% of 1.

### 20. Orientation hypercolumns

- **Channels:** 16 orientations × half-widths `σ_v ∈ RIDGE_WIDTHS[:8]`. Scales of 6 px or more are computed on 2× or 4× reduced grids, as section 18 does.
- **Three receptive fields per channel** (`vascnet.kernels.line_pair`, elongation 2.5): even (line), odd (edge) and front half of the even kernel (end-stopping).
- **Complex unit:** `L = [E − α|O| − β|E_front − E_back|]₊` with α = β = 1. The odd term rejects edges. The front/back term rejects junction blobs and line ends, which is what keeps spurs from forming at every junction. It was the single most important component in vascnet's ablations.
- **Scale competition:** the strongest scale wins, refined by a parabola in log σ. The response is expressed in noise units, then re-referenced to the background floor of the strongest channel.
- **Result:**
  - `orientation_scores[n]`: (16, rows, columns) in noise units.
  - `radius_maps[n]`: (16, rows, columns).
  - One fixation at a time, float16 if needed. 16 × 2048² × 4 B is about 270 MB.
- **Check:**
  - On the large vessels, the winning orientation agrees with the structure-tensor orientation.
  - The radius map agrees with the half-widths section 18 found.

### 21. Orientation-domain interaction and lateral inhibition

This is the step that makes centerlines. It applies three fixed interactions, `LATERAL_STEPS` (3–5) times, each as a relaxed update `U ← (1 − η)·U + η·f(U)`.

1. **Tuning in orientation** ("a Gaussian in the orientation domain with side lobes suppressed"). Convolve each pixel's 16-vector over θ with a difference of Gaussians: centre σ_θ ≈ 1 bin, surround ≈ 3 bins, circular. This sharpens tuning and silences blobs that drive every orientation.
2. **Lateral inhibition across the vessel.** Channel θ is divided by its own activity on the flanks: `U_θ ← U_θ / (1 + k · (flank_θ ∗ U_θ))`. The flank kernel is two elongated Gaussian lobes at ±(1–2)·radius across the vessel. Iterating pulls a ridge profile in towards its crest and suppresses parallel side ridges.
3. **Facilitation along the vessel, modulatory only.** `U_θ ← U_θ · (1 + a·F/(h + F))`, with F the co-circular association field (`vascnet.cortex.AssociationField`). Continuity is strengthened, but nothing is drawn where there was no input. Additive "completion" made vascnet worse, so it stays off.

**Protect crossings.** Compute a co-occurrence signal `X = Σ √(U_θ · U_θ')` over channel pairs at least 3 bins apart, the analogue of V1's "plaid" cells. Where X is high, weaken the cross-orientation part of interaction 1. Otherwise a thin vessel crossing a thick one is erased inside the overlap, which turns the crossing into two T's.

**Readout:**
- **Crest points:** U_θ is largest across its own normal (orientation-space thinning, as in `vascnet.centerline.orientation_nms`).
- **Thresholds:** above `T_LOW`, and 8-connected within the channel to a point above `T_HIGH`.
- **Each crest point carries:** (x, y, θ, strength, radius).

**Check:**
- Cross-section profiles of 5 vessels before and after: the width at half maximum should fall towards 1–2 px.
- Count the parallel duplicate crests.

### 22. String the centerlines in position and orientation, without a skeleton

**Stringing:** from every crest point, step one pixel along ±θ. Accept the crest point found within 1 px of the predicted position whose orientation is within 1 bin, so a line may turn by up to one bin per pixel. Chains of accepted links are centerlines: polylines carrying θ, radius and strength at every point.

**Crossings:** two chains pass through the same pixel in different channels and stay two chains. A skeleton cannot do this.

**Bridging gaps:** join two chain ends that are at most `GAP_MAX`·radius apart, if all of these hold:
- they can be joined by a smooth, monotonic curve turning no more than 90° (Kellman–Shipley relatability);
- the oriented evidence along the bridge exceeds `T_LOW / 2`;
- for an end that abuts a wider vessel, the bridge is allowed only when the end sits flush against that vessel's wall.

**Junction candidates:**
- chain ends that come within (radius + 2) px of another chain;
- points where chains of different orientation meet.

**Parametric pruning:**
- **Pruned:** chains shorter than `L_MIN` or with mean strength below `Z_MIN`.
- **Sweep:** `L_MIN ∈ {5, 10, 20, 40}` px × `Z_MIN ∈ {2, 3, 4, 5}` × `T_HIGH ∈ {3, 4.5, 6}`.
- **Recorded for every setting:**
  - total length, segments and junction candidates;
  - loose ends per 1000 px;
  - share of length in the largest network;
  - found again in the other fixations (`found_again` from section 18).

  Plot each against the settings and pick a setting on a plateau, where small changes don't move the results. Don't pick the best-looking one.
- **Result:** `static_chains[n]` (list of polylines with per-point θ, radius, strength) and `pruning_table`.

**Check:** add these chains as a row of section 18's `graph_table`. Expect fewer loose ends than the mask-based graphs, and crossings that come out as two chains.

---

## Phase B — the flow field

### 23. Is the blood's motion visible at this frame rate?

Settle this before building velocity. On 20–50 well-sampled points along the largest static chains:

- **r₁:** correlation between consecutive kept frames of `temporal_frame` minus its mean over time, at the same pixel.
- **Travelling correlation:** correlation between the line mass at chain position s in frame t and at s + Δ in frame t + 1, for Δ from −40 to +40 px along the chain. (Line mass is density integrated across the vessel; see 25a.)
  - A peak away from Δ = 0 means the pattern is visible flow moving Δ px per frame.
  - A flat curve near 0 means the pattern decorrelates within one frame, so tracking is aliased at this frame rate.
- **Report** by vessel width class. Also count the frame pairs split by gated-out frames. Derivative features only use pairs with `Δt ≤ MAX_GAP_FRAMES / fps`.

**Decision:**
- **Peaks found:** build velocity from kymographs (25b) for those width classes.
- **No peaks:**
  - Detect flow from decorrelation (section 24).
  - Take timing from slow signals (25c).
  - The flux test in phase D then uses slow or pulsatile flux only. Record the frame rate that would be needed, `fps ≥ v / (½ · pattern correlation length)`, for a later faster burst.

### 24. Flow-sensitive hypercolumns

This is the second neural field from the section-18 notes.

- **Input per kept pair (t, t+1) in a fixation:** `D = (temporal_frame(t+1) − temporal_frame(t)) / Δt`. Holes are filled with 0, and D is expressed in its own noise units, measured on quiet pixels.
- **Same hypercolumns as 19–21, with one difference:** D changes sign as red-cell clusters pass, so use the phase-invariant energy `E² + O²` of the even/odd pair (a complex cell), not the signed even response.
- **Don't store per-frame hypercolumns.** 16 × frames × rows × columns is tens of GB. Stream instead, and keep per fixation:
  - the time-averaged energy per orientation, `flow_energy[n]` (16, rows, columns), optionally lucky-weighted (25d);
  - the per-frame values sampled along static chains only, for section 25.
- **Then the same pipeline:** orientation interaction, lateral inhibition, stringing and parametric pruning, giving `flow_chains[n]`.
- **Relation to section 15:** the smear is the unoriented version of `flow_energy`. This field adds orientation and per-frame timing.
- **Check:**
  - `flow_energy` correlates with `cleaned_smears`.
  - On quiet pixels it sits at the noise level.
  - Where it is strong, its orientation agrees with the static orientation, which separates flow from leftover misregistration. Edge jitter moves across vessels, not along them.

### 25. Temporal signals per segment

**a. Line mass `M_k(s, t)`.** For static chain k, at each point s, integrate `temporal_frame` across the vessel (normal to θ, over ±`PAINT_WIDTH`·radius). The result is a kymograph image (s × t).
- Integrated density is roughly unchanged by defocus.
- In absorption imaging it is proportional to hemoglobin per unit length. Red cells are conserved at every junction, so this is the quantity to conserve.

**b. Velocity**, only where section 23 found visible flow.
- **Kymograph streaks:** red-cell streaks are lines in the kymograph, so run the static hypercolumn bank on it. The winning orientation φ gives speed `v = tan φ · fps` (px/s), and its sign the direction. Confidence is the streak's strength in noise units.
- **Cross-check with the 1-D continuity equation:** `v = −Σ ∂ₜM ∂ₛM / Σ (∂ₛM)²` over a window.

**c. Slow signals**, always computed.
- **Heart rate:** the spectral peak of the global mean density between 0.7 and 2.5 Hz.
- **Pulsatility per segment:** amplitude and phase of M at the heart rate and its first harmonic.
- **Vasomotion:** the band below 0.2 Hz, if the burst is long enough.
- Phase continuity along a vessel and across junction arms is a same-vessel cue. Arteries and veins separate by pulse phase and amplitude.

**d. Focus trace and lucky windows.**
- **Apparent half-width:** per chain, `w_k(t)` from scale selection on a 3-frame median, which averages out passing cells.
- **Relative blur:** `b_k(t) = √(w_k(t)² − min_t w_k²)`. A vessel's width is constant, so any change in apparent width is defocus.
- **Weights:** for velocity and flux, soft weights from `b_k(t)` over runs of at least `LUCKY_RUN` consecutive frames.
- **Co-focus:** arms that sharpen and blur together share a depth, which is a grouping cue.
- **Depth order:** if focus drifts monotonically over the burst, each chain's time of best focus orders the chains in depth.

**Result:** `segment_signals[n][k]` = dict(`M`, `times`, `velocity`, `velocity_confidence`, `direction`, `pulse_amplitude`, `pulse_phase`, `blur_trace`, `best_focus_time`).

---

## Phase C — one field and one graph (section 26)

**Agreement view** (the combination the section-18 notes ask for). Compare the static and flow orientation scores pixel by pixel, at the same orientation:

| Category | Meaning |
|---|---|
| both | perfused vessel |
| static only | no flow seen: not perfused, or flow too fast or too slow to see |
| flow only | faint or defocused vessel that the static field missed |

Show the three in colour, with `static_chains`, `flow_chains` and their intersection overlaid.

**Building the graph:**
- **Combined field:** `U = max(U_static, U_flow)` per (x, y, θ), both in noise units, restrung as in section 22, giving `vessel_chains[n]`.
- **Graph:** nodes are chain ends and chain meetings; edges are the chain pieces between them.
- **Fuse split crossings.** A crossing often appears as two 3-way nodes joined by a short piece. Fuse them when they are closer than `max(3, 2.5 · W_thick / sin α)` and their outer arms pair straight across the piece. (W_thick is the thicker vessel's width; α is the crossing angle.)
- **Read the arms on a ring outside the junction core.** Inner radius is at least half the thick vessel's width plus 2 px; outer radius is several widths. Each arm records:
  - direction and blur-corrected width;
  - mean density and line mass;
  - velocity and direction (if measured);
  - pulse phase and blur trace.

  Measuring inside the core roughly doubles the angle error.

---

## Phase D — junction reading (section 27)

**D1. Baseline.** vascnet's geometric rules:
- arms paired by straightness, with a relative margin;
- T test with amodal completion on the far side;
- Murray ratio;
- density additivity at the overlap;
- relocation to where the arm axes meet.

**D2. Hypotheses are partitions of the arms into connected groups** (Estrada et al., TPAMI 2015, generalized).
- **With flow directions:** each group needs at least one inflow and one outflow. Merges are allowed, because venules have real confluences.
- **Without directions:** every partition with at least two arms per group.
- **Example:** the 5-arm crossover-plus-bifurcation has six two-group hypotheses plus "unresolved".

**D3. Score each partition** as a sum of log terms, each in noise units:
- continuation within groups (turning angle, relatability);
- width and Murray consistency;
- blur continuity (co-focus) and pulse-phase continuity;
- density additivity at the overlap (two layers mean the densities add);
- flux balance per group: `Σ_in v·M̄ − Σ_out v·M̄`, time-resolved with transit lags when velocity is visible;
- temporal fingerprint: lagged correlation of M within groups, compared with across groups.

**Decision rule:**
- Prefer the most specific hypothesis that fits. A finer partition wins when it passes its tests, because the single "unresolved" group can never fail.
- Report "ambiguous" when two hypotheses tie within noise.
- Choose on even frames and verify on odd frames, so the scoring cannot talk itself into a balance.

**D4. Network consistency.**
- Orient edges by measured flow direction. With directions known, each junction's decision is independent of the others (Estrada's linear-time case).
- Edges without a reliable direction (stagnant or reversing flow) form small clusters, searched by flipping edge directions.

**Result:**
- `junctions[n]`: type, partition, confidence, arms, features.
- A JSON export in the shape of `Result.annotations()`.
- An overlay gallery ordered by confidence, for manual review.

---

## Phase E — validation (alongside every phase)

**E1. Synthetic bursts.** Extend `vascnet.synth` with:
- discrete red-cell clusters advected through the network, with flux conserved at splits and merges;
- a depth per tree, and a focal plane that drifts during the burst;
- a 12-bit camera (saturation 4095), shot and read noise;
- the burst's frame rate and its gating gaps.

Truth: centerlines, junction types and partitions, velocities, depths.

**E2. Metrics:**
- centerline F1;
- junction detection F1;
- crossing recall and balanced type accuracy. Crossings are 20–25% of junctions, so plain accuracy flatters any classifier that defaults to "bifurcation".
- Estrada's parent and flow similarity;
- velocity error;
- the distribution of flux residuals.

**E3. Real burst, without truth:**
- repeatability across fixations;
- agreement between the static and flow fields;
- flux balance at clean Y junctions. It should centre on 0, and its spread is the flux noise used in D3.
- share of junctions resolved vs ambiguous;
- a manual spot check of the gallery.

---

## Order of work

| # | Build | Done when |
|---|---|---|
| 1 | Upstream changes (0) | `frame_times` exist; `temporal_frame` reproduces section 14's right-hand video |
| 2 | Sections 19–20 | orientation scores and radius maps agree with section 18's best graph |
| 3 | Sections 21–22 | the chains are found again across fixations at least as well as section 18's best picture, with fewer loose ends per 1000 px; a pruning plateau is chosen |
| 4 | Section 23 | decision recorded: is flow visible, per width class, at this frame rate |
| 5 | E1 synthetic bursts | truth available for every later step |
| 6 | Section 24 | `flow_chains` and the agreement view; flow energy at the noise level on quiet pixels |
| 7 | Section 25 | heart rate found; per-segment pulse phase, blur traces and (if feasible) velocities |
| 8 | Section 26 | one graph per fixation; split crossings fused |
| 9 | Section 27 | junction table with partitions and confidence; on synthetic bursts, crossing recall above the D1 baseline |

## Starting values

| Name | Value | Meaning |
|---|---|---|
| `N_ORIENT` | 16 | orientations per hypercolumn |
| `ELONGATION` | 2.5 | receptive-field length / width |
| `EDGE_SUPPRESSION`, `END_STOPPING` | 1.0, 1.0 | α, β of the complex unit |
| `THETA_CENTRE`, `THETA_SURROUND` | 1, 3 bins | orientation-domain difference of Gaussians |
| `FLANK_OFFSET` | 1–2 × radius | position of the inhibitory flanks |
| `LATERAL_STEPS`, `η` | 4, 0.7 | iterations and relaxation of section 21 |
| `MODULATION`, `HALF_SAT` | 0.5, 3 | multiplicative facilitation |
| `T_LOW`, `T_HIGH` | 2.0, 4.5 | crest thresholds in noise units |
| `GAP_MAX` | 3 × radius | longest bridge between chain ends |
| `L_MIN`, `Z_MIN` | swept | pruning |
| `MAX_GAP_FRAMES` | 1 | kept-frame pairs used for differences |
| `LUCKY_RUN` | 3 frames | shortest run for velocity |

## Risks and how each is caught

- **The frame rate may be too low to see red cells move.** Section 23 measures this first. Without visible motion, flow is detected by decorrelation, and timing comes from pulsatility and focus. Flux-based junction resolution then needs a faster burst.
- **Memory and time at full resolution.**
  - Wide scales run on reduced grids.
  - Flow statistics are streamed.
  - Per-frame signals are sampled only along chains.
  - vascnet's cached FFT bank is reused across frames.
- **Shading, flicker or misregistration can pass for flow.**
  - The flow field starts from `temporal_frame`, which has the shading removed.
  - Noise is measured on quiet pixels.
  - Flow must run along the static orientation, which the agreement view checks.
- **Holes from glare, saturation and frame edges.** Valid masks are carried through every layer, and no response is reported within a kernel radius of a hole.
- **A thin vessel can vanish inside a thick one.** Co-occurrence protection in section 21, and a dedicated check on synthetic crossings by width ratio and angle.
