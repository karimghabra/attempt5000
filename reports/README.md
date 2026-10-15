# Reports of the handoff branch

This branch carries out `docs/cyclic_graph_recursion_plan.md` on `burst_2026-09-16_15-50-52`, in the order of its acceptance steps. Read the files below from the top.

| report | what | result |
|---|---|---|
| [reproduction_20261013.md](reproduction_20261013.md) | notebook sections 1–34 on the burst | all 77 cells ran in 26.7 min; tables recorded |
| [step_1_20261013.md](step_1_20261013.md) | acceptance step 1: the generator (section 35) | **gate passed**: 32 of 32 old seeds bit-identical; counts for 64 new bursts |
| [step_2_20261013.md](step_2_20261013.md) | acceptance step 2: the graph with confidences (section 37) | **gate passed**: told_apart 0.946 (section 30's held-out bursts) and 0.951 (section 35's trees); the curves pass the consolidated point |
| [step_3_20261014.md](step_3_20261014.md) | acceptance step 3: the flux network on the whole-vessel graphs (section 40) | **gate failed**: on section 35's held-out trees the network gives 4 edges without a measured direction the right one, `propagate_directions` 18 |
| [step_4_20261014.md](step_4_20261014.md) | acceptance step 4: the junction cells' three checks (section 36) | **gate passed** at blurs of 0.8 and 1.5 px (the thin arm on its own ring in 16 of 16 pictures); at 3 px 5 of 8, and the second reading at the thin width adds nothing |
| [step_5_20261014.md](step_5_20261014.md) | acceptance step 5: `spline_to_graph` and the render-to-score pipeline (sections 37, 38a) | **gate failed**: within the tubes the render's ridges agree with the data's above the data's own repeatability (82.9 / 76.1% against 74.8%), but the windowed re-score is off by 0.119 (gate 0.005) and took 5.3 s (1.1 s on an idle machine) |
| [step_6_20261014.md](step_6_20261014.md) | acceptance step 6: the core z by radius and blur (section 38b) | **gate passed**: at 1.5 px blur, two vessels of R ≥ 3 give median z 23.8 to 58.3 (told_apart 1.00); capillaries are told apart in focus and lost by 3.5 px |
| [step_7_20261014.md](step_7_20261014.md) | acceptance step 7: the recursion round by round (sections 39, 42) | **gate failed** by one junction: J falls at every taken move and the edge confidence holds (0.942 / 0.924), but found-and-typed-right falls 25 → 24 of 169 on section 35's held-out trees; removals take out true pieces whose darkness was misread |

**Steps 4–7 continue at the user's request.** Step 3's gate failed, and the handoff says to stop there; the user then asked to complete the plan's steps. Step 3's gate stands as failed. The softness of the flux balance (`SIGMA_FLUX` × the flux through a junction, measured at 0.30–0.38) caps any direction inferred through one junction at about 1.7σ.

**All seven steps are reported.** Step 7 is built in part: its report lists what of section 39 is not. Gates passed: 1, 2, 4, 6. Failed: 3, 5, 7.

`logs/` holds the full printed text of every run, cell by cell, and `data/` holds the tables the reports quote.
