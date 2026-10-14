# Reproduction of sections 1 to 34, 2026-10-13

**Result: all 77 code cells ran in order, none skipped, none failed. Wall-clock time 26.7 min (1603 s), from 21:17:34 to 21:44:17.** Every number below was copied from the run's log, [`logs/reproduction_20261013.log`](logs/reproduction_20261013.log), which holds the printed text and time of every cell. The ablation table is also kept as [`data/ablation_reproduction_20261013.csv`](data/ablation_reproduction_20261013.csv).

## How it was run

| | |
|---|---|
| machine | Windows 11 on ARM64 (Snapdragon X Elite, 12 logical cores), 15.6 GB RAM, shared with other work while running |
| Python | 3.13.14, an x64 build under emulation, in a venv outside the repository |
| packages | numpy 2.5.4, scipy 1.18.1, scikit-image 0.26.0, opencv 5.0.0, matplotlib 3.11.2, pillow 12.3.0, nbclient 0.11.0 |
| tests before the run | 56 passed (39 of the plan's, 17 of the package's) |
| command | `python scripts/run_notebook.py "notebooks/development copy 2.ipynb" reports/logs/reproduction_20261013.log` |
| memory | 9.66 GB at the most (the notebook's own `memory_note`) |

`scripts/run_notebook.py` does what `jupyter nbconvert --execute --inplace --ExecutePreprocessor.timeout=-1` does: the same nbclient, one kernel, the notebook's folder as the working directory, no timeout. It also writes each cell's printed text to the log as soon as the cell ends.

The executed notebook is 578 MB because its preview GIFs are embedded, so it is not committed. It is kept at `notebooks/executed/development copy 2.reproduction_20261013.ipynb`, and `notebooks/executed/` is now in `.gitignore`. The tracked notebook still has no outputs, as before. The run also wrote `notebooks/section_29_inputs/` (19 files, 1.49 GB), `notebooks/vessel_pictures/` (77 MB) and the cached previews; all of these are ignored by git.

One warning was printed. In cell `e5a0d93a` (section 18), scikit-image 0.26 says that `remove_small_objects(min_size=...)` is deprecated. The parameter still behaves as before, and nothing was changed.

## The burst

| | |
|---|---|
| folder | `burst_2026-09-16_15-50-52`, one level above the repository. Its name starts with `burst_`, so no link was needed |
| frames | 140 TIFFs, shape (140, 1200, 1920), uint16 (12-bit) |
| frame rate | 32.236 fps, from `manifest.json` (`capture.effective_fps`), not the 15 fps fallback |
| gate (section 3) | 131 kept, 9 dropped: 68, 79, 87, 92, 98, 101, 113, 120, 121 |
| motion (section 4) | frame-to-frame shift: median 3.54 px, largest 112.90 px; the great-shift limit of 35.4 px gives 2 fixations |
| fixation 0 | frames 0–112 (113 frames, 0.000–3.692 s), reference frame 44, 105 of 112 neighbouring pairs one frame apart |
| fixation 1 | frames 113–130 (18 frames, 3.785–4.312 s), reference frame 121, 17 of 17 pairs one frame apart |
| fixation map | 1 → 0 kept as a homography with a 0.66 px misfit, above the 0.5 px tolerance (printed by section 7) |
| blur (section 31) | 1.50 px in both fixations |
| pattern length (section 23) | 7.0 px, so speeds up to 113 px/s can be followed |

## Section 29: vessels from the static score

| centerlines (anchor fixation) | length | lines | loose ends | per 1000 px | found again | by chance | above chance | with flow data |
|---|---|---|---|---|---|---|---|---|
| ridges of the static score | 60240 | 860 | 1013 | 16.8 | 79% | 18% | 74% | 55% |
| the same, consolidated (taken on) | 59882 | 1459 | 185 | 3.1 | 78% | 18% | 73% | 54% |
| section 22's chains | 112583 | 784 | 593 | 5.3 | 78% | 26% | 71% | 55% |
| ridges of the flow score inside the mask | 24702 | 304 | 582 | 23.6 | 64% | 5% | 62% | 100% |

- The mask covers 26% and 25% of the field. Half of it has flow data (50% in fixation 0, 47% in fixation 1).
- Splines: fixation 0 has 1318 vessels and fixation 1 has 1145. The points lie 0.26 px from their spline (median rms). A clear direction of flow was read on 391 and 389 vessels, with median speeds of 111 and 143 px/s.
- Junctions in fixation 0: 410 junctions on 1327 edges, read as 265 bifurcations, 91 unresolved and 51 crossings. Flow confirms 147, speaks against 215 and changes 1.
- Junctions in fixation 1: 362 junctions on 1166 edges, read as 247 bifurcations, 75 unresolved and 37 crossings. Flow confirms 284, speaks against 42 and changes 1.

## Section 31: the tube rule, swept on the real burst

| centerlines | length | pieces | loose ends per 1000 px | found again above chance |
|---|---|---|---|---|
| plain ridges | 60240 | 860 | 16.8 | 74% |
| consolidated, by length | 59882 | 1459 | 3.1 | 73% |
| consolidated, tube rule, 0.00 field lengths | 77999 | 3472 | 16.7 | 72% |
| consolidated, tube rule, 1.00 field lengths | 68246 | 1978 | 8.6 | 71% |
| consolidated, tube rule, 2.00 field lengths | 64093 | 1728 | 6.3 | 71% |
| consolidated, tube rule, 3.00 field lengths | 61142 | 1571 | 4.3 | 70% |

The settings 0.25, 0.5, 0.75 and 1.5 are in the log.

## Section 32: speed along whole vessels (real burst)

| | fixation 0 | fixation 1 |
|---|---|---|
| pieces → whole vessels | 1327 → 836 | 1166 → 757 |
| joins undone (sides against each other) | 13 | 10 |
| clear speed: vessels (share of length) | 220 (40%) | 156 (32%) |
| median speed, median sigma | 51 px/s, 36 px/s | 64 px/s, 19 px/s |
| above the 113 px/s limit, flagged | 145 (32%) | 248 (53%) |
| sigmas widened by | 4.4 (54 window pairs) | 1.2 (42 pairs) |
| window pairs agreeing within 2 sigma after widening | 83% | 69% |

## Section 33: reliability and evidence (real burst)

| | fixation 0 | fixation 1 |
|---|---|---|
| three-arm junctions all in or all out | 2 of 6, p = 0.50 | 0 of 4, p = 0.89 |
| sides of a junction a vessel passes that agree | 7 of 7, p = 0.97 | 5 of 5, p = 0.96 |
| flow as a likelihood ratio | 293 no data, 61 no other reading, 53 fits both, 3 changes | 293 no data, 39 no other reading, 22 confirms, 5 against, 2 changes, 1 fits both |
| directions impossible under their reading | 2 | 2 |
| direction rule names the trunk | at 4 of 265 bifurcations; widest arm at 2 (50%) | at 4 of 247; widest arm at 0 (0%) |

## Section 34: the ablation table

**Held out** (seeds 1001–1016; 92 true junctions asked for). Shares are in percent, with 95% intervals over bursts.

| row | centerline precision | centerline recall | junction precision | junction recall | type right, of found | found and typed right |
|---|---|---|---|---|---|---|
| chains of sections 19 to 27 | 70 (66–75) | 99 (98–99) | 22 (17–28) | 92 (88–97) | 44 (33–56) | 40 (31–51) |
| plain ridges | 95 (92–97) | 93 (90–95) | 84 (72–93) | 77 (70–84) | 70 (58–82) | 54 (43–66) |
| consolidated, keeping short pieces | 93 (91–96) | 92 (89–94) | 72 (57–87) | 63 (55–71) | 66 (50–79) | 41 (29–53) |
| consolidated, dropping short pieces | 94 (92–96) | 88 (84–91) | 91 (82–98) | 45 (37–52) | 54 (34–72) | 24 (14–33) |
| tube rule on plain ridges | 99 (99–99) | 80 (71–87) | 94 (82–100) | 49 (33–63) | 58 (42–73) | 28 (16–41) |
| consolidated, tube rule | 95 (92–97) | 85 (81–90) | 71 (57–85) | 66 (54–78) | 64 (52–76) | 42 (31–56) |
| plain ridges, whole vessels | 95 (92–97) | 93 (90–95) | 84 (72–93) | 77 (70–84) | 70 (58–82) | 54 (43–66) |
| consolidated, whole vessels | 93 (91–96) | 92 (89–94) | 72 (57–87) | 63 (55–71) | 66 (50–79) | 41 (29–53) |
| plain ridges, whole vessels, likelihood ratio | 95 (92–97) | 93 (90–95) | 84 (72–93) | 77 (70–84) | 70 (58–82) | 54 (43–66) |
| consolidated, whole vessels, likelihood ratio | 93 (91–96) | 92 (89–94) | 72 (57–87) | 63 (55–71) | 66 (50–79) | 41 (29–53) |

**Development** (seeds 1–16; 97 true junctions asked for).

| row | centerline precision | centerline recall | junction precision | junction recall | type right, of found | found and typed right |
|---|---|---|---|---|---|---|
| chains of sections 19 to 27 | 70 (65–76) | 98 (98–99) | 21 (16–28) | 95 (91–98) | 51 (38–64) | 48 (35–61) |
| plain ridges | 95 (92–97) | 94 (92–96) | 76 (67–85) | 73 (61–84) | 73 (60–83) | 54 (39–68) |
| consolidated, keeping short pieces | 94 (92–96) | 92 (90–94) | 75 (64–85) | 64 (58–70) | 65 (54–75) | 41 (33–49) |
| consolidated, dropping short pieces | 95 (93–97) | 89 (86–92) | 81 (68–92) | 39 (30–48) | 71 (55–85) | 28 (20–35) |
| tube rule on plain ridges | 99 (99–99) | 81 (73–88) | 87 (74–96) | 47 (32–61) | 65 (46–81) | 31 (17–45) |
| consolidated, tube rule | 96 (94–97) | 88 (83–92) | 77 (67–86) | 71 (65–78) | 68 (56–77) | 48 (38–58) |

The "whole vessels" and "likelihood ratio" rows of the development set are the same as their base rows, as they are for the held-out set.

**What section 34 shows on the held-out set** for sections 32 and 33:

| | plain ridges, whole vessels | consolidated, whole vessels |
|---|---|---|
| whole-vessel speed: median error, 90th percentile | 4%, 19% | 6%, 36% |
| true speed within 2 sigma | 74% (63–83; n = 121) | 71% (57–85; n = 112) |
| whole-vessel direction right | 100% (n = 121) | 99% (97–100; n = 112) |
| p without truth: three-arm junctions; sides | 0.85 (2 of 19 all in or out); 0.99 (23 of 23) | 0.98 (0 of 20); 0.98 (15 of 15) |
| p with truth: arms; whole vessels | 0.96 of 151; 1.00 of 121 | 0.96 of 129; 0.99 of 112 |
| junction confidence told apart, geometry / with flow | 0.81 / 0.79 (50 right, 21 wrong) | 0.53 / 0.57 (38 right, 20 wrong) |
| trunk: widest; direction; both | 73% (n = 26); 100% (n = 10); 88% (n = 26) | 52% (n = 27); 92% (n = 12); 70% (n = 27) |

The section kept 2879 junction rows, 3576 arm rows and 976 whole-vessel rows in `notebooks/vessel_pictures/harness/`.

## What can and cannot be said

- The repository has no earlier copy of these tables (the plan quotes no section 34 numbers), so this run cannot be checked against a previous one. Step 1 instead re-runs section 34 from a fresh kernel and compares its `ablation.csv` with this one; see `step_1_20261013.md`.
- On the held-out bursts, "plain ridges" is the best row for junction recall (77%) and for found-and-typed-right (54%). The consolidated row, which is the reference point of the plan's first gate (acceptance step 2), is at 72% junction precision and 63% junction recall.
