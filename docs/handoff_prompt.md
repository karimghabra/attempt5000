# Prompt for a fresh machine

Paste everything below the line into a new session on a machine whose working directory holds only the burst folder.

---

You are starting from a clean machine. The working directory contains one folder, the burst: a folder of TIFF frames with a `manifest.json`. Nothing else is here. Your job is to set up the repository, reproduce the notebook's results on this burst, and then carry out the plan in `docs/cyclic_graph_recursion_plan.md` in the order its acceptance steps give, committing as you go.

## 1. Get the code

```
git clone --branch claude/burst-pipeline-vessel-motion-tnsxep https://github.com/karimghabra/attempt5000.git
cd attempt5000
git checkout -b handoff/<your-name>-$(date +%Y%m%d)
```

Work on that new branch. Never push to `claude/burst-pipeline-vessel-motion-tnsxep` or to `main`. Commit small, with messages that say what changed and why; push after every step below with `git push -u origin <branch>`. Do not open a pull request unless asked.

## 2. Set up Python

Python 3.10 or newer.

```
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip install opencv-python-headless matplotlib psutil pillow nbformat jupytext jupyter ipykernel
python -m pytest tests -q
```

All tests must pass before anything else (39 of the plan's, 17 of the package's). If one fails, fix the environment, not the test, and say so in your first commit message.

## 3. Put the burst where the notebook looks

The notebook (section 1) takes the first folder named `burst_*` next to it or in any parent folder. The burst folder sits one level above the repository, so it is found as long as its name starts with `burst_`. If it does not, do not rename or move the data: make a symbolic link next to it, `ln -s <folder> burst_<folder>`, and say so. The frame rate comes from the folder's `manifest.json` (`capture.effective_fps`); if the file is missing the notebook falls back to 15 fps, which is wrong for most recordings, so check.

## 4. Reproduce the notebook, sections 1 to 34

The notebook is `notebooks/development copy 2.ipynb`; `notebooks/development copy 2.py` is the same in jupytext percent format. Run it from inside `notebooks/`, start to finish, with no cell skipped:

```
cd notebooks
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 "development copy 2.ipynb"
```

This takes a long time (hypercolumns on every frame of every fixation, the synthetic harness on 32 bursts) and writes, next to the notebook, `vessel_pictures/` (pictures, JSON, CSV), `section_29_inputs/` (about 1.5 GB: the saved maps, scores and frames that everything after section 29 reads) and cached previews. Keep them; do not commit them (add them to `.gitignore` if they are not already ignored). Record in a file `reports/reproduction_<date>.md`: the burst's shape, frame rate and fixations; the printed result lines of sections 29, 31, 32, 33 and 34 (the ablation table's held-out rows above all); and the wall-clock time. Commit that file. If a cell fails, fix the cause, note it in the report, and rerun from the failing section (section 29's "fresh kernel" cell can restart from the saved inputs).

Do not change any constant of sections 1 to 34 and do not change any code that section 34's numbers depend on: the plan's first rule for the generator is that the old seeds reproduce to the digit.

## 5. Read, in this order

1. `docs/cyclic_graph_recursion_plan.md`, all of it. Part I says what the plan claims and what it builds on; Part II is sections 35 to 42; Part III is for you: the code units with what each takes, gives, wraps and the test that shows it works, the acceptance steps with their gates, the defaults for the open choices, and the known limits.
2. `docs/cyclic_graph_recursion_plan_review_6.md`, then the earlier reviews as you need them. They are the record of what was tried and why it was changed; when a choice in the plan looks odd, the reason is usually in a review.
3. `vascnet/fluxnet.py`, `tests/test_fluxnet.py`, `vascnet/watch.py`, `scripts/watch_flux_demo.py`: the one section of the plan that is already code, and the way every fit is filmed.
4. In the notebook: sections 19 to 22 (contrast layer, hypercolumns, interactions, chains), 26 and 27 (the graph and the junction readings), 29 (the ridges, splines and saved inputs), 30 and 34 (the harness), 32 and 33 (whole vessels, directions).

## 6. Then work the acceptance steps, in order

Part III of the plan lists seven steps. Each has the number it must produce and the gate that number has to pass; nothing in a later step is built until its gate is passed, and the first gate decides whether the rest is built at all. For each step:

- add the new code as notebook cells after section 34, each definition cell with its own id (section 29's fresh-kernel cell runs definition cells by id; follow `DEFINITION_CELLS` there), or as package code with tests where the plan puts it in `vascnet/`;
- write the tests the plan names for each code unit before the unit is used;
- run the step on one synthetic burst first, then the harness, then the real burst, as the step says;
- film every fit with `vascnet.watch` and keep the GIFs under `vessel_pictures/fit/`;
- record the step's number in `reports/step_<n>_<date>.md` with the gate's verdict, and commit.

Where the plan's text and the code disagree, the code of sections 1 to 34 is the truth and the plan is wrong: note it in the report, fix the plan's text in the same commit, and go on. Where a step's gate fails, stop at that step, write what the number was and what you think it means, and push; do not go on to the next step and do not change the gate.

The four open choices in Part III have defaults; use them, and say in the report where a default mattered.

## 7. How to report

Every push should leave the branch in a state where someone can read `reports/` from the top and know: what was reproduced, which step you are on, its number, and what is blocking if anything. Short reports; numbers in tables; no claim without the cell or test that produced it.
