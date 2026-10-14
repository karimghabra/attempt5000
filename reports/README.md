# Reports of the handoff branch

This branch carries out `docs/cyclic_graph_recursion_plan.md` on `burst_2026-09-16_15-50-52`, in the order of its acceptance steps. Read the files below from the top.

| report | what | result |
|---|---|---|
| [reproduction_20261013.md](reproduction_20261013.md) | notebook sections 1–34 on the burst | all 77 cells ran in 26.7 min; tables recorded |
| [step_1_20261013.md](step_1_20261013.md) | acceptance step 1: the generator (section 35) | **gate passed**: 32 of 32 old seeds bit-identical; counts for 64 new bursts |
| [step_2_20261013.md](step_2_20261013.md) | acceptance step 2: the graph with confidences (section 37) | **gate passed**: told_apart 0.946 (section 30's held-out bursts) and 0.951 (section 35's trees); the curves pass the consolidated point |
| [step_3_20261014.md](step_3_20261014.md) | acceptance step 3: the flux network on the whole-vessel graphs (section 40) | **gate failed**: on section 35's held-out trees the network gives 4 edges without a measured direction the right one, `propagate_directions` 18 |

**Stopped at step 3.** Its gate failed, so step 4 was not built.

**What is blocking:** a decision by the plan's author. The balance's softness, `SIGMA_FLUX` × the flux through a junction, measured at 0.30–0.38 on these bursts, caps the confidence of any direction inferred through one junction at about 1.7σ. So the network cannot beat `propagate_directions`' sign rule at the 2σ cutoff. At lower cutoffs it gives more edges a direction, at a lower share right. The report lists options; the gate was not changed.

`logs/` holds the full printed text of every run, cell by cell, and `data/` holds the tables the reports quote.
