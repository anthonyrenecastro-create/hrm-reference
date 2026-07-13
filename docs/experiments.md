# Experiments

Seven canonical synthetic tasks probe distinct contracts:

| Task | Question | Appropriate controls |
|---|---|---|
| Spatial reconstruction | Can a masked distributed field reconstruct a known pattern? | persistence, graph diffusion, reaction–diffusion |
| Sequence prediction | Does predictive state improve next-step error? | persistence, small recurrent model |
| Associative recall | Are stored associations recovered under corrupted cues? | cosine nearest associate |
| Perturbation recovery | Does the system recover field fidelity after declared mixed damage? | no-repair and fixed-structure variants |
| Resource adaptation | Does active width change real work under a fixed budget? | equal-budget fixed allocation |
| Topology usefulness | Does structural adaptation help when graph structure matters? | fixed topology and fixed graph |
| Topology irrelevance | Does adaptation avoid unjustified benefit on irrelevant structure? | fixed topology and simpler baselines |

`hrm experiment` preserves each seed as a complete run bundle. `hrm ablate` reuses the task,
input-generation rule, seed set, step count, and budget, then reports per-seed trajectory divergence,
outcome, cost ratio, bootstrap interval, robust effect, sign test, Wilcoxon test, Holm correction, and
Benjamini–Hochberg correction.

Matched divergence uses a digest of the ledger-authoritative numerical carrier. Complete state
digests include configuration and run metadata and must never be used as behavioral divergence
evidence.

Statistical results are interpretation-disabled when ledgers fail, the mechanism never activates,
matched trajectories do not diverge, or cost/capacity differs without disclosure. A small p-value is
not implementation evidence and synthetic support is not external validation.

The canonical 30-seed configuration is intentionally not committed as if it were a completed study.
Reference artifacts use short smoke/calibration runs and label their limited evidentiary scope.
