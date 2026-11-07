# Pre-specified analysis plan (fixed 2026-11-07, before the 5-seed results were seen)

Small test (seeds 7, 107): PARTIAL (nominal GO; randomized: 27 % lower RMS@H, equal failure, seed-level CIs overlap).
The author approved continuing with all 5 seeds.

Primary endpoint: RMS@H of baseline SAC vs the best tuned classical controller, paired by evaluation episode
(same seeds, gusts and physics draws; 5 training seeds x 50 episodes), Wilcoxon signed-rank, nominal and randomized.
Secondary (all reported regardless of outcome): RMS_ss, failure rate, settling time; performance by reference
magnitude (envelope); model-mismatch robustness; curriculum/residual vs baseline at seed level.
Contingency, only if the primary test is not significant at alpha = 0.05: (1) equal-budget SAC tuning (9 configurations,
selected on held-out episodes, never the test set), (2) checkpoint selection on validation episodes, (3) additional seeds.
Not permitted: dropping seeds/episodes, changing the test set, metric or failure rule, weakening baselines.


## Amendment 2026-11-07 (airframe v3), fixed before any v3 comparison between SAC and the classical controllers

* Classical tuning is two-stage (revision/refine_classical.py): the 9-configuration grid, then 30 Nelder-Mead
  evaluations in log-parameter space started from the best grid point (initial simplex = factor 2 per parameter,
  physical bounds), on the same held-out tuning episodes (seeds 50000+). Every parametric controller receives
  39 evaluations. The selected configuration is evaluated once on the evaluation episodes.
* Reported per controller: stage-1 objective, stage-2 objective, refinement gain, and whether the stage-1 optimum
  lay on the grid edge.
* Alloc-ML was redesigned for airframe v3 (coordinated-turn bank reference, attitude-priority allocation,
  anti-windup, tuned bank limit) after its v2 structure departed on the v3 airframe; all design iterations used the
  tuning seeds only. The redesign is disclosed in the response to the reviewers.
* Small-test auto-continue rule: see revision/smalltest.py (added 21:37).


## Note 2026-11-07 23:10 (after the small test; no change to the pre-specified comparison)

* Small test (22:40): GO, auto-continue = true. SAC RMS@H 0.137 / 0.168 vs best classical (Heuristic) 0.325 / 0.396.
* Alloc-ML departed in 50/50 nominal small-test episodes (attitude limit, 2-7 s). Diagnosis (tuning seeds and
  static analysis only, revision/lateral_derivatives.py): the bowed spar gives a destabilising roll-due-to-sideslip
  (Cl_beta_NED = +0.18/rad) that exceeds the full roll authority at ~3.5 deg sideslip, and differential span (the
  strongest roll effector in the allocation) makes N_beta negative for sideslip toward the retracted wing.
  Raising the tips reduces |L_beta| five-fold.
* Decision: the pre-specified Alloc-ML is kept unchanged as the benchmark (no post-hoc redesign of a baseline after
  seeing SAC results). A supplementary variant, Alloc-PS (revision/alloc_variants.py: allocation re-identified about
  a raised-tip operating point, 72 configurations, selected on the tuning seeds with the standard objective), is
  reported separately in the appendix with its evaluation-episode performance, whatever the outcome.


## Note 2026-11-08 02:00 (declared before any 2M-step result existed)

* Budget sensitivity (revision/budget_sensitivity.py): flat SAC (seeds 7, 107) and curriculum SAC (seeds 17, 117)
  retrained from scratch with an equal 2M-step budget, evaluated on the same evaluation episodes, compared per seed
  with their 1M counterparts (paired Wilcoxon). Reported as a supplementary sensitivity analysis whatever the
  outcome; the pre-specified 1M comparison remains the primary result. An equal 3M budget for all 15 runs (~45M
  steps, ~20 h on this machine) was not feasible before the deadline.
* 02:15 update (still before any 2M result): extended to an equal 2M-step budget for all three paradigms and all
  5 seeds (15 runs), run in balanced order (one seed per paradigm per round). Reported alongside the 1M comparison
  with the same statistics; no 3M runs for any single paradigm (an unequal budget between the learned variants is
  exactly what the reviewers criticised).


## Note 2026-11-08 02:25 (user instruction: finish 1M first; improve the curriculum before any 2M run)

* The 2M runner (budget_sensitivity.py) was stopped before launching any job; no 2M run exists.
* Diagnosis of the original curriculum (morphing_glider/training/curriculum.py + phases.py), from its design only:
  phases change the dynamics (roll/pitch moments scaled by 0.10-0.90), the termination limit (90 -> 65 deg),
  the reward (stability weight, shaping), the action mapping (dihedral scaled 0 -> 1) and the learning rate,
  while 70% of the replay buffer is carried across phases -> the critic is trained on transitions that are not
  valid for the target MDP; the unmodified task receives only ~1/6 of the budget.
* Selection leakage found: within each phase the original curriculum restores its best checkpoint according to
  the training-time evaluation (25 episodes, seeds EVAL_SEED_BASE+i, rand 1.0), which are identical to episodes
  0-24 of the randomized evaluation set (same protocol, revision/evaluate.py). Its randomized-condition results
  are therefore optimistic. Reported: all 50 episodes as pre-specified, plus episodes 25-49 (not used for
  selection) as a check; flat SAC selects nothing.
* New method (development, not pre-specified): task-distribution curriculum (morphing_glider/training/
  task_curriculum.py): phases change only the commanded yaw-rate set, the randomization range and the hold
  time; dynamics, reward, termination, action mapping, optimizer, learning rate identical to flat SAC; replay
  buffer never reset; no checkpoint selection; last phase = flat-SAC training distribution (55% of budget).
  Development tests (revision/task_curriculum_tests.py, 500k steps, seeds 17/117) compare training-time curves
  with flat SAC; they use the training-time evaluation episodes for development only, and any final comparison
  uses the full pre-specified evaluation protocol.
