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
