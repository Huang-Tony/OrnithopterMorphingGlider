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


## Note 2026-11-08 06:20 (development only)

* Robustness found that flat SAC departs in every episode with the measured 2.65 Hz wing mode in the loop; a post-hoc
  first-order command filter (cutoff selected on tuning episodes, 0.9-3.7 Hz grid) does not help (98% failures).
  Reported in the paper as a limitation. Pilot of a remedy (revision/mode_aware_pilot.py): flat SAC seed 7 retrained
  with a randomized wing mode in the loop; compared on validation episodes only. Not part of the pre-specified
  comparison; any use in the paper is the user's decision.


## Note 2026-11-08 06:50 (task curriculum: design fixed, validation result, confirmation on the evaluation protocol)

* Validation episodes (seeds 60000+, 50 per condition; analysis/task_curriculum_validation.json), 1M steps:
  flat SAC (5 seeds) 0.146 / 0.170 RMS@H nominal / randomized; original curriculum (5) 0.233 / 0.312;
  task curriculum (2 seeds, 17 and 117) 0.154 / 0.150. The design was not changed after seeing these numbers.
* The fixed design is completed to 5 seeds (217, 317, 417; revision/tc_complete.sh) and evaluated once with the
  evaluation protocol (revision/eval_task_curriculum.py). Whether it enters the paper is the user's decision.


## Note 2026-11-08 09:50 (task curriculum confirmed; 2M runs started)

* Fixed task-curriculum design, 5 seeds, evaluation protocol (analysis/task_curriculum_eval.json): RMS@H 0.135
  [0.124, 0.148] nominal, 0.166 [0.141, 0.194] randomized (failures 1.2% / 4.8%); flat SAC 0.138 / 0.172 (seed-level
  Mann-Whitney p = 0.69 / 0.84); original curriculum 0.242 / 0.292 (p = 0.008 / 0.008).
* Equal-2M runs started in the idle slots (revision/budget_sensitivity.py, revision_runs/budget2M): flat SAC and the
  task curriculum, 5 seeds each, balanced order. Residual variant at 2M left to the user. Stoppable at any time.


## Note 2026-11-08 13:10 (command smoothness: CAPS from scratch stopped; warm-start fine-tuning, rule fixed)

* CAPS from scratch (revision/caps_study.py; lambda_T = 5 selected by the 11:45 rule, which had no accuracy guard)
  was stopped at 600k steps: seed 107 stayed at the do-nothing level (training-time RMS@H 0.45-0.46 from 50k to
  600k steps), seed 7 was less reliable than flat SAC. Not reported as a result.
* Diagnosis (flat SAC seed 7, 20 validation episodes): removing sensor noise and gusts halves the mean tip speed
  (0.98 -> 0.50 m/s) and lowers the command share above 2 Hz from 29% to 22%; the rest is intrinsic.
* New study: revision/smooth_study.py (smoothness fine-tuning of the 1M flat-SAC policies, +200k steps). Pilot
  variants, selection rule (validation episodes only, accuracy guard <= 1.10 x base RMS@H in both conditions,
  failures <= base + 2/30, HF share <= 2/3 of base; lowest HF share wins; none eligible -> stop) and evaluation are
  fixed in its docstring before any result.


## Note 2026-11-08 14:10 (smoothness fine-tuning: outcome)

* Pilot (validation, seed 7): only caps1_jerk1 was admissible (RMS@H 0.125 vs 0.126 base nominal, 0.133 vs 0.133
  randomized; HF 15% vs 29%); caps1 failed the accuracy guard (0.155), jerk1 the HF criterion (23%); the plain
  continuation left HF at 31%. Selected: caps1_jerk1 (lambda_T 1, lambda_S 0.5, sigma 0.05, w_jerk 1.0, +200k steps).
* Evaluation protocol, 5 seeds (analysis/smooth_summary.json, smooth_attitude.json): RMS@H 0.137 vs 0.138 nominal,
  0.159 vs 0.172 randomized; HF 16% vs 35%; tip speed 0.64 vs 1.27 m/s; wing mode 99% vs 100% failures. Reported in
  the paper as SAC-S (Sect. res_smooth, Table tab:smooth); not part of the equal-budget comparisons.


## Note 2026-11-08 16:20 (user decisions on the development studies)

* Task-distribution curriculum: included in the paper as a test of the explanation for the original curriculum's
  result (methods Sect. 4.4, results Sect. 5.7, Discussion), explicitly as a post-hoc variant evaluated once.
* Mode-aware pilot (1 seed, validation episodes): mentioned in the Discussion as preliminary, with both sides of the
  trade-off (struct-mode departures 100% -> 28%; nominal RMS@H 0.130 -> 0.254, nominal failures 0% -> 10%).
