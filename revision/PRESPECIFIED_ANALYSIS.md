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
