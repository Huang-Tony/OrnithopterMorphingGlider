"""Evaluation subpackage: episode execution, metrics, controller evaluation, and robustness tests."""

from morphing_glider.evaluation.metrics import (
    run_episode,
    _segment_indices,
    compute_episode_metrics,
    summarize_metrics,
)
from morphing_glider.evaluation.evaluate import (
    evaluate_controller,
    _bca_summary,
    _mean_of_metric,
    eval_model_run_metrics,
    EVAL_METRIC_KEYS,
    summarize_trained_algo_hierarchical,
    summarize_controller_over_episodes_bca,
    print_final_eval_table,
    _standardize_evaltrace_append,
)
from morphing_glider.evaluation.robustness import (
    eval_ood_yaw_targets,
    eval_distribution_shift,
    eval_sensor_corruption,
    eval_long_horizon,
    eval_mid_episode_parameter_jump,
    model_quality_ceiling,
)
