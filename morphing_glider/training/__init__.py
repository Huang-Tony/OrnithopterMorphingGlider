"""Training subpackage: environment construction, SAC model building, and training loops."""

from morphing_glider.training.curriculum import PhaseSpec, TrainRunResult, summarize_curriculum_progression, train_with_curriculum
from morphing_glider.training.baseline import train_baseline_sac
from morphing_glider.training.infrastructure import (
    make_env,
    make_vec_env,
    _find_wrapper,
    warmup_vecnormalize,
    apply_phase_runtime_settings,
    build_sac_model_baseline,
    build_sac_model,
    _set_phase_lr_on_sac,
    _standardize_evaltrace_append,
    _partial_replay_reset,
    _apply_residual_limit_on_vec,
    build_training_env_for_phase,
    save_model_and_vecnorm,
    load_vecnorm_for_eval,
    save_training_checkpoint,
    verify_checkpoint_reproducibility,
)
