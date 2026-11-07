"""Curriculum phase table (moved verbatim from run_simulation.py so that staged drivers can import it).

Values are those of the paper-mode run. In revision R1 the per-phase `max_timesteps` cap is further
limited by the shared total budget passed to `train_with_curriculum(total_budget=...)`.
"""

import numpy as np

from morphing_glider.config import PAPER_RUN, MEDIUM_RUN
from morphing_glider.environment.wrappers import mild_curriculum_reward_shaper
from morphing_glider.training.curriculum import PhaseSpec


def make_phases(max_timesteps_per_phase: int):
    T = int(max_timesteps_per_phase)
    return [
        PhaseSpec(name="basic_yaw", twist_factor=0.0, rand_scale=0.0, max_timesteps=T,
                  ramp_steps=1800 if PAPER_RUN else (900 if MEDIUM_RUN else 500),
                  reward_shaper=mild_curriculum_reward_shaper,
                  residual_limit=np.array([0.10, 0.05, 0.06, 0.10, 0.05, 0.06], dtype=float),
                  learning_rate=3e-4, target_rms=0.52, min_steps_before_gate=20_000,
                  roll_pitch_limit_deg=90.0, coupling_scale=0.10, stability_weight=0.08),
        PhaseSpec(name="partial_twist", twist_factor=0.3, rand_scale=0.3, max_timesteps=T,
                  ramp_steps=3000 if PAPER_RUN else (1800 if MEDIUM_RUN else 1000),
                  reward_shaper=mild_curriculum_reward_shaper,
                  residual_limit=np.array([0.10, 0.05, 0.06, 0.10, 0.05, 0.06], dtype=float),
                  learning_rate=5e-4, target_rms=0.38, min_steps_before_gate=30_000,
                  roll_pitch_limit_deg=88.0, coupling_scale=0.25, stability_weight=0.06),
        PhaseSpec(name="intermediate_twist", twist_factor=0.45, rand_scale=0.45, max_timesteps=T,
                  ramp_steps=3000 if PAPER_RUN else (1800 if MEDIUM_RUN else 1000),
                  reward_shaper=mild_curriculum_reward_shaper,
                  residual_limit=np.array([0.12, 0.06, 0.08, 0.12, 0.06, 0.08], dtype=float),
                  learning_rate=4e-4, target_rms=0.33, min_steps_before_gate=35_000,
                  roll_pitch_limit_deg=85.0, coupling_scale=0.35, stability_weight=0.055),
        PhaseSpec(name="moderate_twist", twist_factor=0.6, rand_scale=0.6, max_timesteps=T,
                  ramp_steps=3500 if PAPER_RUN else (2000 if MEDIUM_RUN else 1200),
                  reward_shaper=mild_curriculum_reward_shaper,
                  residual_limit=np.array([0.15, 0.08, 0.10, 0.15, 0.08, 0.10], dtype=float),
                  learning_rate=4e-4, target_rms=0.28, min_steps_before_gate=40_000,
                  roll_pitch_limit_deg=84.0, coupling_scale=0.45, stability_weight=0.05),
        PhaseSpec(name="full_twist", twist_factor=1.0, rand_scale=1.0, max_timesteps=T,
                  ramp_steps=4000 if PAPER_RUN else (2500 if MEDIUM_RUN else 1500),
                  reward_shaper=mild_curriculum_reward_shaper,
                  residual_limit=np.array([0.20, 0.10, 0.12, 0.20, 0.10, 0.12], dtype=float),
                  learning_rate=2e-4, target_rms=0.20, min_steps_before_gate=50_000,
                  roll_pitch_limit_deg=70.0, coupling_scale=0.90, stability_weight=0.04),
        PhaseSpec(name="raw_finetune", twist_factor=1.0, rand_scale=1.0, max_timesteps=T,
                  ramp_steps=0, reward_shaper=mild_curriculum_reward_shaper,
                  residual_limit=np.array([0.25, 0.12, 0.15, 0.25, 0.12, 0.15], dtype=float),
                  learning_rate=1e-4, target_rms=None, min_steps_before_gate=60_000,
                  roll_pitch_limit_deg=65.0, coupling_scale=1.00, stability_weight=0.035),
    ]
