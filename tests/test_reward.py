"""Unit tests for morphing_glider.environment.reward.RewardComputer."""

import numpy as np
import pytest

from morphing_glider.environment.reward import RewardComputer


@pytest.fixture
def rc():
    """Create a RewardComputer with default weights."""
    return RewardComputer()


def _make_reward_kwargs(**overrides):
    """Build a minimal valid keyword dict for RewardComputer.compute()."""
    defaults = dict(
        yaw_error=0.0,
        roll=0.0,
        pitch=0.0,
        omega_p_clipped=0.0,
        omega_q_clipped=0.0,
        action=np.zeros(6),
        prev_action=np.zeros(6),
        power_norm=0.0,
        e_sum_norm=0.0,
        z_sym=0.0,
        stability_weight=1.0,
    )
    defaults.update(overrides)
    return defaults


class TestRewardComputerInstantiation:
    def test_default_instantiation(self, rc):
        assert isinstance(rc, RewardComputer)

    def test_survival_bonus_attribute(self, rc):
        assert rc.survival_bonus > 0.0

    def test_custom_weights(self):
        rc = RewardComputer(w_track=5.0, survival_bonus=0.5)
        assert rc.w_track == 5.0
        assert rc.survival_bonus == 0.5


class TestRewardComputeReturnType:
    def test_returns_float_and_dict(self, rc):
        reward, breakdown = rc.compute(**_make_reward_kwargs())
        assert isinstance(reward, float)
        assert isinstance(breakdown, dict)

    def test_breakdown_keys(self, rc):
        _, breakdown = rc.compute(**_make_reward_kwargs())
        expected = {
            "cost_track", "cost_att", "cost_rates", "cost_ctrl",
            "cost_jerk", "cost_power", "cost_struct", "cost_zsym",
            "tracking_reward", "survival_bonus", "wall_cost",
            "total_penalty", "total_cost", "total_reward",
            "w_att_eff", "w_rates_eff",
        }
        assert expected.issubset(set(breakdown.keys()))


class TestSurvivalBonus:
    def test_survival_bonus_is_positive(self, rc):
        _, breakdown = rc.compute(**_make_reward_kwargs())
        assert breakdown["survival_bonus"] > 0.0

    def test_survival_bonus_value_matches_config(self, rc):
        _, breakdown = rc.compute(**_make_reward_kwargs())
        np.testing.assert_allclose(
            breakdown["survival_bonus"], rc.survival_bonus, atol=1e-10
        )


class TestTrackingReward:
    def test_perfect_tracking_max_reward(self, rc):
        """Zero yaw error should give the maximum tracking reward."""
        _, bd_zero = rc.compute(**_make_reward_kwargs(yaw_error=0.0))
        _, bd_large = rc.compute(**_make_reward_kwargs(yaw_error=1.0))
        assert bd_zero["tracking_reward"] > bd_large["tracking_reward"]

    def test_tracking_reward_positive(self, rc):
        _, breakdown = rc.compute(**_make_reward_kwargs(yaw_error=0.0))
        assert breakdown["tracking_reward"] > 0.0

    def test_tracking_reward_decreases_with_error(self, rc):
        _, bd_small = rc.compute(**_make_reward_kwargs(yaw_error=0.1))
        _, bd_large = rc.compute(**_make_reward_kwargs(yaw_error=0.5))
        assert bd_small["tracking_reward"] > bd_large["tracking_reward"]


class TestPenalties:
    def test_zero_penalties_at_nominal(self, rc):
        """When all inputs are zero, all cost terms should be zero."""
        _, breakdown = rc.compute(**_make_reward_kwargs())
        for key in ("cost_track", "cost_att", "cost_rates",
                     "cost_ctrl", "cost_jerk", "cost_power",
                     "cost_struct", "cost_zsym"):
            np.testing.assert_allclose(breakdown[key], 0.0, atol=1e-10,
                                       err_msg=f"{key} should be zero at nominal")

    def test_attitude_penalty_increases_with_roll(self, rc):
        _, bd_flat = rc.compute(**_make_reward_kwargs(roll=0.0))
        _, bd_roll = rc.compute(**_make_reward_kwargs(roll=0.5))
        assert bd_roll["cost_att"] > bd_flat["cost_att"]

    def test_control_penalty_increases_with_action(self, rc):
        _, bd_zero = rc.compute(**_make_reward_kwargs(action=np.zeros(6)))
        _, bd_act = rc.compute(**_make_reward_kwargs(action=np.ones(6) * 0.5))
        assert bd_act["cost_ctrl"] > bd_zero["cost_ctrl"]

    def test_reward_clipping(self, rc):
        """Reward should be clipped within [clip_min, clip_max]."""
        reward, _ = rc.compute(**_make_reward_kwargs(
            yaw_error=100.0, roll=10.0, pitch=10.0,
            omega_p_clipped=10.0, omega_q_clipped=10.0,
            action=np.ones(6) * 10.0, prev_action=np.zeros(6),
            power_norm=10.0, e_sum_norm=2.0, z_sym=5.0,
        ))
        assert reward >= rc.clip_min
        assert reward <= rc.clip_max
