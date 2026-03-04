"""Unit tests for morphing_glider.utils.numeric functions."""

import math

import numpy as np
import pytest

from morphing_glider.utils.numeric import (
    rms,
    mae,
    finite_mean_std,
    bootstrap_mean_ci_bca,
    holm_bonferroni,
    paired_tests,
)


class TestRms:
    def test_known_values(self):
        # rms([3, 4]) = sqrt((9+16)/2) = sqrt(12.5) = 3.5355...
        result = rms([3.0, 4.0])
        np.testing.assert_allclose(result, math.sqrt(12.5), atol=1e-10)

    def test_constant_value(self):
        # rms of a constant c is |c|
        result = rms([5.0, 5.0, 5.0])
        np.testing.assert_allclose(result, 5.0, atol=1e-10)

    def test_single_value(self):
        result = rms([7.0])
        np.testing.assert_allclose(result, 7.0, atol=1e-10)

    def test_empty_array(self):
        result = rms([])
        assert result == 0.0

    def test_zeros(self):
        result = rms([0.0, 0.0, 0.0])
        assert result == 0.0

    def test_negative_values(self):
        # rms should be the same for [3, -4] as [3, 4]
        np.testing.assert_allclose(rms([3.0, -4.0]), rms([3.0, 4.0]), atol=1e-10)


class TestMae:
    def test_known_values(self):
        # mae([1, -2, 3]) = (1+2+3)/3 = 2.0
        result = mae([1.0, -2.0, 3.0])
        np.testing.assert_allclose(result, 2.0, atol=1e-10)

    def test_zeros(self):
        result = mae([0.0, 0.0])
        assert result == 0.0

    def test_empty_array(self):
        result = mae([])
        assert result == 0.0

    def test_positive_values(self):
        result = mae([1.0, 2.0, 3.0])
        np.testing.assert_allclose(result, 2.0, atol=1e-10)

    def test_single_value(self):
        result = mae([-4.5])
        np.testing.assert_allclose(result, 4.5, atol=1e-10)


class TestFiniteMeanStd:
    def test_normal_values(self):
        mean, std = finite_mean_std([1.0, 2.0, 3.0])
        np.testing.assert_allclose(mean, 2.0, atol=1e-10)
        expected_std = math.sqrt(((1-2)**2 + (2-2)**2 + (3-2)**2) / 2)  # ddof=1
        np.testing.assert_allclose(std, expected_std, atol=1e-10)

    def test_nan_filtering(self):
        mean, std = finite_mean_std([1.0, float("nan"), 3.0])
        np.testing.assert_allclose(mean, 2.0, atol=1e-10)

    def test_all_nan(self):
        mean, std = finite_mean_std([float("nan"), float("nan")])
        assert math.isnan(mean)
        assert math.isnan(std)

    def test_inf_filtering(self):
        mean, std = finite_mean_std([1.0, float("inf"), 3.0])
        np.testing.assert_allclose(mean, 2.0, atol=1e-10)

    def test_single_value(self):
        mean, std = finite_mean_std([5.0])
        np.testing.assert_allclose(mean, 5.0, atol=1e-10)
        # ddof=1 with n=1 gives NaN (undefined sample std)
        assert math.isnan(std)


class TestBootstrapMeanCiBca:
    def test_returns_triple(self):
        result = bootstrap_mean_ci_bca([1.0, 2.0, 3.0, 4.0, 5.0], n_boot=200)
        assert len(result) == 3

    def test_mean_value(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        mean, lo, hi = bootstrap_mean_ci_bca(data, n_boot=200)
        np.testing.assert_allclose(mean, 3.0, atol=1e-10)

    def test_lo_le_mean_le_hi(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        mean, lo, hi = bootstrap_mean_ci_bca(data, n_boot=500)
        assert lo <= mean <= hi

    def test_single_element(self):
        mean, lo, hi = bootstrap_mean_ci_bca([42.0])
        assert mean == 42.0
        assert lo == 42.0
        assert hi == 42.0

    def test_identical_values_narrow_ci(self):
        data = [3.0] * 20
        mean, lo, hi = bootstrap_mean_ci_bca(data, n_boot=200)
        np.testing.assert_allclose(mean, 3.0, atol=1e-10)
        np.testing.assert_allclose(lo, 3.0, atol=1e-10)
        np.testing.assert_allclose(hi, 3.0, atol=1e-10)


class TestHolmBonferroni:
    def test_basic_correction(self):
        pvals = {"a": 0.01, "b": 0.04, "c": 0.80}
        result = holm_bonferroni(pvals, alpha=0.05)
        assert set(result.keys()) == {"a", "b", "c"}
        # "a" has smallest p, compared against 0.05/3
        assert result["a"]["reject"] is True
        # "c" should not be rejected
        assert result["c"]["reject"] is False

    def test_all_significant(self):
        pvals = {"x": 0.001, "y": 0.002}
        result = holm_bonferroni(pvals, alpha=0.05)
        assert result["x"]["reject"] is True
        assert result["y"]["reject"] is True

    def test_none_significant(self):
        pvals = {"x": 0.50, "y": 0.80}
        result = holm_bonferroni(pvals, alpha=0.05)
        assert result["x"]["reject"] is False
        assert result["y"]["reject"] is False

    def test_rank_ordering(self):
        pvals = {"b": 0.02, "a": 0.01, "c": 0.03}
        result = holm_bonferroni(pvals, alpha=0.05)
        assert result["a"]["rank"] == 1
        assert result["b"]["rank"] == 2
        assert result["c"]["rank"] == 3

    def test_result_fields(self):
        pvals = {"test": 0.03}
        result = holm_bonferroni(pvals, alpha=0.05)
        entry = result["test"]
        assert "p" in entry
        assert "rank" in entry
        assert "adj_alpha" in entry
        assert "reject" in entry


class TestPairedTests:
    def test_equal_arrays(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = paired_tests(x, x)
        assert "mean_diff" in result
        np.testing.assert_allclose(result["mean_diff"], 0.0, atol=1e-10)
        np.testing.assert_allclose(result["cohen_d"], 0.0, atol=1e-10)

    def test_different_arrays(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 3.0, 4.0, 5.0, 6.0]
        result = paired_tests(x, y)
        np.testing.assert_allclose(result["mean_diff"], -1.0, atol=1e-10)
        assert result["cohen_d"] != 0.0

    def test_too_few_samples(self):
        result = paired_tests([1.0], [2.0])
        assert math.isnan(result["p_ttest"])
        assert math.isnan(result["mean_diff"])

    def test_return_keys(self):
        result = paired_tests([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        expected_keys = {"p_ttest", "p_wilcoxon", "p_mannwhitney", "mean_diff", "cohen_d", "cohen_d_paired"}
        assert set(result.keys()) == expected_keys
