"""Unit tests for Stage 08 statistical significance and ablation utilities."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

MODULE_PATH = Path(__file__).resolve().parent.parent / "src" / "08_comparative_analysis_ablation.py"
SPEC = importlib.util.spec_from_file_location("stage08", MODULE_PATH)
stage08 = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(stage08)


def test_delong_identical_and_superior_scores() -> None:
    rng = np.random.default_rng(7)
    y = np.array([0] * 100 + [1] * 50)
    prob_good = np.concatenate([rng.uniform(0.0, 0.5, 100), rng.uniform(0.4, 1.0, 50)])
    prob_weak = np.concatenate([rng.uniform(0.2, 0.7, 100), rng.uniform(0.2, 0.7, 50)])

    # Identical scores -> delta=0, p=1
    res_same = stage08.delong_roc_test(y, prob_good, prob_good)
    assert res_same["delta_auc"] == pytest.approx(0.0, abs=1e-12)
    assert res_same["p_value"] == pytest.approx(1.0, abs=1e-12)

    # Superior vs weak -> matches sklearn AUC and significant p < 0.01
    res_diff = stage08.delong_roc_test(y, prob_good, prob_weak)
    assert res_diff["auc_a"] == pytest.approx(roc_auc_score(y, prob_good), abs=1e-10)
    assert res_diff["auc_b"] == pytest.approx(roc_auc_score(y, prob_weak), abs=1e-10)
    assert res_diff["delta_auc"] > 0.2
    assert res_diff["p_value"] < 0.01


def test_mcnemar_and_holm_bonferroni() -> None:
    y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    pred_a = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    pred_b = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    res = stage08.mcnemar_test(y, pred_a, pred_b)
    assert res["n10_a_right_b_wrong"] == 4
    assert res["n01_a_wrong_b_right"] == 0
    assert res["discordant"] == 4
    assert 0.0 < res["p_value_exact"] < 0.2

    adj = stage08.holm_bonferroni([0.001, 0.04, 0.03])
    assert adj == pytest.approx([0.003, 0.06, 0.06], abs=1e-12)
