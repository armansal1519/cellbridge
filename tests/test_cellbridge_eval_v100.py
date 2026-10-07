import numpy as np

from responsebridge.cellbridge_eval import cohort_effect, fixed_sequence, meta_analyze


def test_sequence_stops_at_the_first_non_rejection():
    donors = [f"d{i}" for i in range(8)]
    cell = np.ones(8)
    strong = np.full(8, 3.0)
    weak = np.array([1.2, 0.8, 1.2, 0.8, 1.2, 0.8, 1.2, 0.8])
    loss = {("cellbridge", "E2"): cell, ("mean", "E2"): strong, ("joint_ridge", "E2"): weak}
    hypotheses = [
        {"id": "H1", "endpoint": "E2", "comparator": "mean"},
        {"id": "H2", "endpoint": "E2", "comparator": "joint_ridge"},
        {"id": "H3", "endpoint": "E2", "comparator": "absent"},
    ]
    tests = fixed_sequence(loss, hypotheses, donors, alpha=0.05, seed=1, metadata={"protocol": "test"})
    assert tests[0]["rejected_null"] and tests[0]["tested_in_sequence"]
    assert tests[1]["tested_in_sequence"] and not tests[1]["rejected_null"]
    assert tests[2]["status"] == "comparator_not_available_removed_by_protocol"


def test_meta_analysis_matches_inverse_variance():
    cohorts = [
        {"cohort": "a", **cohort_effect([1.0, 1.2, 0.8, 1.1])},
        {"cohort": "b", **cohort_effect([0.4, 0.6, 0.5, 0.7, 0.3])},
    ]
    result = meta_analyze(cohorts)
    weights = np.array([1 / c["variance"] for c in cohorts])
    expected = float(np.sum(weights * [c["mean"] for c in cohorts]) / np.sum(weights))
    assert abs(result["fixed_effect_mean"] - expected) < 1e-12
    assert result["random_effects"]["mean"] == result["random_effects"]["mean"]
