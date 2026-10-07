from responsebridge.cellbridge_open import conformal_coverage_rate, simulate_metacell_attenuation


def test_averaging_cells_that_share_a_state_reduces_attenuation():
    rows = simulate_metacell_attenuation(sigmas=(2.0,), n_donors=6, seed=1)
    by = {row["grouping"]: row["slope_norm_ratio"] for row in rows}
    assert by["oracle_states"] > by["cells"] + 0.2


def test_split_conformal_covers_about_ninety_percent():
    result = conformal_coverage_rate(n_repeats=200, seed=2)
    assert 0.84 <= result["coverage"] <= 0.96
