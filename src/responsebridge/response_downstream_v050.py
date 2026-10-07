"""Learning curves, panel transfer, noise ceilings and gate G1/G2 summaries."""
from __future__ import annotations

import numpy as np

from .response_evaluate_v050 import holm, random_effects_mean, skill_vs_mean, standardized_mae


def sign_flip_p(differences):
    """Two-sided exact sign-flip p-value for a mean paired difference."""
    diffs = np.asarray(differences, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    n = len(diffs)
    if n == 0:
        return None
    observed = abs(float(diffs.mean()))
    if n > 16:
        rng = np.random.default_rng(20261007)
        draws = rng.choice([-1.0, 1.0], size=(10000, n))
        means = np.abs((draws * diffs).mean(axis=1))
        return float((1 + np.sum(means >= observed - 1e-15)) / (len(means) + 1))
    total = 1 << n
    count = 0
    for bits in range(total):
        signs = np.array([1.0 if (bits >> i) & 1 else -1.0 for i in range(n)])
        if abs(float((signs * diffs).mean())) >= observed - 1e-15:
            count += 1
    return float(count / total)


def learning_curve_donor_counts(n_train, sizes=(2, 4, 8, 16, 32)):
    return [k for k in sizes if 3 <= k <= int(n_train)]


def overlapping_proteins(names_a, names_b):
    a, b = set(map(str, names_a)), set(map(str, names_b))
    return sorted(a & b)


def gate_g1(task_summaries, *, skill_majority=0.5):
    """Development gate: skill vs mean, no systematic RNA-only harm, baseline."""
    gated, mean_skill, harm, baseline_delta = [], [], [], []
    for task in task_summaries:
        rows = {row["method"]: row for row in task["summary"]}
        if "gated_abundance_response" not in rows:
            continue
        g = rows["gated_abundance_response"]
        gated.append(g)
        mean_skill.append(g.get("skill_vs_mean"))
        if "rna_only" in rows:
            harm.append(g["standardized_mae"] - rows["rna_only"]["standardized_mae"])
        same = [rows[k]["standardized_mae"] for k in ("joint_ridge", "two_penalty_ridge", "elastic_net",
                                                      "kernel_ridge", "pls") if k in rows]
        if same:
            baseline_delta.append(min(same) - g["standardized_mae"])
    n = len(mean_skill)
    positive = sum(s is not None and s > 0 for s in mean_skill)
    no_harm = (not harm) or (sum(h <= 1e-12 for h in harm) >= max(1, int(np.ceil(skill_majority * len(harm)))))
    beats = (not baseline_delta) or (float(np.mean(baseline_delta)) > 0)
    passed = n > 0 and (positive / n) >= skill_majority and no_harm and beats
    return {"n_tasks": n, "positive_skill_fraction": (positive / n) if n else None,
            "no_systematic_harm_vs_rna_only": bool(no_harm),
            "beats_tuned_same_information_mean": bool(beats),
            "mean_skill_vs_mean": None if not mean_skill else float(np.nanmean(mean_skill)),
            "mean_mae_gain_vs_best_baseline": None if not baseline_delta else float(np.mean(baseline_delta)),
            "passed": bool(passed)}


def gate_g2(cohort_inferences, coverage_rows, alpha=0.1):
    """Independent-claim gate after Holm across sealed cohorts."""
    pvals, names = [], []
    for row in cohort_inferences:
        p = row.get("p_sign_flip")
        if p is None:
            continue
        pvals.append(p)
        names.append(row.get("cohort", str(len(names))))
    adj = holm(pvals).tolist() if pvals else []
    n_strong = sum(p < 0.05 and row.get("mean_difference", 0) > 0
                   for p, row in zip(adj, [r for r in cohort_inferences if r.get("p_sign_flip") is not None]))
    coverage_ok = True
    for cov in coverage_rows or []:
        value = cov.get("coverage")
        if value is None:
            continue
        if value + 1e-12 < 1 - alpha - 0.05:
            coverage_ok = False
    outcome = "strong" if n_strong >= 3 and coverage_ok else "mixed_or_null"
    return {"n_cohorts_tested": len(pvals), "holm_adjusted": dict(zip(names, adj)),
            "n_positive_after_holm": int(n_strong), "coverage_not_far_below_nominal": coverage_ok,
            "outcome": outcome, "venue_rule": (
                "Genome Biology methodology or Nature Methods stretch"
                if outcome == "strong"
                else "Benchmark and limits paper (Nature Communications or Genome Biology)")}


def pooled_meta(cohort_inferences):
    effects = [r["mean_difference"] for r in cohort_inferences if r.get("mean_difference") is not None]
    variances = [r["variance"] for r in cohort_inferences if r.get("variance") is not None]
    if len(effects) < 2:
        return None
    return random_effects_mean(effects, variances)


def relative_to_ceiling(mae, ceiling_mae):
    mae, ceiling_mae = float(mae), float(ceiling_mae)
    if not np.isfinite(mae) or not np.isfinite(ceiling_mae) or ceiling_mae <= 0:
        return None
    return float(mae / ceiling_mae)
