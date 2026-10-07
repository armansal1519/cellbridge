"""Static research figures and an evidence ledger, without invented results."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from .artifacts import validate_receipt, complete_stage, require_receipt


def _plot_available_bars(ax, table):
    """Keep unavailable metrics visibly missing instead of plotting them as zero."""
    positions = np.arange(len(table))
    width = .8 / max(len(table.columns), 1)
    for index, column in enumerate(table.columns):
        values = table[column].to_numpy(dtype=float)
        finite = np.isfinite(values)
        x = positions + (index-(len(table.columns)-1)/2)*width
        ax.bar(x[finite], values[finite], width=width, label=str(column))
        for missing in x[~finite]:
            ax.annotate("N/A", (missing, 0), xytext=(0, 4), textcoords="offset points",
                        rotation=90, ha="center", va="bottom", fontsize=7, color="dimgray")
    ax.set_xticks(positions, table.index, rotation=20)


def render_report(run):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run = Path(run); out = run / "report"; out.mkdir(exist_ok=True)
    require_receipt(run / "evaluation")
    if validate_receipt(out): return out / "report.md"
    summary = pd.read_csv(run / "evaluation" / "primary_summary.csv")
    gates = json.loads((run / "evaluation" / "advancement.json").read_text())
    protocol = json.loads((run / "protocol.json").read_text())
    dataset = protocol.get("development", "unspecified dataset")
    selected = summary.loc[summary.panel.isin(["optimized_4", "optimized_8", "fixed_4", "fixed_8"])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, metric, title in zip(axes, ["standardized_absolute_error", "direction_error"],
                                ["CD25/CD69 response error", "Response direction error"]):
        table = selected.pivot(index="panel", columns="method", values=metric)
        _plot_available_bars(ax, table)
        ax.set_title(title); ax.set_xlabel(""); ax.set_ylabel("Donor/context-balanced mean")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=7)
    fig.savefig(out / "benchmark.png", dpi=180); fig.savefig(out / "benchmark.pdf"); plt.close(fig)
    rank_info = []
    for path in sorted(run.glob("fold_*/response/rank_selection.json")):
        info = json.loads(path.read_text()); rank_info.append(f"- {path.parent.parent.name}: {info['selected']}")
    lines = ["# ResponseBridge execution report", "", "This is an executed research benchmark, not evidence of publication readiness.", "",
        f"Dataset: **{dataset}**. Primary lineage: **{protocol.get('cohort_lineage') or 'all included lineages'}**. Primary perturbations: **{protocol.get('primary_perturbations') or 'all eligible contrasts'}**.", "",
        f"Pilot subset of outer folds: **{gates['pilot']}**.", "", "![Primary comparisons](benchmark.png)", "",
        "N/A denotes an unavailable metric, not zero error. When ResponseBridge refuses every primary response, its accuracy and direction error are undefined; finite comparator results remain visible.", "",
        "## Primary operational comparisons", "", "| Panel | Relative error reduction vs RNA+anchors ridge | Direction error change | Accuracy gate |",
        "|---|---:|---:|---|"]
    for x in gates["comparisons"]:
        gain = "unavailable" if x['relative_mae_gain'] is None else f"{100*x['relative_mae_gain']:.2f}%"
        direction = "unavailable" if x['direction_error_change'] is None else f"{x['direction_error_change']:+.4f}"
        lines.append(f"| {x['panel']} | {gain} | {direction} | {x['operational_accuracy_gate_passed']} |")
    lines += ["", "The 10% threshold is a project decision rule, not a publication standard. A negative gain is an unfavorable result and remains in the report.", "",
        "## Selected response ranks", "", *rank_info, "", "## Claims and evidence", "",
        "| Claim | Status |", "|---|---|",
        "| Numerical recovery and information limits | See executed simulation and test receipts; no new theorem claimed |",
        "| Accuracy superiority | Assess the frozen-fold table above; not assumed |",
        "| Independent biological replication | Not established by this single-study benchmark |",
        "| Arbitrary unseen-context validity | Not claimed; invisible target-only shifts are possible |",
        f"| Population inference | {gates.get('evaluated_donors', 'See donor table')} evaluated donors; inspect donor directions and conditional uncertainty limits |",
        "| Post-transcriptional mechanism | Not inferred from an RNA model residual |", "",
        "## Required remaining evidence", "", *[f"- {s}" for s in gates["unresolved"]], "",
        "## Interpretation and uncertainty", "",
        "RNA fits operate on equal-weight donor/context/perturbation means. An affine predictor commutes with aggregation. Cell counts determine precision but are not independent biological replicates. Each matched contrast uses control cells restricted to its partition. HVGs, scaling, ridge penalties, response ranks and anchor panels are learned from development partitions.", "",
        "Structural observability is conditional on the estimated response basis. Interval components for anchor sampling, response-basis estimation and held-context mismatch must be distinguished. Current benchmark envelopes come from rank-selected inner validation residuals; evaluate outer empirical coverage rather than interpreting them as guaranteed confidence intervals. Technical controls and ADT-depth artifacts require separate checks before biological interpretation.", "",
        protocol.get("previous_exposure", "Prior dataset exposure must be documented before interpreting validation.")]
    coverage = summary.loc[summary.method == "responsebridge", ["panel", "prediction_available", "interval_available", "precision_usable", "interval_covered", "interval_width"]]
    coverage.to_csv(out / "coverage.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    coverage_plot = coverage.set_index("panel")[["prediction_available", "precision_usable",
        "interval_available", "interval_covered"]].rename(columns={
            "prediction_available": "Point availability", "precision_usable": "Precision usability",
            "interval_available": "Interval availability", "interval_covered": "Coverage of available intervals"})
    _plot_available_bars(ax, coverage_plot)
    ax.legend(fontsize=7)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Fraction"); ax.set_title("Availability, precision and interval coverage")
    fig.savefig(out / "coverage.png", dpi=180); plt.close(fig)
    lines += ["", "## Refusal and uncertainty", "", "![Availability and coverage](coverage.png)", "",
              "The accuracy gate requires complete CD25/CD69 point coverage. `common_accepted_summary.csv` compares every method on precisely the targets accepted by ResponseBridge. Refused targets are never counted as correct, and missing intervals are not counted as covered.", "",
              "Point availability, precision usability and interval availability retain all primary responses in their denominators. Empirical interval coverage and mean width are evaluated only where intervals are available. If no intervals are available, their coverage and width remain missing (N/A), while interval availability and precision usability are zero.", "",
              "The optional scVAEIT comparator is a Gaussian group-mean adaptation of the official package. Executing it does not establish adequate tuning or reproduce the paper's cell-level experiment."]
    (out / "report.md").write_text("\n".join(lines)+"\n")
    complete_stage(out)
    return out / "report.md"
