"""Publication figures for the CellBridge paper. Reads sealed and post-hoc tables only."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
SRC = OUT / "source"
MM = 1 / 25.4
TEAL, PURPLE, ORANGE = "#147D92", "#79549D", "#C66A35"
INK, GREY, PALE = "#1F2933", "#64748B", "#CBD5E1"
ANCHORS = {"CD38", "ICOS", "PD-1", "HLA-DR", "CD127", "CD27", "CD28", "CD45RO"}
CONTRASTS = ["mean", "joint_ridge", "rna_ridge", "totalvi", "v040_abundance_shrinkage"]
CONTRAST_LABEL = {
    "mean": "Mean", "joint_ridge": "Joint ridge", "rna_ridge": "RNA ridge",
    "totalvi": "totalVI", "v040_abundance_shrinkage": "Abundance shrinkage",
}
CONTRAST_COLOR = {
    "mean": PALE, "joint_ridge": GREY, "rna_ridge": "#94A3B8",
    "totalvi": ORANGE, "v040_abundance_shrinkage": PURPLE,
}


def _style():
    for path in ("/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/System/Library/Fonts/Supplemental/Arial Bold.ttf"):
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 7.5, "axes.labelsize": 8,
        "axes.titlesize": 8, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    })


def _json(path):
    return json.loads(Path(path).read_text())


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    SRC.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=600)
    plt.close(fig)


def _letter(ax, letter):
    ax.text(-0.01, 1.06, letter, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom", ha="right", clip_on=False)


def _source(name, frame):
    SRC.mkdir(parents=True, exist_ok=True)
    frame.to_csv(SRC / f"{name}.csv", index=False)


def _gains(path, endpoint):
    frame = pd.read_csv(path)
    test = frame[frame.role == "test"]
    base = test[(test.method == "cellbridge") & (test.endpoint == endpoint)].set_index("donor").loss
    rows = []
    for contrast in CONTRASTS:
        other = test[(test.method == contrast) & (test.endpoint == endpoint)].set_index("donor").loss
        for donor, loss in other.reindex(base.index).items():
            rows.append({"contrast": contrast, "donor": donor, "gain": float(loss - base.loc[donor]),
                         "endpoint": endpoint})
    return pd.DataFrame(rows)


def _forest(ax, decision, mark=None):
    rows = list(reversed(decision["hypotheses"]))
    for i, row in enumerate(rows):
        if "difference_ci95" not in row:
            continue
        lo, hi = row["difference_ci95"]
        tested = row.get("tested_in_sequence")
        rejected = row.get("rejected_null")
        color = TEAL if tested and rejected else (ORANGE if tested else PALE)
        ax.plot([lo, hi], [i, i], color=color, lw=1.2, solid_capstyle="round")
        ax.plot(row["mean_difference"], i, "o", color=color, ms=3.5)
        if mark and row["id"] == mark:
            ax.annotate("sequence stops", (hi, i), textcoords="offset points", xytext=(4, 6),
                        fontsize=6.5, color=ORANGE)
    ax.axvline(0, color=GREY, lw=0.6)
    labels = []
    for row in rows:
        end = "E1" if row["endpoint"] == "E1_primary" else "E2"
        labels.append(f"{row['id']}  {end} vs {CONTRAST_LABEL.get(row['comparator'], row['comparator'])}")
    ax.set_yticks(range(len(rows)), labels)
    ax.set_xlabel("Gain in standardized MAE")
    ax.set_ylim(-0.6, len(rows) - 0.4)


def _scatter(ax, path):
    frame = pd.read_csv(path)
    colors = {"CD25": TEAL, "CD69": ORANGE, "CD69-1": ORANGE}
    labels = {"CD25": "CD25", "CD69": "CD69", "CD69-1": "CD69"}
    for protein, part in frame.groupby("protein"):
        ax.scatter(part.observed, part.predicted, s=16, color=colors.get(protein, GREY),
                   label=labels.get(protein, protein), zorder=3, linewidths=0)
    vals = np.concatenate([frame.observed.to_numpy(), frame.predicted.to_numpy()])
    lo, hi = float(vals.min()), float(vals.max())
    pad = 0.08 * (hi - lo + 1e-6)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=PALE, lw=0.7, zorder=1)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Observed response")
    ax.set_ylabel("Predicted response")
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")


def figure1():
    fig, axes = plt.subplots(1, 3, figsize=(174 * MM, 62 * MM), constrained_layout=True)
    ax = axes[0]
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    steps = [(0.70, "1", "Both arms", "RNA and eight anchors"),
             (0.42, "2", "Donor change", "Difference of the arm means"),
             (0.14, "3", "Hidden proteins", "CD25, CD69 and the rest")]
    for y, number, title, body in steps:
        ax.add_patch(FancyBboxPatch((0.08, y), 0.84, 0.22, boxstyle="round,pad=0.008,rounding_size=0.02",
                                    facecolor="#E6F3F6", edgecolor=TEAL, lw=0.8))
        ax.text(0.16, y + 0.11, number, ha="center", va="center", fontsize=9, fontweight="bold", color=TEAL)
        ax.text(0.24, y + 0.14, title, ha="left", va="center", fontsize=7.5, fontweight="bold", color=TEAL)
        ax.text(0.24, y + 0.06, body, ha="left", va="center", fontsize=6.5, color=INK)
    ax.set_title("What is predicted", loc="left", fontsize=8)
    _letter(ax, "a")

    ax = axes[1]
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.78, r"$y=c_{d}+\gamma+z^{\top}\beta$", ha="center", va="center", fontsize=11, color=INK)
    notes = [(0.17, "Donor effects\nremoved"), (0.50, r"$\rho$ balances" "\ncells and donors"),
             (0.83, "Closed-form\nridge")]
    for x, text in notes:
        ax.add_patch(FancyBboxPatch((x - 0.14, 0.22), 0.28, 0.32, boxstyle="round,pad=0.02,rounding_size=0.03",
                                    facecolor="#F4F7F8", edgecolor=PALE, lw=0.6))
        ax.text(x, 0.38, text, ha="center", va="center", fontsize=6.5)
    ax.set_title("Cell model", loc="left", fontsize=8)
    _letter(ax, "b")

    ax = axes[2]
    water = pd.read_csv(ROOT / "runs/cellbridge_v100/gse334503_posthoc/waterfall_cd69/table.csv")
    record = _json(ROOT / "runs/cellbridge_v100/gse334503_posthoc/waterfall_cd69/record.json")
    show = water.iloc[::-1]
    colors = [TEAL if f in ANCHORS else (PALE if f == "other" else GREY) for f in show.feature]
    ax.barh(np.arange(len(show)), show.contribution, color=colors, height=0.72)
    ax.axvline(0, color=INK, lw=0.4)
    ax.set_yticks(np.arange(len(show)), show.feature, fontsize=6)
    ax.set_xlabel("Contribution to the prediction")
    ax.set_title(f"{record['donor']}, CD69", loc="left", fontsize=8)
    _letter(ax, "c")
    _source("figure1c_waterfall", water)
    _save(fig, "figure1_model")


def figure2():
    fig, axes = plt.subplots(1, 3, figsize=(174 * MM, 78 * MM), constrained_layout=True)
    dev = pd.read_csv(ROOT / "runs/cellbridge_v100/analysis/development/summary.csv")
    methods = ["cellbridge[shared_top05]", "v040_abundance_shrinkage", "joint_ridge", "rna_ridge", "mean"]
    labels = ["CellBridge", "Shrinkage", "Joint", "RNA", "Mean"]
    colors = [TEAL, PURPLE, GREY, "#94A3B8", PALE]
    ax = axes[0]
    rows = []
    rng = np.random.default_rng(0)
    for i, (method, color) in enumerate(zip(methods, colors)):
        vals = dev.loc[dev.method == method, "all_mae"].to_numpy()
        jitter = rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=12, color=color, zorder=3, linewidths=0)
        ax.plot([i - 0.18, i + 0.18], [vals.mean(), vals.mean()], color=INK, lw=0.8, zorder=4)
        for task, value in dev.loc[dev.method == method, ["task", "all_mae"]].itertuples(index=False):
            rows.append({"task": task, "method": method, "all_mae": value})
    ax.set_xticks(range(len(labels)), labels, fontsize=6.5, rotation=25, ha="right")
    ax.set_ylabel("Standardized MAE")
    ax.set_title("Nine development tasks", loc="left")
    _letter(ax, "a")
    _source("figure2a_development", pd.DataFrame(rows))

    sub = pd.read_csv(ROOT / "runs/cellbridge_v100/open_ablations/subgrid_means.csv")
    order = [
        ("cellbridge[shared_top05]", "RNA+anchors"),
        ("cellbridge[shared_top05|anchors_required]", "Anchors in\nevery fit"),
        ("cellbridge[shared_top05|k1]", "k = 1"),
        ("cellbridge[shared_top05|rho0]", r"$\rho$ = 0"),
        ("cellbridge[shared_top05|rna_only]", "RNA only"),
        ("joint_ridge", "Joint ridge"),
        ("mean", "Mean"),
    ]
    vals = [float(sub.loc[sub.method == m, "all_mae"].iloc[0]) for m, _ in order]
    ax = axes[1]
    colors = [TEAL, TEAL, TEAL, TEAL, "#94A3B8", GREY, PALE]
    ax.barh(np.arange(len(vals))[::-1], vals, color=colors, height=0.72)
    ax.set_yticks(np.arange(len(order))[::-1], [lab.replace("\n", " ") for _, lab in order], fontsize=6.5)
    ax.set_xlabel("Mean standardized MAE")
    ax.set_title("What the fit uses", loc="left")
    _letter(ax, "b")
    _source("figure2b_ablation", pd.DataFrame({"label": [lab.replace("\n", " ") for _, lab in order], "all_mae": vals}))

    panel = pd.read_csv(ROOT / "runs/cellbridge_v100/open_ablations/panel_curve.csv")
    grouped = panel.groupby("n_anchors").all_mae
    ax = axes[2]
    ax.fill_between(grouped.mean().index, grouped.min(), grouped.max(), color=TEAL, alpha=0.15, lw=0)
    ax.plot(grouped.mean().index, grouped.mean().values, "o-", color=TEAL, ms=4)
    ax.set_xticks([0, 1, 2, 4, 8])
    ax.set_xlabel("Anchors, added by inner error")
    ax.set_ylabel("Standardized MAE")
    ax.set_title("Panel size, nine tasks", loc="left")
    _letter(ax, "c")
    _source("figure2c_panel", panel)
    _save(fig, "figure2_development")


def _gain_panel(ax, path, endpoint):
    gains = _gains(path, endpoint)
    for i, contrast in enumerate(CONTRASTS):
        part = gains[gains.contrast == contrast]
        jitter = np.linspace(-0.12, 0.12, len(part))
        ax.scatter(np.full(len(part), i) + jitter, part.gain, s=11,
                   color=CONTRAST_COLOR[contrast], zorder=3, linewidths=0)
        ax.plot([i - 0.18, i + 0.18], [part.gain.mean(), part.gain.mean()], color=INK, lw=0.8, zorder=4)
    ax.axhline(0, color=GREY, lw=0.6)
    short = {"mean": "Mean", "joint_ridge": "Joint", "rna_ridge": "RNA",
             "totalvi": "totalVI", "v040_abundance_shrinkage": "Shrinkage"}
    ax.set_xticks(range(len(CONTRASTS)), [short[c] for c in CONTRASTS], fontsize=6.5, rotation=25, ha="right")
    ax.set_ylabel("Donor-level gain")
    return gains


def figure3():
    fig = plt.figure(figsize=(174 * MM, 118 * MM), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05])
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, :]), fig.add_subplot(grid[0, 1])]
    gains = _gain_panel(axes[0], ROOT / "runs/cellbridge_v100/gse334503/evaluation/donor_losses.csv", "E2_all_hidden")
    axes[0].set_title("All hidden proteins", loc="left")
    _letter(axes[0], "a")
    _source("figure3a_gains", gains)
    decision = _json(ROOT / "runs/cellbridge_v100/gse334503/evaluation/decision.json")
    _forest(axes[1], decision)
    axes[1].set_title("Fixed sequence", loc="left")
    _letter(axes[1], "b")
    _scatter(axes[2], ROOT / "runs/cellbridge_v100/gse334503_posthoc/predicted_observed/table.csv")
    axes[2].set_title("Primary proteins", loc="left")
    _letter(axes[2], "c")
    _save(fig, "figure3_gse334503")


def figure4():
    fig, axes = plt.subplots(2, 2, figsize=(174 * MM, 132 * MM), constrained_layout=True)
    gains = _gain_panel(axes[0, 0], ROOT / "runs/cellbridge_v100/emtab9357/evaluation/donor_losses.csv", "E2_all_hidden")
    above = int((gains.gain > 1.5).sum())
    axes[0, 0].set_ylim(-0.35, 1.55)
    if above:
        axes[0, 0].text(0.98, 0.96, f"{above} points above 1.5", transform=axes[0, 0].transAxes,
                        ha="right", va="top", fontsize=6, color=GREY)
    axes[0, 0].set_title("All hidden proteins", loc="left")
    _letter(axes[0, 0], "a")
    _source("figure4a_gains", gains)
    decision = _json(ROOT / "runs/cellbridge_v100/emtab9357/evaluation/decision.json")
    _forest(axes[0, 1], decision, mark="H4")
    axes[0, 1].set_title("Fixed sequence", loc="left")
    _letter(axes[0, 1], "b")
    _scatter(axes[1, 0], ROOT / "runs/cellbridge_v100/emtab9357_posthoc/predicted_observed/table.csv")
    axes[1, 0].set_title("Primary proteins", loc="left")
    _letter(axes[1, 0], "c")

    losses = pd.read_csv(ROOT / "runs/cellbridge_v100/emtab9357/evaluation/donor_losses.csv")
    test = losses[(losses.role == "test") & (losses.endpoint == "E2_all_hidden")]
    wide = test.pivot(index="donor", columns="method", values="loss").sort_values("cellbridge")
    ax = axes[1, 1]
    for _, row in wide.iterrows():
        ax.plot([0, 1], [row.cellbridge, row.cellbridge_rna_only], color=PALE, lw=0.7, zorder=1)
    ax.scatter(np.zeros(len(wide)), wide.cellbridge, s=12, color=TEAL, zorder=3, linewidths=0)
    ax.scatter(np.ones(len(wide)), wide.cellbridge_rna_only, s=12, color="#94A3B8", zorder=3, linewidths=0)
    ax.plot([0, 1], [wide.cellbridge.mean(), wide.cellbridge_rna_only.mean()], color=INK, lw=1.2, zorder=4)
    ax.set_xticks([0, 1], ["CellBridge", "RNA only"])
    ax.set_ylabel("Donor standardized MAE")
    ax.set_xlim(-0.25, 1.25)
    ax.set_title("Anchors carry the gain", loc="left")
    _letter(ax, "d")
    _source("figure4d_rna_only", wide.reset_index())
    _save(fig, "figure4_emtab9357")


def figure5():
    fig, axes = plt.subplots(1, 3, figsize=(174 * MM, 72 * MM), constrained_layout=True)
    meta = _json(ROOT / "runs/cellbridge_v100/meta/meta.json")
    rows = [row for row in meta["results"] if row["endpoint"] == "E2_all_hidden"]
    rows = list(reversed(rows))
    ax = axes[0]
    for i, row in enumerate(rows):
        for cohort, marker, dy in (("GSE334503", "o", 0.16), ("E-MTAB-9357", "s", -0.16)):
            part = next(c for c in row["cohorts"] if c["cohort"] == cohort)
            se = np.sqrt(part["variance"])
            color = TEAL if cohort == "GSE334503" else ORANGE
            ax.plot([part["mean"] - 1.96 * se, part["mean"] + 1.96 * se], [i + dy, i + dy], color=color, lw=1.0)
            ax.plot(part["mean"], i + dy, marker, color=color, ms=3.5)
        re = row["random_effects"]
        ax.plot(re["mean"], i, "D", color=INK, ms=3.2, zorder=4)
    ax.axvline(0, color=GREY, lw=0.6)
    ax.set_yticks(range(len(rows)), [CONTRAST_LABEL[r["contrast"]] for r in rows])
    ax.set_xlabel("Gain, all hidden proteins")
    ax.set_title("Two cohorts", loc="left")
    _letter(ax, "a")
    ax.plot([], [], "o", color=TEAL, label="GSE334503")
    ax.plot([], [], "s", color=ORANGE, label="E-MTAB-9357")
    ax.plot([], [], "D", color=INK, label="Random effect")
    ax.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.55, -0.2), ncol=3)

    ax = axes[1]
    methods = [("cellbridge", "CellBridge"), ("v040_abundance_shrinkage", "Shrinkage"), ("totalvi", "totalVI")]
    coverage_rows = []
    width = 0.36
    for j, (method, label) in enumerate(methods):
        for cohort, path, color, shift in (
            ("GSE334503", ROOT / "runs/cellbridge_v100/gse334503/evaluation/decision.json", TEAL, -width / 2),
            ("E-MTAB-9357", ROOT / "runs/cellbridge_v100/emtab9357/evaluation/decision.json", ORANGE, width / 2),
        ):
            conf = _json(path)["conformal"]
            if method not in conf:
                continue
            value = conf[method]["test_coverage_all_hidden"]
            ax.bar(j + shift, value, width=width, color=color)
            coverage_rows.append({"cohort": cohort, "method": label, "coverage": value})
    ax.axhline(0.9, color=GREY, lw=0.7, ls="--")
    ax.set_xticks(range(len(methods)), [label for _, label in methods], fontsize=6.5, rotation=20, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Test coverage")
    ax.set_title("90% intervals", loc="left")
    ax.plot([], [], color=TEAL, lw=4, label="GSE334503")
    ax.plot([], [], color=ORANGE, lw=4, label="E-MTAB-9357")
    ax.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    _letter(ax, "b")
    _source("figure5b_coverage", pd.DataFrame(coverage_rows))

    gse_fit = _json(ROOT / "runs/cellbridge_v100/gse334503/fit/fit.json")
    gse_tv = _json(ROOT / "runs/cellbridge_v100/gse334503/totalvi_log.json")
    em_tv = _json(ROOT / "runs/cellbridge_v100/emtab9357/totalvi_log.json")
    seconds = [
        gse_fit["timing_seconds"]["cellbridge_fit_seconds"],
        gse_tv["fit_seconds"],
        em_tv["fit_seconds"],
    ]
    names = ["CellBridge\nGSE", "totalVI\nGSE", "totalVI\nE-MTAB"]
    ax = axes[2]
    ax.bar(np.arange(3), seconds, color=[TEAL, ORANGE, ORANGE], width=0.62)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(3), names, fontsize=6)
    ax.set_ylabel("Fit time (seconds)")
    ax.set_title("One laptop", loc="left")
    for i, value in enumerate(seconds):
        ax.text(i, value * 1.15, f"{value:.0f}s" if value > 20 else f"{value:.1f}s", ha="center", fontsize=6)
    _letter(ax, "c")
    _source("figure5c_runtime", pd.DataFrame({"fit": names, "seconds": seconds}))
    _save(fig, "figure5_synthesis")


def figure6():
    fig, axes = plt.subplots(1, 3, figsize=(174 * MM, 68 * MM), constrained_layout=True)
    panel = pd.read_csv(ROOT / "runs/cellbridge_v100/open_ablations/panel_curve.csv")
    grouped = panel.groupby("n_anchors").all_mae
    ax = axes[0]
    ax.fill_between(grouped.mean().index, grouped.quantile(0.25), grouped.quantile(0.75), color=TEAL, alpha=0.15, lw=0)
    ax.plot(grouped.mean().index, grouped.mean(), "o-", color=TEAL, ms=4.5)
    ax.set_xticks([0, 1, 2, 4, 8])
    ax.set_xlabel("Number of anchors")
    ax.set_ylabel("Mean standardized MAE")
    ax.set_title("Which antibodies to measure", loc="left")
    _letter(ax, "a")

    shares = pd.read_csv(ROOT / "runs/cellbridge_v100/gse334503_posthoc/contribution_shares/table.csv")
    ax = axes[1]
    ax.hist(shares.anchor_share, bins=16, color=TEAL, edgecolor="white", lw=0.4)
    tops = ax.get_ylim()[1]
    for protein, color, dy in (("CD25", ORANGE, 0.72), ("CD69", PURPLE, 0.92)):
        value = float(shares.loc[shares.protein == protein, "anchor_share"].iloc[0])
        ax.axvline(value, color=color, lw=1.0)
        ax.text(value + 0.004, tops * dy, protein, color=color, fontsize=6.5, ha="left")
    ax.set_xlabel("Anchor share of absolute contributions")
    ax.set_ylabel("Proteins")
    ax.set_title("GSE334503 test donors", loc="left")
    _letter(ax, "b")
    _source("figure6b_shares", shares)

    attr = pd.read_csv(ROOT / "runs/cellbridge_v100/open_ablations/attribution_summary.csv")
    ax = axes[2]
    ax.bar(attr.protein, 100 * attr.top_is_anchor, color=TEAL, width=0.62)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Resamples (%)")
    ax.set_title("Largest feature is an anchor", loc="left")
    _letter(ax, "c")
    _source("figure6c_bootstrap", attr)
    _save(fig, "figure6_panel")


def figure_s1():
    fig, axes = plt.subplots(1, 2, figsize=(174 * MM, 70 * MM), constrained_layout=True)
    for ax, path, title in (
        (axes[0], ROOT / "runs/cellbridge_v100/gse334503_posthoc/per_protein/table.csv", "GSE334503"),
        (axes[1], ROOT / "runs/cellbridge_v100/emtab9357_posthoc/per_protein/table.csv", "E-MTAB-9357"),
    ):
        frame = pd.read_csv(path).sort_values("standardized_mae")
        ax.plot(np.arange(len(frame)), frame.standardized_mae, color=TEAL, lw=1.0)
        primary = frame.primary.to_numpy()
        ax.scatter(np.flatnonzero(primary), frame.standardized_mae.to_numpy()[primary],
                   color=ORANGE, s=14, zorder=3, label="CD25 or CD69")
        ax.set_xlabel("Proteins, ordered by error")
        ax.set_ylabel("Standardized MAE")
        ax.set_title(title, loc="left")
        ax.legend(frameon=False, fontsize=6.5)
    _save(fig, "figureS1_per_protein")


def figure_s2():
    fig, axes = plt.subplots(1, 2, figsize=(174 * MM, 62 * MM), constrained_layout=True)
    sim = _json(ROOT / "runs/cellbridge_v100/open_ablations/simulation.json")
    part = [row for row in sim["metacells"] if row["sigma"] == 2.0]
    order = ["cells", "kmeans", "oracle_states"]
    labels = ["Single cells", "K-means", "Shared state"]
    vals = [next(row["slope_norm_ratio"] for row in part if row["grouping"] == name) for name in order]
    axes[0].bar(np.arange(3), vals, color=[GREY, PALE, TEAL], width=0.62)
    axes[0].set_xticks(np.arange(3), labels)
    axes[0].set_ylabel("Recovered slope, relative to truth")
    axes[0].set_title("Known slope, noisy features", loc="left")
    _letter(axes[0], "a")
    scale = [row for row in sim["scaling"] if "n_cells_per_arm" in row]
    cells = [row["n_cells_per_arm"] * 2 * row["n_donors"] for row in scale]
    seconds = [row["build_seconds"] + row["fit_seconds"] for row in scale]
    axes[1].plot(cells, seconds, "o-", color=TEAL, ms=4)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Cells")
    axes[1].set_ylabel("Seconds")
    axes[1].set_title("Building the statistics", loc="left")
    _letter(axes[1], "b")
    _save(fig, "figureS2_simulation")


def figure_s3():
    fig, axes = plt.subplots(1, 2, figsize=(174 * MM, 58 * MM), constrained_layout=True)
    for ax, path, title in (
        (axes[0], ROOT / "runs/cellbridge_v100/gse334503/totalvi_log.json", "GSE334503"),
        (axes[1], ROOT / "runs/cellbridge_v100/emtab9357/totalvi_log.json", "E-MTAB-9357"),
    ):
        hist = _json(path)["history"]
        if "elbo_train" in hist:
            ax.plot(np.arange(1, len(hist["elbo_train"]) + 1), hist["elbo_train"], color=TEAL, lw=1.0, label="Train")
        if "elbo_validation" in hist:
            ax.plot(np.arange(1, len(hist["elbo_validation"]) + 1), hist["elbo_validation"],
                    color=ORANGE, lw=1.0, label="Validation")
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("ELBO")
        ax.set_title(title, loc="left")
        ax.legend(frameon=False, fontsize=6.5)
    _save(fig, "figureS3_totalvi")


def figure_s4():
    gene = pd.read_csv(ROOT / "runs/cellbridge_v100/open_ablations/gene_curve.csv")
    fig, ax = plt.subplots(figsize=(85 * MM, 58 * MM), constrained_layout=True)
    if "n_genes" in gene.columns and "all_mae" in gene.columns:
        grouped = gene.groupby("n_genes").all_mae
        ax.plot(grouped.mean().index, grouped.mean().values, "o-", color=TEAL, ms=3.5)
        ax.set_xlabel("Genes")
        ax.set_ylabel("Mean standardized MAE")
    ax.set_title("Gene count, reduced grid", loc="left")
    _save(fig, "figureS4_genes")


if __name__ == "__main__":
    _style()
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    figure6()
    figure_s1()
    figure_s2()
    figure_s3()
    figure_s4()
    print("figures written", OUT)
