"""
Permutation feature importance for the final full_geometry_dataset models.

Uses: data/share/needle_features_v1.csv
Feature set: all_geometry_no_availability_plus_clinical (14 features)
  biopsy_geometry (6) + target_geometry_no_availability (4) + clinical (4)
  target_mesh_available is excluded.

Endpoints and models analysed:
  GG2+  (binary_label_int):         logistic_regression, xgboost (if available)
  GG3+  (binary_label_gg3plus_int): logistic_regression

Method:
  - Model is fit on the train split.
  - Permutation importance is computed on the test split.
  - Each feature is permuted independently for N_REPEATS iterations.
  - Importance = mean decrease in test ROC-AUC (or PR-AUC) after permuting the
    feature.  Positive values mean the feature helps; near-zero or negative
    values indicate no reliable contribution.

Outputs
-------
    reports/feature_importance.csv
    reports/feature_importance.md

Usage
-----
    python src/analyze_feature_importance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score

# ── Import shared utilities from the main experiment script ───────────────────

REPO_ROOT   = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_shareable_tabular_experiments import (  # noqa: E402
    BIOPSY_GEOMETRY_FEATURES,
    CLINICAL_FEATURES,
    DATA_PATH,
    HAS_XGBOOST,
    RANDOM_STATE,
    SPLIT_COL,
    TARGET_GEOMETRY_NO_AVAILABILITY_FEATURES,
    build_model,
    load_dataset,
    prepare_features,
    resolve_feature_list,
)

# ── Constants ──────────────────────────────────────────────────────────────────

N_REPEATS       = 50
FEATURE_GROUPS  = ["biopsy_geometry", "target_geometry_no_availability", "clinical"]
EXPERIMENT_NAME = "all_geometry_no_availability_plus_clinical"

# Ordered list of (label_col, model_name) to run.
ENDPOINT_MODELS: list[tuple[str, str]] = [
    ("binary_label_int",         "logistic_regression"),
    ("binary_label_int",         "xgboost"),
    ("binary_label_gg3plus_int", "logistic_regression"),
]

ENDPOINT_LABELS: dict[str, str] = {
    "binary_label_int":         "GG2+ / csPCa",
    "binary_label_gg3plus_int": "GG3+ / high-grade",
}

# Feature-category sets used only for interpretation text.
_TARGET_GEO_SET   = set(TARGET_GEOMETRY_NO_AVAILABILITY_FEATURES)
_PROSTATE_GEO_SET = set(BIOPSY_GEOMETRY_FEATURES)
_CLINICAL_SET     = set(CLINICAL_FEATURES)


# ── Core analysis ──────────────────────────────────────────────────────────────

def run_permutation_importance(
    label_col: str,
    model_name: str,
    df: pd.DataFrame,
    feature_cols: list[str],
    n_repeats: int = N_REPEATS,
    random_state: int = RANDOM_STATE,
) -> list[dict]:
    """
    Fit model on the train split, compute permutation importance on the test split.

    Uses sklearn.inspection.permutation_importance, which permutes the raw input
    features (pre-preprocessing) so the full pipeline — imputation, optional
    scaling, and the classifier — sees the permuted data each time.

    Returns one dict per feature, or an empty list if the model is unavailable
    or no usable features remain.
    """
    train = df[df[SPLIT_COL] == "train"]
    test  = df[df[SPLIT_COL] == "test"]

    # Same all-NaN guard as run_experiment in run_shareable_tabular_experiments.py
    usable_cols = [c for c in feature_cols if train[c].notna().any()]
    if not usable_cols:
        print(f"  [{label_col}/{model_name}] no usable features — skipping.")
        return []

    X_train, y_train = train[usable_cols], train[label_col]
    X_test,  y_test  = test[usable_cols],  test[label_col]

    pipeline = build_model(model_name, usable_cols, y_train)
    if pipeline is None:
        print(f"  [{label_col}/{model_name}] model not available — skipping.")
        return []

    print(f"  [{label_col}/{model_name}] fitting on {len(X_train):,} train samples …")
    pipeline.fit(X_train, y_train)

    # Baseline metrics on the unmodified test set
    test_probs   = pipeline.predict_proba(X_test)[:, 1]
    baseline_roc = float(roc_auc_score(y_test, test_probs))
    baseline_pr  = float(average_precision_score(y_test, test_probs))
    print(f"    baseline ROC-AUC={baseline_roc:.3f}  PR-AUC={baseline_pr:.3f}")
    print(f"    running permutation importance ({n_repeats} repeats) …")

    # sklearn string scorers: "roc_auc" and "average_precision"
    perm_roc = permutation_importance(
        pipeline, X_test, y_test,
        scoring="roc_auc",
        n_repeats=n_repeats, random_state=random_state, n_jobs=1,
    )
    perm_pr = permutation_importance(
        pipeline, X_test, y_test,
        scoring="average_precision",
        n_repeats=n_repeats, random_state=random_state, n_jobs=1,
    )

    rows = []
    for fi, feat in enumerate(usable_cols):
        rows.append({
            "label_col":         label_col,
            "model":             model_name,
            "feature":           feat,
            "baseline_roc_auc":  baseline_roc,
            "baseline_pr_auc":   baseline_pr,
            "roc_auc_drop_mean": float(perm_roc.importances_mean[fi]),
            "roc_auc_drop_std":  float(perm_roc.importances_std[fi]),
            "pr_auc_drop_mean":  float(perm_pr.importances_mean[fi]),
            "pr_auc_drop_std":   float(perm_pr.importances_std[fi]),
        })
    return rows


# ── Interpretation ─────────────────────────────────────────────────────────────

def _interpret(df: pd.DataFrame) -> list[str]:
    """
    Generate four data-driven interpretation bullets from one endpoint/model slice.

    Covers: target geometry dominance, clinical feature contribution (accounting
    for PSA-variable correlation), prostate geometry signal, and result stability.
    """
    df_roc = df.sort_values("roc_auc_drop_mean", ascending=False).reset_index(drop=True)

    # ── 1. Target geometry dominance ─────────────────────────────────────────
    pos_total = float(df["roc_auc_drop_mean"].clip(lower=0).sum())
    tgt_pos   = float(df[df["feature"].isin(_TARGET_GEO_SET)]["roc_auc_drop_mean"].clip(lower=0).sum())
    tgt_pct   = tgt_pos / pos_total * 100 if pos_total > 0 else 0.0

    if tgt_pct > 50:
        geo_bullet = (
            f"**Target geometry dominates**: target-trajectory features contribute "
            f"~{tgt_pct:.0f}% of the total positive ROC-AUC importance, consistent "
            f"with proximity to the MRI lesion being the primary predictive signal."
        )
    else:
        geo_bullet = (
            f"**Target geometry contributes** but does not dominate alone "
            f"(~{tgt_pct:.0f}% of positive ROC-AUC importance), suggesting other "
            f"feature groups add complementary signal."
        )

    # ── 2. Clinical features — PSA-related variables (correlated group) ───────
    # psa_ng_ml, log_psa_ng_ml, prostate_volume_cc, and psa_density are correlated.
    # Permutation importance can distribute or mask importance across them, so we
    # assess the best-ranked PSA variable (psa_density or log_psa_ng_ml) rather
    # than any single predictor in isolation.
    _PSA_KEY = {"psa_density", "log_psa_ng_ml"}
    top_half = max(1, len(df_roc) // 2)

    best_psa_idxs = df_roc.index[df_roc["feature"].isin(_PSA_KEY)].tolist()
    best_psa_rank = best_psa_idxs[0] + 1 if best_psa_idxs else None
    best_psa_feat = df_roc.loc[best_psa_idxs[0], "feature"] if best_psa_idxs else None
    best_psa_drop = float(df_roc.loc[best_psa_idxs[0], "roc_auc_drop_mean"]) if best_psa_idxs else 0.0

    psa_in_top = best_psa_rank is not None and best_psa_rank <= top_half and best_psa_drop > 0

    psa_density_row  = df_roc[df_roc["feature"] == "psa_density"]
    psa_density_drop = float(psa_density_row.iloc[0]["roc_auc_drop_mean"]) if len(psa_density_row) else 0.0

    if psa_in_top:
        psa_bullet = (
            f"**PSA-related variables contribute meaningfully** "
            f"(`{best_psa_feat}` ranked #{best_psa_rank}, ROC-AUC drop = {best_psa_drop:.4f}). "
            f"Clinical signal is present, though importance may be distributed across "
            f"correlated features (psa_ng_ml, log_psa_ng_ml, psa_density). "
            f"Low individual importance for any one of these should not be interpreted "
            f"as absence of clinical value."
        )
    else:
        psa_bullet = (
            f"**Clinical signal is present but distributed** across correlated PSA-related "
            f"variables (psa_ng_ml, log_psa_ng_ml, psa_density). "
            f"Low permutation importance for psa_density "
            f"(ROC-AUC drop = {psa_density_drop:.4f}) in this multivariate model should "
            f"not be interpreted as absence of clinical value, since univariate analysis "
            f"showed psa_density to be predictive."
        )

    # ── 3. Prostate geometry signal ───────────────────────────────────────────
    bio_drops = df[df["feature"].isin(_PROSTATE_GEO_SET)]["roc_auc_drop_mean"]
    bio_max   = float(bio_drops.max()) if len(bio_drops) > 0 else 0.0
    bio_feat  = df[df["feature"].isin(_PROSTATE_GEO_SET)].sort_values(
        "roc_auc_drop_mean", ascending=False
    ).iloc[0]["feature"] if len(bio_drops) > 0 else ""

    if bio_max > 0.005:
        bio_bullet = (
            f"**Prostate/biopsy geometry adds signal** (best feature `{bio_feat}`, "
            f"ROC-AUC drop = {bio_max:.4f}), indicating that needle placement relative "
            f"to the prostate adds information beyond target proximity."
        )
    else:
        bio_bullet = (
            f"**Prostate/biopsy geometry adds little signal** (max ROC-AUC drop "
            f"= {bio_max:.4f}), suggesting these features are largely redundant "
            f"given target geometry and clinical features in this model."
        )

    # ── 4. Stability ─────────────────────────────────────────────────────────
    top5 = df_roc.head(5)
    n_noisy = int((top5["roc_auc_drop_std"] > top5["roc_auc_drop_mean"].abs()).sum())

    if n_noisy >= 2:
        stab_bullet = (
            f"**Stability: noisy** — {n_noisy} of the top-5 features have std > mean "
            f"importance, suggesting individual values should be interpreted cautiously. "
            f"Feature rankings are more reliable than absolute drop magnitudes."
        )
    elif n_noisy == 1:
        stab_bullet = (
            "**Stability: moderate** — most top features are stable across repeats, "
            "though one shows std > mean importance. Overall rankings should be reliable."
        )
    else:
        stab_bullet = (
            "**Stability: good** — for all top-5 features, std < mean importance, "
            "indicating the importance rankings are stable across permutation repeats."
        )

    return [f"- {b}" for b in [geo_bullet, psa_bullet, bio_bullet, stab_bullet]] + [""]


# ── Reporting ──────────────────────────────────────────────────────────────────

def _f(v: float) -> str:
    return f"{v:.4f}"


def write_report(all_rows: list[dict], csv_path: Path, md_path: Path) -> None:
    if not all_rows:
        print("No results to write.")
        return

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(csv_path, index=False)
    print(f"Saved -> {csv_path.relative_to(REPO_ROOT)}")

    lines: list[str] = [
        "# Feature Importance — Permutation Analysis",
        "",
        f"Feature set: `{EXPERIMENT_NAME}`  ",
        "(biopsy geometry ×6 + target geometry no availability ×4 + clinical ×4 = 14 features)  ",
        f"Permutation repeats: {N_REPEATS} · random seed: {RANDOM_STATE}",
        "",
        "**Importance** = mean decrease in test metric when a single feature is randomly shuffled.  ",
        "Positive values indicate a useful feature; near-zero or negative values suggest no reliable contribution.  ",
        "Each section shows the same features sorted first by ROC-AUC drop, then by PR-AUC drop.",
        "",
        "> **Correlation caveat:** permutation importance should be interpreted cautiously "
        "when features are correlated; importance may be shared or masked among correlated "
        "predictors. The four clinical features (psa_ng_ml, log_psa_ng_ml, "
        "prostate_volume_cc, psa_density) are correlated — any single feature may appear "
        "unimportant while the group as a whole contributes meaningful signal. "
        "See the per-section interpretation for guidance.",
        "",
    ]

    for label_col, model_name in ENDPOINT_MODELS:
        rows = [r for r in all_rows if r["label_col"] == label_col and r["model"] == model_name]
        if not rows:
            continue

        ep_label    = ENDPOINT_LABELS.get(label_col, label_col)
        baseline_roc = rows[0]["baseline_roc_auc"]
        baseline_pr  = rows[0]["baseline_pr_auc"]

        lines += [
            f"---",
            "",
            f"## {ep_label} — {model_name}",
            "",
            f"Baseline test ROC-AUC: **{baseline_roc:.3f}**  ·  "
            f"Baseline test PR-AUC: **{baseline_pr:.3f}**",
            "",
        ]

        df = pd.DataFrame(rows)

        # ── Table sorted by ROC-AUC drop ──────────────────────────────────────
        df_roc = df.sort_values("roc_auc_drop_mean", ascending=False).reset_index(drop=True)
        lines += [
            "### Sorted by ROC-AUC importance",
            "",
            "| # | Feature | ROC-AUC drop mean | ± std | PR-AUC drop mean | ± std |",
            "|---|---|---|---|---|---|",
        ]
        for i, (_, r) in enumerate(df_roc.iterrows(), 1):
            lines.append(
                f"| {i} | `{r['feature']}` "
                f"| {_f(r['roc_auc_drop_mean'])} "
                f"| ±{_f(r['roc_auc_drop_std'])} "
                f"| {_f(r['pr_auc_drop_mean'])} "
                f"| ±{_f(r['pr_auc_drop_std'])} |"
            )
        lines.append("")

        # ── Table sorted by PR-AUC drop ───────────────────────────────────────
        df_pr = df.sort_values("pr_auc_drop_mean", ascending=False).reset_index(drop=True)
        lines += [
            "### Sorted by PR-AUC importance",
            "",
            "| # | Feature | PR-AUC drop mean | ± std | ROC-AUC drop mean | ± std |",
            "|---|---|---|---|---|---|",
        ]
        for i, (_, r) in enumerate(df_pr.iterrows(), 1):
            lines.append(
                f"| {i} | `{r['feature']}` "
                f"| {_f(r['pr_auc_drop_mean'])} "
                f"| ±{_f(r['pr_auc_drop_std'])} "
                f"| {_f(r['roc_auc_drop_mean'])} "
                f"| ±{_f(r['roc_auc_drop_std'])} |"
            )
        lines.append("")

        # ── Interpretation ─────────────────────────────────────────────────────
        lines += ["### Interpretation", ""]
        lines += _interpret(df)

    md_path.write_text("\n".join(lines) + "\n")
    print(f"Saved -> {md_path.relative_to(REPO_ROOT)}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== Feature importance — permutation analysis ===\n")
    print(f"Feature groups : {FEATURE_GROUPS}")
    print(f"Repeats        : {N_REPEATS}")
    if not HAS_XGBOOST:
        print("Note: xgboost not installed — XGBoost runs will be skipped.")

    all_rows: list[dict] = []

    for label_col, model_name in ENDPOINT_MODELS:
        if model_name == "xgboost" and not HAS_XGBOOST:
            print(f"\n[{label_col}/{model_name}] xgboost not available — skipping.")
            continue

        print(f"\n[{label_col} / {model_name}]")
        df, _ = load_dataset(DATA_PATH, target_col=label_col)
        df = prepare_features(df)

        feature_cols = resolve_feature_list(FEATURE_GROUPS, set(df.columns))
        print(f"  {len(feature_cols)} features: {feature_cols}")

        rows = run_permutation_importance(label_col, model_name, df, feature_cols)
        all_rows.extend(rows)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_report(
        all_rows,
        csv_path=REPORTS_DIR / "feature_importance.csv",
        md_path=REPORTS_DIR  / "feature_importance.md",
    )


if __name__ == "__main__":
    main()
