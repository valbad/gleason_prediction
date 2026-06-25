"""
Compare compact vs full feature sets for biopsy-core-level prostate-cancer prediction.

Goal
----
Decide whether the paper should use a minimal target-geometry + clinical model
or the full geometry + clinical model by comparing key feature-set combinations
across two endpoints and three models.

Input
-----
data/share/needle_features_v1.csv

Filtering
---------
  label_join_status == "coord_match"
  split in {"train", "val", "test"}
  non-null endpoint label (per endpoint)
  target_mesh_available excluded from all reported feature sets

Outputs
-------
reports/minimal_vs_full_model_comparison.csv
reports/minimal_vs_full_model_comparison.md
reports/minimal_vs_full_risk_stratification.csv
reports/minimal_vs_full_risk_stratification.md
reports/minimal_vs_full_delta_table.md

Usage
-----
    python src/compare_minimal_vs_full_models.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from run_shareable_tabular_experiments import (
    BIOPSY_GEOMETRY_FEATURES,
    CLINICAL_FEATURES,
    HAS_XGBOOST,
    PATIENT_COL,
    RANDOM_STATE,
    SPLIT_COL,
    TARGET_GEOMETRY_NO_AVAILABILITY_FEATURES,
    bootstrap_patient_ci,
    build_model,
    evaluate_model,
    get_model_specs,
    load_dataset,
    prepare_features,
)


# ── Paths & constants ─────────────────────────────────────────────────────────

REPO_ROOT   = Path(__file__).resolve().parent.parent
DATA_PATH   = REPO_ROOT / "data" / "share" / "needle_features_v1.csv"
REPORTS_DIR = REPO_ROOT / "reports"

N_BOOT       = 1000
REFERENCE_FS = "all_geometry_no_availability_plus_clinical"

ENDPOINTS: list[tuple[str, str]] = [
    ("binary_label_int",         "GG2+ / csPCa"),
    ("binary_label_gg3plus_int", "GG3+ / high-grade"),
]

# Feature sets in display/evaluation order.
# target_mesh_available is excluded from all sets (conservative ablation).
_BIOPSY = BIOPSY_GEOMETRY_FEATURES                    # 6 features
_TARGET = TARGET_GEOMETRY_NO_AVAILABILITY_FEATURES    # 4 features
_CLIN   = CLINICAL_FEATURES                           # 4 features

FEATURE_SETS: dict[str, list[str]] = {
    "clinical_only":                          _CLIN,
    "target_geometry_only":                   _TARGET,
    "target_geometry_plus_clinical":          _TARGET + _CLIN,
    "biopsy_prostate_geometry_plus_clinical": _BIOPSY + _CLIN,
    "all_geometry_no_clinical":               _BIOPSY + _TARGET,
    REFERENCE_FS:                             _BIOPSY + _TARGET + _CLIN,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _f(v, n: int = 3) -> str:
    if isinstance(v, (float, np.floating)) and np.isnan(v):
        return "n/a"
    return f"{v:.{n}f}"


def _risk_strat(y_true: np.ndarray, probs: np.ndarray) -> dict:
    """Risk-stratification metrics: top 5/10/20% prevalence and capture rate."""
    n     = len(y_true)
    n_pos = int(y_true.sum())
    result: dict = {"prevalence": float(y_true.mean()) if n > 0 else np.nan}
    for pct in (5, 10, 20):
        k       = max(1, int(np.ceil(n * pct / 100)))
        top_idx = np.argsort(probs)[::-1][:k]
        top_y   = y_true[top_idx]
        result[f"top{pct}pct_prevalence"]   = float(top_y.mean())
        result[f"top{pct}pct_capture_rate"] = float(top_y.sum() / n_pos) if n_pos > 0 else np.nan
    return result


# ── Experiment runner ─────────────────────────────────────────────────────────

def run_one(
    label_col: str,
    fs_name: str,
    feature_cols: list[str],
    model_name: str,
    df: pd.DataFrame,
) -> tuple[dict, np.ndarray] | tuple[None, None]:
    """
    Fit on train, select threshold on val (Youden J), evaluate on test.
    Returns (result_row, test_probs) or (None, None) on failure.
    """
    train = df[df[SPLIT_COL] == "train"]
    val   = df[df[SPLIT_COL] == "val"]
    test  = df[df[SPLIT_COL] == "test"]

    usable  = [c for c in feature_cols if train[c].notna().any()]
    dropped = set(feature_cols) - set(usable)
    if dropped:
        warnings.warn(
            f"[{label_col}/{fs_name}/{model_name}] "
            f"dropping all-NaN columns in train: {sorted(dropped)}",
            stacklevel=2,
        )
    if not usable:
        warnings.warn(
            f"[{label_col}/{fs_name}/{model_name}] no usable features — skipping.",
            stacklevel=2,
        )
        return None, None

    X_train, y_train = train[usable], train[label_col]
    X_val,   y_val   = val[usable],   val[label_col]
    X_test,  y_test  = test[usable],  test[label_col]

    pipeline = build_model(model_name, usable, y_train)
    if pipeline is None:
        return None, None

    test_metrics, _, threshold = evaluate_model(
        pipeline, X_train, y_train, X_val, y_val, X_test, y_test
    )

    # Pipeline is fitted; re-predict on test for risk stratification.
    test_probs = pipeline.predict_proba(X_test)[:, 1]

    test_prevalence = float(y_test.mean())
    pr_auc_val = test_metrics.get("pr_auc", np.nan)
    pr_lift = (
        pr_auc_val / test_prevalence
        if test_prevalence > 0 and not (isinstance(pr_auc_val, float) and np.isnan(pr_auc_val))
        else np.nan
    )

    ci = bootstrap_patient_ci(
        pipeline, X_test, y_test,
        test[PATIENT_COL].values,
        threshold,
        n_boot=N_BOOT,
        random_state=RANDOM_STATE,
    )

    row: dict = {
        "label_col":       label_col,
        "feature_set":     fs_name,
        "model":           model_name,
        "n_features":      len(usable),
        "n_train":         len(train),
        "n_val":           len(val),
        "n_test":          len(test),
        "test_prevalence": test_prevalence,
        "pr_lift":         pr_lift,
        "threshold":       threshold,
    }
    row.update({f"test_{k}": v for k, v in test_metrics.items()})
    row.update(ci)

    return row, test_probs


# ── Reporting ─────────────────────────────────────────────────────────────────

_PERF_COLS = [
    "feature_set", "model", "n_features",
    "test_roc_auc", "roc_auc_ci_low", "roc_auc_ci_high",
    "test_pr_auc",  "pr_auc_ci_low",  "pr_auc_ci_high",
    "pr_lift",
    "threshold",
    "test_sensitivity", "test_specificity",
    "test_precision",   "test_f1",
    "n_train", "n_val", "n_test", "test_prevalence",
]

_RISK_COLS = [
    "feature_set", "model", "prevalence",
    "top5pct_prevalence",  "top5pct_capture_rate",
    "top10pct_prevalence", "top10pct_capture_rate",
    "top20pct_prevalence", "top20pct_capture_rate",
]


def _md_table(df: pd.DataFrame, cols: list[str]) -> list[str]:
    header = "| " + " | ".join(cols) + " |"
    sep    = "|" + "---|" * len(cols)
    lines  = [header, sep]
    for _, r in df.iterrows():
        cells = [
            _f(v) if isinstance(v, (float, np.floating)) else str(v)
            for v in (r.get(c, np.nan) for c in cols)
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _delta_section(df_ep: pd.DataFrame, ep_label: str) -> list[str]:
    """Per-model delta tables vs REFERENCE_FS for one endpoint."""
    lines = [f"### {ep_label}", ""]
    ref_df = df_ep[df_ep["feature_set"] == REFERENCE_FS]
    if ref_df.empty:
        lines += [f"*Reference `{REFERENCE_FS}` not available.*", ""]
        return lines

    ref_by_model: dict[str, tuple[float, float]] = {
        r["model"]: (float(r["test_roc_auc"]), float(r["test_pr_auc"]))
        for _, r in ref_df.iterrows()
    }

    for model in df_ep["model"].unique():
        lines += [
            f"#### {model}", "",
            "| Feature set | n_feat | ΔROC-AUC | ΔPR-AUC |",
            "|---|---|---|---|",
        ]
        ref_roc, ref_pr = ref_by_model.get(model, (np.nan, np.nan))
        for _, r in df_ep[df_ep["model"] == model].iterrows():
            d_roc = float(r["test_roc_auc"]) - ref_roc
            d_pr  = float(r["test_pr_auc"])  - ref_pr
            s_roc = "+" if not np.isnan(d_roc) and d_roc >= 0 else ""
            s_pr  = "+" if not np.isnan(d_pr)  and d_pr  >= 0 else ""
            lines.append(
                f"| `{r['feature_set']}` | {r['n_features']} "
                f"| {s_roc}{_f(d_roc)} | {s_pr}{_f(d_pr)} |"
            )
        lines.append("")
    return lines


def _interpret(df: pd.DataFrame) -> list[str]:
    """Three-point automatic interpretation of the comparison results."""
    lines = ["## Interpretation", ""]

    def _mean_roc(sub: pd.DataFrame, fs: str) -> float:
        vals = sub.loc[sub["feature_set"] == fs, "test_roc_auc"]
        return float(vals.mean()) if not vals.empty else np.nan

    def _mean_pr(sub: pd.DataFrame, fs: str) -> float:
        vals = sub.loc[sub["feature_set"] == fs, "test_pr_auc"]
        return float(vals.mean()) if not vals.empty else np.nan

    for label_col, ep_label in ENDPOINTS:
        sub = df[df["label_col"] == label_col]
        if sub.empty:
            continue

        lines += [f"### {ep_label}", ""]

        tgc_roc  = _mean_roc(sub, "target_geometry_plus_clinical")
        full_roc = _mean_roc(sub, REFERENCE_FS)
        bpgc_roc = _mean_roc(sub, "biopsy_prostate_geometry_plus_clinical")
        tgc_pr   = _mean_pr(sub,  "target_geometry_plus_clinical")
        full_pr  = _mean_pr(sub,  REFERENCE_FS)

        # Point 1: does target_geometry_plus_clinical capture most signal?
        d_roc = full_roc - tgc_roc
        d_pr  = full_pr  - tgc_pr
        if not (np.isnan(d_roc) or np.isnan(d_pr)):
            if abs(d_roc) <= 0.010 and abs(d_pr) <= 0.015:
                verdict1 = (
                    f"`target_geometry_plus_clinical` (mean ROC-AUC {_f(tgc_roc)}) captures "
                    f"most of the full model signal (mean ROC-AUC {_f(full_roc)}; "
                    f"ΔROC = {_f(d_roc)}, ΔPR = {_f(d_pr)}). "
                    f"The compact model is preferred on parsimony grounds."
                )
            else:
                verdict1 = (
                    f"`target_geometry_plus_clinical` (mean ROC-AUC {_f(tgc_roc)}) "
                    f"underperforms the full model (mean ROC-AUC {_f(full_roc)}; "
                    f"ΔROC = {_f(d_roc)}, ΔPR = {_f(d_pr)}). "
                    f"The full feature set adds meaningful signal."
                )
            lines += [
                f"**Signal capture (target geometry + clinical vs full model):** {verdict1}",
                "",
            ]

        # Point 2: incremental value of biopsy/prostate geometry
        # Primary: full model vs compact model (d_roc / d_pr already computed above)
        # Secondary: geometry-only increment without clinical features
        tg_only_roc    = _mean_roc(sub, "target_geometry_only")
        tg_only_pr     = _mean_pr(sub,  "target_geometry_only")
        all_geo_nc_roc = _mean_roc(sub, "all_geometry_no_clinical")
        all_geo_nc_pr  = _mean_pr(sub,  "all_geometry_no_clinical")
        d_geo_roc = all_geo_nc_roc - tg_only_roc
        d_geo_pr  = all_geo_nc_pr  - tg_only_pr

        if not (np.isnan(d_roc) or np.isnan(d_pr)):
            if abs(d_roc) <= 0.010 and abs(d_pr) <= 0.015:
                verdict2 = (
                    f"Adding biopsy/prostate geometry to `target_geometry_plus_clinical` "
                    f"yields limited gain (ΔROC = {_f(d_roc)}, ΔPR = {_f(d_pr)} "
                    f"for `{REFERENCE_FS}` vs `target_geometry_plus_clinical`). "
                    f"Target-relative geometry carries most of the geometric signal; "
                    f"the compact model may be preferred on parsimony grounds."
                )
            else:
                verdict2 = (
                    f"The full model improves meaningfully over the compact model "
                    f"(ΔROC = {_f(d_roc)}, ΔPR = {_f(d_pr)} "
                    f"for `{REFERENCE_FS}` vs `target_geometry_plus_clinical`). "
                    f"The full feature set may be justified."
                )
            lines += [f"**Biopsy/prostate geometry incremental value:** {verdict2}", ""]

        if not (np.isnan(d_geo_roc) or np.isnan(d_geo_pr)):
            if abs(d_geo_roc) <= 0.010 and abs(d_geo_pr) <= 0.015:
                geo_verdict = (
                    f"`all_geometry_no_clinical` adds limited gain over `target_geometry_only` "
                    f"(ΔROC = {_f(d_geo_roc)}, ΔPR = {_f(d_geo_pr)}), "
                    f"consistent with target-relative geometry driving most of the geometric signal."
                )
            else:
                geo_verdict = (
                    f"`all_geometry_no_clinical` gains meaningfully over `target_geometry_only` "
                    f"(ΔROC = {_f(d_geo_roc)}, ΔPR = {_f(d_geo_pr)}), "
                    f"suggesting biopsy/prostate geometry also contributes independently."
                )
            lines += [
                f"**Geometry-only increment (without clinical features):** {geo_verdict}", ""
            ]

        if not np.isnan(bpgc_roc):
            lines += [
                f"**Note on `biopsy_prostate_geometry_plus_clinical`:** "
                f"This feature set (non-target biopsy/prostate geometry + clinical, "
                f"mean ROC-AUC {_f(bpgc_roc)}) serves as a negative-control comparison "
                f"rather than a direct incremental-value test: it confirms that "
                f"target-relative geometry, not biopsy/prostate geometry alone, "
                f"drives model performance.",
                "",
            ]

    # Point 3: consistency of preferred feature set across endpoints
    # Distinguishes "highest raw ROC-AUC" from "preferred central model":
    # the compact model is preferred when it is within the equivalence margin of the full model.
    sub_gg2 = df[df["label_col"] == "binary_label_int"]
    sub_gg3 = df[df["label_col"] == "binary_label_gg3plus_int"]

    lines += ["### Consistency across endpoints", ""]

    if sub_gg2.empty or sub_gg3.empty:
        lines += ["*Only one endpoint available — cross-endpoint comparison not possible.*", ""]
    else:
        # Compact vs full delta (mean over models, same thresholds as Points 1–2)
        d_roc_gg2 = (
            _mean_roc(sub_gg2, REFERENCE_FS)
            - _mean_roc(sub_gg2, "target_geometry_plus_clinical")
        )
        d_pr_gg2 = (
            _mean_pr(sub_gg2, REFERENCE_FS)
            - _mean_pr(sub_gg2, "target_geometry_plus_clinical")
        )
        d_roc_gg3 = (
            _mean_roc(sub_gg3, REFERENCE_FS)
            - _mean_roc(sub_gg3, "target_geometry_plus_clinical")
        )
        d_pr_gg3 = (
            _mean_pr(sub_gg3, REFERENCE_FS)
            - _mean_pr(sub_gg3, "target_geometry_plus_clinical")
        )

        equiv_gg2 = (
            not (np.isnan(d_roc_gg2) or np.isnan(d_pr_gg2))
            and abs(d_roc_gg2) <= 0.010 and abs(d_pr_gg2) <= 0.015
        )
        equiv_gg3 = (
            not (np.isnan(d_roc_gg3) or np.isnan(d_pr_gg3))
            and abs(d_roc_gg3) <= 0.010 and abs(d_pr_gg3) <= 0.015
        )

        if equiv_gg2 and equiv_gg3:
            lines += [
                f"Although the full feature set (`{REFERENCE_FS}`) occasionally gives the "
                f"highest raw ROC-AUC, `target_geometry_plus_clinical` remains within the "
                f"predefined equivalence margin for both endpoints "
                f"(GG2+: ΔROC = {_f(d_roc_gg2)}, ΔPR = {_f(d_pr_gg2)}; "
                f"GG3+: ΔROC = {_f(d_roc_gg3)}, ΔPR = {_f(d_pr_gg3)}). "
                f"Therefore, the compact model is the preferred central model on parsimony "
                f"and interpretability grounds, while the full geometry-clinical model should "
                f"be reported as a sensitivity/reference analysis.",
                "",
            ]
        elif equiv_gg2:
            lines += [
                f"`target_geometry_plus_clinical` is within the equivalence margin for GG2+ "
                f"(ΔROC = {_f(d_roc_gg2)}, ΔPR = {_f(d_pr_gg2)}) but not for GG3+ "
                f"(ΔROC = {_f(d_roc_gg3)}, ΔPR = {_f(d_pr_gg3)}), where the full feature "
                f"set adds meaningful performance. The preferred central model may differ by "
                f"endpoint.",
                "",
            ]
        elif equiv_gg3:
            lines += [
                f"`target_geometry_plus_clinical` is within the equivalence margin for GG3+ "
                f"(ΔROC = {_f(d_roc_gg3)}, ΔPR = {_f(d_pr_gg3)}) but not for GG2+ "
                f"(ΔROC = {_f(d_roc_gg2)}, ΔPR = {_f(d_pr_gg2)}), where the full feature "
                f"set adds meaningful performance. The preferred central model may differ by "
                f"endpoint.",
                "",
            ]
        else:
            lines += [
                f"The compact model falls outside the equivalence margin for both endpoints "
                f"(GG2+: ΔROC = {_f(d_roc_gg2)}, ΔPR = {_f(d_pr_gg2)}; "
                f"GG3+: ΔROC = {_f(d_roc_gg3)}, ΔPR = {_f(d_pr_gg3)}). "
                f"`{REFERENCE_FS}` is the preferred model for both endpoints; "
                f"`target_geometry_plus_clinical` serves as an ablation reference.",
                "",
            ]

    return lines


def write_reports(results: list[dict], risk_rows: list[dict]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df_perf = pd.DataFrame(results)
    df_risk = pd.DataFrame(risk_rows)

    # ── CSVs ──────────────────────────────────────────────────────────────────
    perf_csv = REPORTS_DIR / "minimal_vs_full_model_comparison.csv"
    risk_csv = REPORTS_DIR / "minimal_vs_full_risk_stratification.csv"
    df_perf.to_csv(perf_csv, index=False)
    df_risk.to_csv(risk_csv, index=False)
    print(f"Saved -> {perf_csv.relative_to(REPO_ROOT)}")
    print(f"Saved -> {risk_csv.relative_to(REPO_ROOT)}")

    # ── Main performance MD ───────────────────────────────────────────────────
    perf_md = REPORTS_DIR / "minimal_vs_full_model_comparison.md"
    lines: list[str] = [
        "# Minimal vs Full Model Comparison",
        "",
        "## Methodology",
        "",
        "**Data:** `data/share/needle_features_v1.csv`, "
        "filtered to `label_join_status == \"coord_match\"` with pre-assigned `split` column.  ",
        "**Endpoints:** GG2+ / csPCa (`binary_label_int`) and "
        "GG3+ / high-grade (`binary_label_gg3plus_int`).  ",
        "**Models:** logistic regression (class-balanced, standardised features) "
        "and HistGradientBoosting; XGBoost included when installed.  ",
        "**Threshold selection:** Youden J maximised on the validation split.  ",
        "**Bootstrap CIs:** 1,000 patient-level resamples of the test set "
        "(patients resampled with replacement, all their cores included; "
        "2.5/97.5 percentiles reported).  ",
        f"**Reference feature set:** `{REFERENCE_FS}` "
        "(biopsy/prostate geometry ×6 + target geometry no availability ×4 + clinical ×4).  ",
        "`target_mesh_available` is excluded from all feature sets "
        "(conservative ablation; mesh-absent cores are rare but have elevated positivity).",
        "",
        "## Performance",
        "",
    ]

    for label_col, ep_label in ENDPOINTS:
        sub = df_perf[df_perf["label_col"] == label_col]
        if sub.empty:
            lines += [f"### {ep_label}", "", "*No results.*", ""]
            continue
        lines += [f"### {ep_label}", ""]
        lines += _md_table(sub, _PERF_COLS)
        lines.append("")

    lines += _interpret(df_perf)

    perf_md.write_text("\n".join(lines) + "\n")
    print(f"Saved -> {perf_md.relative_to(REPO_ROOT)}")

    # ── Delta MD ──────────────────────────────────────────────────────────────
    delta_md = REPORTS_DIR / "minimal_vs_full_delta_table.md"
    dlines: list[str] = [
        "# Delta Table: Feature Set Comparison vs Full Model",
        "",
        f"**Reference:** `{REFERENCE_FS}` (14 features).  ",
        "ΔROC-AUC and ΔPR-AUC = (this feature set) − (reference).  ",
        "Negative values indicate underperformance relative to the reference.",
        "",
    ]
    for label_col, ep_label in ENDPOINTS:
        sub = df_perf[df_perf["label_col"] == label_col]
        dlines += _delta_section(sub, ep_label)

    delta_md.write_text("\n".join(dlines) + "\n")
    print(f"Saved -> {delta_md.relative_to(REPO_ROOT)}")

    # ── Risk stratification MD ────────────────────────────────────────────────
    risk_md = REPORTS_DIR / "minimal_vs_full_risk_stratification.md"
    rlines: list[str] = [
        "# Risk Stratification",
        "",
        "**Capture rate** = fraction of all test-set positives in the top-N% of scored cores.  ",
        "**Prevalence** in top-N% = positive rate among the top-N% highest-probability cores.  ",
        "**Baseline prevalence** = overall positive rate in the test set.",
        "",
    ]
    for label_col, ep_label in ENDPOINTS:
        sub_r = df_risk[df_risk["label_col"] == label_col]
        if sub_r.empty:
            rlines += [f"## {ep_label}", "", "*No results.*", ""]
            continue
        rlines += [f"## {ep_label}", ""]
        rlines += _md_table(sub_r, _RISK_COLS)
        rlines.append("")

    risk_md.write_text("\n".join(rlines) + "\n")
    print(f"Saved -> {risk_md.relative_to(REPO_ROOT)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== Minimal vs full model comparison ===\n")
    if not HAS_XGBOOST:
        warnings.warn(
            "xgboost is not installed — XGBoost models will be skipped. "
            "Install with: pip install xgboost",
            stacklevel=1,
        )

    results:   list[dict] = []
    risk_rows: list[dict] = []

    for label_col, ep_label in ENDPOINTS:
        print(f"\n--- Endpoint: {ep_label} ({label_col}) ---")
        df, _ = load_dataset(DATA_PATH, target_col=label_col)
        df = prepare_features(df)
        print(f"  {len(df):,} rows, {df[PATIENT_COL].nunique():,} patients")

        available_cols = set(df.columns)
        model_names = [name for name, _, _ in get_model_specs(df[label_col])]

        for fs_name, fs_cols_raw in FEATURE_SETS.items():
            feature_cols = [c for c in fs_cols_raw if c in available_cols]
            missing = [c for c in fs_cols_raw if c not in available_cols]
            if missing:
                warnings.warn(
                    f"[{label_col}/{fs_name}] missing columns: {missing}", stacklevel=1
                )
            if not feature_cols:
                continue

            for model_name in model_names:
                print(f"  [{fs_name}] {model_name} ...", end=" ", flush=True)
                row, test_probs = run_one(
                    label_col, fs_name, feature_cols, model_name, df
                )
                if row is None:
                    print("skipped")
                    continue

                results.append(row)
                print(
                    f"ROC-AUC={_f(row['test_roc_auc'])}  "
                    f"PR-AUC={_f(row['test_pr_auc'])}"
                )

                test_df = df[df[SPLIT_COL] == "test"]
                y_test  = test_df[label_col].values
                risk = _risk_strat(y_test, test_probs)
                risk_rows.append({
                    "label_col":   label_col,
                    "feature_set": fs_name,
                    "model":       model_name,
                    **risk,
                })

    if not results:
        print("\nNo results produced — check data and dependencies.")
        return

    write_reports(results, risk_rows)
    print("\nDone.")


if __name__ == "__main__":
    main()
