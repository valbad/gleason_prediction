"""
Direct patient-level models for prostate cancer prediction.

Goal
----
Build true patient-level models (one row per patient, features aggregated from
core-level measurements) and compare them against the previous naive core-score
aggregation approach from analyze_patient_level_performance.py.

Input
-----
data/share/needle_features_v1.csv

Filtering
---------
  label_join_status == "coord_match"
  split in {"train", "val", "test"}
  non-null endpoint label (per endpoint)
  target_mesh_available excluded from all feature sets
  split uniqueness per patient is asserted

Patient label
-------------
Positive if any core is positive for the endpoint.

Outputs
-------
reports/patient_level_direct_model_comparison.csv
reports/patient_level_direct_model_comparison.md

Usage
-----
    python src/analyze_patient_level_direct_models.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from run_shareable_tabular_experiments import (
    CLINICAL_FEATURES,
    HAS_XGBOOST,
    PATIENT_COL,
    RANDOM_STATE,
    SPLIT_COL,
    build_model,
    compute_metrics,
    get_model_specs,
    load_dataset,
    prepare_features,
    select_threshold_youden,
)


# ── Paths & constants ─────────────────────────────────────────────────────────

REPO_ROOT   = Path(__file__).resolve().parent.parent
DATA_PATH   = REPO_ROOT / "data" / "share" / "needle_features_v1.csv"
REPORTS_DIR = REPO_ROOT / "reports"

N_BOOT = 1000

ENDPOINTS: list[tuple[str, str]] = [
    ("binary_label_int",         "GG2+ / csPCa"),
    ("binary_label_gg3plus_int", "GG3+ / high-grade"),
]

# Known results: naive core-score aggregation (analyze_patient_level_performance.py)
NAIVE_REFERENCE: dict[str, dict] = {
    "binary_label_int": {
        "label": "XGBoost mean prob",
        "roc_auc": 0.617,
        "pr_auc": 0.695,
    },
    "binary_label_gg3plus_int": {
        "label": "Logistic Regression mean prob",
        "roc_auc": 0.682,
        "pr_auc": 0.512,
    },
}

# Best core-level results (all_geometry_no_availability_plus_clinical, test set)
_CORE_LEVEL_REF: dict[str, dict] = {
    "binary_label_int":         {"roc_auc": 0.758, "pr_auc": 0.327},
    "binary_label_gg3plus_int": {"roc_auc": 0.829, "pr_auc": 0.265},
}


# ── Patient-level feature definitions ─────────────────────────────────────────

_PAT_CLINICAL: list[str] = CLINICAL_FEATURES  # 4 features

_PAT_TARGET_GEO_AGG: list[str] = [
    "min_distance_to_target_surface_mm",
    "mean_distance_to_target_surface_mm",
    "median_distance_to_target_surface_mm",
    "p10_distance_to_target_surface_mm",
    "p25_distance_to_target_surface_mm",
    "min_distance_to_target_centroid_mm",
    "mean_distance_to_target_centroid_mm",
    "median_distance_to_target_centroid_mm",
    "max_fraction_inside_target",
    "mean_fraction_inside_target",
    "fraction_cores_intersecting_target",
    "n_cores_intersecting_target",
]

_PAT_BIOPSY_GEO_AGG: list[str] = [
    "mean_core_length_mm",
    "median_core_length_mm",
    "mean_distance_midpoint_to_prostate_surface_mm",
    "median_distance_midpoint_to_prostate_surface_mm",
    "mean_fraction_centerline_inside_prostate",
    "fraction_midpoints_inside_prostate",
]

PAT_FEATURE_SETS: dict[str, list[str]] = {
    "clinical_only":                            _PAT_CLINICAL,
    "target_geometry_aggregates_only":          _PAT_TARGET_GEO_AGG,
    "target_geometry_aggregates_plus_clinical": _PAT_TARGET_GEO_AGG + _PAT_CLINICAL,
    "all_geometry_aggregates_plus_clinical":    _PAT_TARGET_GEO_AGG + _PAT_BIOPSY_GEO_AGG + _PAT_CLINICAL,
}


# ── Data loading & patient table ──────────────────────────────────────────────

def assert_split_unique(df: pd.DataFrame) -> None:
    """Each patient must appear in exactly one split."""
    multi = df.groupby(PATIENT_COL)[SPLIT_COL].nunique()
    bad = multi[multi > 1]
    if len(bad):
        raise AssertionError(
            f"{len(bad)} patients appear in more than one split: {bad.index.tolist()}"
        )


def build_patient_table(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """
    Aggregate one-row-per-core data to one-row-per-patient.

    Returns a DataFrame indexed by PATIENT_COL.  Columns: split,
    patient_label, n_cores, all patient-level aggregate features.
    """
    g = df.groupby(PATIENT_COL)
    parts: dict[str, pd.Series] = {}

    parts["split"]         = g[SPLIT_COL].first()
    parts["patient_label"] = g[label_col].max().astype(int)
    parts["n_cores"]       = g.size()

    # Clinical: first non-null per patient (pandas groupby.first() skips NaN)
    for col in _PAT_CLINICAL:
        if col in df.columns:
            parts[col] = g[col].first()

    # Target geometry aggregates
    _surf = "distance_midpoint_to_target_surface_mm"
    if _surf in df.columns:
        s = g[_surf]
        parts["min_distance_to_target_surface_mm"]    = s.min()
        parts["mean_distance_to_target_surface_mm"]   = s.mean()
        parts["median_distance_to_target_surface_mm"] = s.median()
        parts["p10_distance_to_target_surface_mm"]    = s.quantile(0.10)
        parts["p25_distance_to_target_surface_mm"]    = s.quantile(0.25)

    _cent = "distance_midpoint_to_target_centroid_mm"
    if _cent in df.columns:
        s = g[_cent]
        parts["min_distance_to_target_centroid_mm"]    = s.min()
        parts["mean_distance_to_target_centroid_mm"]   = s.mean()
        parts["median_distance_to_target_centroid_mm"] = s.median()

    _frac_tgt = "approximate_fraction_of_centerline_inside_target"
    if _frac_tgt in df.columns:
        s = g[_frac_tgt]
        parts["max_fraction_inside_target"]  = s.max()
        parts["mean_fraction_inside_target"] = s.mean()

    # trajectory_intersects_target has been converted to 0/1 float by prepare_features
    _traj = "trajectory_intersects_target"
    if _traj in df.columns:
        s = g[_traj]
        parts["fraction_cores_intersecting_target"] = s.mean()
        parts["n_cores_intersecting_target"]        = s.sum()

    # Biopsy/prostate geometry aggregates
    if "core_length_mm" in df.columns:
        s = g["core_length_mm"]
        parts["mean_core_length_mm"]   = s.mean()
        parts["median_core_length_mm"] = s.median()

    _pro_surf = "distance_midpoint_to_prostate_surface_mm"
    if _pro_surf in df.columns:
        s = g[_pro_surf]
        parts["mean_distance_midpoint_to_prostate_surface_mm"]   = s.mean()
        parts["median_distance_midpoint_to_prostate_surface_mm"] = s.median()

    _frac_pro = "approximate_fraction_of_centerline_inside_prostate"
    if _frac_pro in df.columns:
        parts["mean_fraction_centerline_inside_prostate"] = g[_frac_pro].mean()

    # midpoint_inside_prostate has been converted to 0/1 float by prepare_features
    if "midpoint_inside_prostate" in df.columns:
        parts["fraction_midpoints_inside_prostate"] = g["midpoint_inside_prostate"].mean()

    return pd.DataFrame(parts)


# ── Bootstrap CI ──────────────────────────────────────────────────────────────

def bootstrap_ci(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_boot: int = N_BOOT,
    random_state: int = RANDOM_STATE,
) -> dict:
    """
    Percentile bootstrap CIs for ROC-AUC and PR-AUC, resampling patients.

    Each row in the test set is one patient, so row-resampling = patient-resampling.
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    roc_samp: list[float] = []
    pr_samp:  list[float] = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        roc_samp.append(float(roc_auc_score(y_b, probs[idx])))
        pr_samp.append(float(average_precision_score(y_b, probs[idx])))

    ci: dict = {}
    for key, samp in [("roc_auc", roc_samp), ("pr_auc", pr_samp)]:
        if samp:
            ci[f"{key}_ci_low"]  = float(np.percentile(samp, 2.5))
            ci[f"{key}_ci_high"] = float(np.percentile(samp, 97.5))
        else:
            ci[f"{key}_ci_low"]  = np.nan
            ci[f"{key}_ci_high"] = np.nan
    return ci


# ── Experiment runner ─────────────────────────────────────────────────────────

def run_one(
    label_col: str,
    fs_name: str,
    feature_cols: list[str],
    model_name: str,
    pat_train: pd.DataFrame,
    pat_val: pd.DataFrame,
    pat_test: pd.DataFrame,
) -> dict | None:
    """
    Fit on train patients, select threshold on val (Youden J), evaluate on test.

    pat_* DataFrames are indexed by PATIENT_COL and must contain a
    'patient_label' column plus all feature columns.
    """
    usable  = [c for c in feature_cols if pat_train[c].notna().any()]
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
        return None

    X_train, y_train = pat_train[usable], pat_train["patient_label"]
    X_val,   y_val   = pat_val[usable],   pat_val["patient_label"]
    X_test,  y_test  = pat_test[usable],  pat_test["patient_label"]

    pipeline = build_model(model_name, usable, y_train)
    if pipeline is None:
        return None

    pipeline.fit(X_train, y_train)

    val_probs  = pipeline.predict_proba(X_val)[:, 1]
    test_probs = pipeline.predict_proba(X_test)[:, 1]

    threshold    = select_threshold_youden(y_val.values, val_probs)
    test_metrics = compute_metrics(y_test.values, test_probs, threshold)

    test_prevalence = float(y_test.mean())
    pr_auc_val = test_metrics.get("pr_auc", np.nan)
    pr_lift = (
        pr_auc_val / test_prevalence
        if test_prevalence > 0 and not np.isnan(pr_auc_val)
        else np.nan
    )

    ci = bootstrap_ci(test_probs, y_test.values)

    row: dict = {
        "label_col":          label_col,
        "feature_set":        fs_name,
        "model":              model_name,
        "n_features":         len(usable),
        "n_train_patients":   len(pat_train),
        "n_val_patients":     len(pat_val),
        "n_test_patients":    len(pat_test),
        "test_prevalence":    test_prevalence,
        "pr_lift":            pr_lift,
        "threshold":          threshold,
    }
    row.update({f"test_{k}": v for k, v in test_metrics.items()})
    row.update(ci)
    return row


# ── Formatting ────────────────────────────────────────────────────────────────

def _f(v, n: int = 3) -> str:
    if isinstance(v, (float, np.floating)) and np.isnan(v):
        return "n/a"
    return f"{v:.{n}f}"


def _md_table(rows: list[dict], cols: list[str]) -> list[str]:
    header = "| " + " | ".join(cols) + " |"
    sep    = "|" + "---|" * len(cols)
    lines  = [header, sep]
    for r in rows:
        cells = [
            _f(v) if isinstance(v, (float, np.floating)) else str(v)
            for v in (r.get(c, np.nan) for c in cols)
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


# ── Interpretation ────────────────────────────────────────────────────────────

def _interpret(df: pd.DataFrame) -> list[str]:
    """
    Four-point automatic interpretation (without top-level section header).
    Caller is responsible for prepending the ## header.
    """
    lines: list[str] = []

    def _best_row(sub: pd.DataFrame) -> pd.Series | None:
        if sub.empty:
            return None
        return sub.loc[sub["test_roc_auc"].idxmax()]

    def _mean_roc(sub: pd.DataFrame, fs: str) -> float:
        v = sub.loc[sub["feature_set"] == fs, "test_roc_auc"]
        return float(v.mean()) if not v.empty else np.nan

    for label_col, ep_label in ENDPOINTS:
        sub   = df[df["label_col"] == label_col]
        naive = NAIVE_REFERENCE.get(label_col, {})
        core  = _CORE_LEVEL_REF.get(label_col, {})
        best  = _best_row(sub)

        lines += [f"### {ep_label}", ""]
        if best is None:
            lines += ["*No results available.*", ""]
            continue

        # Point 1: direct patient-level vs naive aggregation
        if naive:
            d_roc = float(best["test_roc_auc"]) - naive["roc_auc"]
            d_pr  = float(best["test_pr_auc"])  - naive["pr_auc"]
            s_roc = "+" if d_roc >= 0 else ""
            s_pr  = "+" if d_pr  >= 0 else ""
            if d_roc > 0.010:
                v1 = (
                    f"Direct patient-level modelling (`{best['feature_set']}` / "
                    f"`{best['model']}`, ROC-AUC {_f(best['test_roc_auc'])}) "
                    f"improves over naive core-score aggregation "
                    f"({naive['label']}, ROC-AUC {_f(naive['roc_auc'])}; "
                    f"ΔROC = {s_roc}{_f(d_roc)}, ΔPR = {s_pr}{_f(d_pr)})."
                )
            elif d_roc < -0.010:
                v1 = (
                    f"Naive core-score aggregation "
                    f"({naive['label']}, ROC-AUC {_f(naive['roc_auc'])}) "
                    f"outperforms the best direct patient-level model "
                    f"(`{best['feature_set']}` / `{best['model']}`, "
                    f"ROC-AUC {_f(best['test_roc_auc'])}; "
                    f"ΔROC = {s_roc}{_f(d_roc)}, ΔPR = {s_pr}{_f(d_pr)})."
                )
            else:
                v1 = (
                    f"Direct patient-level modelling and naive aggregation perform "
                    f"comparably (best direct: `{best['feature_set']}` / "
                    f"`{best['model']}`, ROC-AUC {_f(best['test_roc_auc'])}; "
                    f"naive: {naive['label']}, ROC-AUC {_f(naive['roc_auc'])}; "
                    f"ΔROC = {s_roc}{_f(d_roc)}, ΔPR = {s_pr}{_f(d_pr)})."
                )
            lines += [f"**Direct vs naive aggregation:** {v1}", ""]

        # Point 2: target geometry aggregates vs clinical only
        tgc_roc  = _mean_roc(sub, "target_geometry_aggregates_plus_clinical")
        clin_roc = _mean_roc(sub, "clinical_only")
        if not (np.isnan(tgc_roc) or np.isnan(clin_roc)):
            d = tgc_roc - clin_roc
            if d > 0.010:
                v2 = (
                    f"Target geometry aggregates add meaningful value over clinical "
                    f"features alone (mean ROC-AUC {_f(tgc_roc)} vs {_f(clin_roc)}; "
                    f"ΔROC = +{_f(d)})."
                )
            else:
                v2 = (
                    f"Target geometry aggregates add limited incremental value over "
                    f"clinical features alone at the patient level "
                    f"(mean ROC-AUC {_f(tgc_roc)} vs {_f(clin_roc)}; ΔROC = {_f(d)})."
                )
            lines += [f"**Target geometry aggregates vs clinical only:** {v2}", ""]

        # Point 3: patient-level vs core-level
        if core:
            d_cl = float(best["test_roc_auc"]) - core["roc_auc"]
            s_cl = "+" if d_cl >= 0 else ""
            if d_cl < -0.010:
                v3 = (
                    f"Patient-level prediction remains weaker than the best core-level model "
                    f"(direct patient-level ROC-AUC {_f(best['test_roc_auc'])} vs "
                    f"core-level {_f(core['roc_auc'])}; ΔROC = {s_cl}{_f(d_cl)}). "
                    f"Aggregating to the patient level dilutes the core-level geometric signal."
                )
            else:
                v3 = (
                    f"Patient-level models are competitive with core-level models "
                    f"(direct patient-level ROC-AUC {_f(best['test_roc_auc'])} vs "
                    f"core-level {_f(core['roc_auc'])}; ΔROC = {s_cl}{_f(d_cl)})."
                )
            lines += [f"**Patient-level vs core-level performance:** {v3}", ""]

    # Point 4: implication for paper framing
    lines += ["### Implication for paper framing", ""]

    # Check whether any direct model substantially improves over naive aggregation
    improvements = []
    for label_col, ep_label in ENDPOINTS:
        sub   = df[df["label_col"] == label_col]
        naive = NAIVE_REFERENCE.get(label_col, {})
        best  = _best_row(sub)
        if best is not None and naive:
            improvements.append(float(best["test_roc_auc"]) - naive["roc_auc"])

    if improvements and max(improvements) > 0.010:
        framing = (
            "Direct patient-level modelling shows some improvement over naive aggregation "
            "for at least one endpoint. However, patient prevalence is substantially higher "
            "than core prevalence, many patients have one positive core among many negatives, "
            "and the core-level geometry signal is diluted when summarised to a single "
            "patient-level score. "
            "The paper should report direct patient-level models as a secondary or exploratory "
            "analysis, while keeping core-level risk stratification as the primary result "
            "where the geometric signal is strongest."
        )
    else:
        framing = (
            "Direct patient-level modelling does not substantially improve over naive "
            "core-score aggregation. Patient prevalence is substantially higher than core "
            "prevalence, many patients have one positive core among many negatives, and "
            "aggregating geometric features to the patient level dilutes the core-level "
            "signal. "
            "These results support keeping the paper centred on core-level risk "
            "stratification, with patient-level analyses reported as supplementary."
        )
    lines += [framing, ""]

    return lines


# ── Report writing ────────────────────────────────────────────────────────────

_PERF_COLS = [
    "feature_set", "model", "n_features",
    "test_roc_auc", "roc_auc_ci_low", "roc_auc_ci_high",
    "test_pr_auc",  "pr_auc_ci_low",  "pr_auc_ci_high",
    "pr_lift", "threshold",
    "test_sensitivity", "test_specificity",
    "test_precision",   "test_f1",
    "n_train_patients", "n_val_patients", "n_test_patients",
    "test_prevalence",
]


def write_reports(results: list[dict], pat_summary: list[dict]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)

    csv_path = REPORTS_DIR / "patient_level_direct_model_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved -> {csv_path.relative_to(REPO_ROOT)}")

    lines: list[str] = [
        "# Direct Patient-Level Model Comparison",
        "",
        "## 1. Methodology",
        "",
        "Core-level features are aggregated to one row per patient before model fitting. "
        "The patient label is positive if any core is positive for the endpoint. "
        "Models are trained on train-split patients, threshold is selected on val-split "
        "patients (Youden J), and test-split patients are evaluated once.  ",
        "**Bootstrap CIs:** 1,000 patient resamples of the test set with replacement "
        "(2.5/97.5 percentiles).  ",
        "`target_mesh_available` is excluded from all feature sets.  ",
        "`n_cores` is computed for auditing but excluded from the main predictive "
        "feature sets, because biopsy count may encode sampling/procedural intensity "
        "rather than anatomy or biology.",
        "",
    ]

    # ── Section 2: patient-level dataset summary ──────────────────────────────
    lines += ["## 2. Patient-Level Dataset Summary", ""]
    for row in pat_summary:
        lines += [
            f"### {row['ep_label']}",
            "",
            "| Split | Patients | Positive | Prevalence |",
            "|---|---|---|---|",
        ]
        for split in ("train", "val", "test"):
            lines.append(
                f"| {split} "
                f"| {row[f'{split}_n']} "
                f"| {row[f'{split}_pos']} "
                f"| {_f(row[f'{split}_prev'])} |"
            )
        lines.append("")

    # ── Section 3: main performance table ─────────────────────────────────────
    lines += ["## 3. Main Performance Table", ""]
    for label_col, ep_label in ENDPOINTS:
        sub = df[df["label_col"] == label_col]
        if sub.empty:
            lines += [f"### {ep_label}", "", "*No results.*", ""]
            continue
        lines += [f"### {ep_label}", ""]
        lines += _md_table(sub.to_dict("records"), _PERF_COLS)
        lines.append("")

    # ── Section 4: best model per endpoint ────────────────────────────────────
    lines += ["## 4. Best Direct Patient-Level Model per Endpoint", ""]
    for label_col, ep_label in ENDPOINTS:
        sub = df[df["label_col"] == label_col]
        if sub.empty:
            lines += [f"*{ep_label}: no results.*", ""]
            continue
        best = sub.loc[sub["test_roc_auc"].idxmax()]
        lines += [
            f"**{ep_label}:** `{best['feature_set']}` / `{best['model']}` — "
            f"ROC-AUC {_f(best['test_roc_auc'])} "
            f"[{_f(best['roc_auc_ci_low'])}–{_f(best['roc_auc_ci_high'])}], "
            f"PR-AUC {_f(best['test_pr_auc'])} "
            f"[{_f(best['pr_auc_ci_low'])}–{_f(best['pr_auc_ci_high'])}]",
            "",
        ]

    # ── Section 5: comparison against naive aggregation ───────────────────────
    lines += [
        "## 5. Comparison Against Previous Naive Core-Score Aggregation",
        "",
        "| Endpoint | Approach | Feature set / model | ROC-AUC | PR-AUC | ΔROC-AUC | ΔPR-AUC |",
        "|---|---|---|---|---|---|---|",
    ]
    for label_col, ep_label in ENDPOINTS:
        sub   = df[df["label_col"] == label_col]
        naive = NAIVE_REFERENCE.get(label_col, {})
        if not naive:
            continue
        lines.append(
            f"| {ep_label} | Naive aggregation | {naive['label']} "
            f"| {_f(naive['roc_auc'])} | {_f(naive['pr_auc'])} | — | — |"
        )
        if sub.empty:
            continue
        best  = sub.loc[sub["test_roc_auc"].idxmax()]
        d_roc = float(best["test_roc_auc"]) - naive["roc_auc"]
        d_pr  = float(best["test_pr_auc"])  - naive["pr_auc"]
        s_roc = "+" if d_roc >= 0 else ""
        s_pr  = "+" if d_pr  >= 0 else ""
        lines.append(
            f"| {ep_label} | Direct patient-level (best) "
            f"| `{best['feature_set']}` / `{best['model']}` "
            f"| {_f(best['test_roc_auc'])} | {_f(best['test_pr_auc'])} "
            f"| {s_roc}{_f(d_roc)} | {s_pr}{_f(d_pr)} |"
        )
    lines.append("")

    # ── Section 6: interpretation ─────────────────────────────────────────────
    lines += ["## 6. Interpretation", ""]
    lines += _interpret(df)

    md_path = REPORTS_DIR / "patient_level_direct_model_comparison.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(f"Saved -> {md_path.relative_to(REPO_ROOT)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== Direct patient-level model comparison ===\n")
    if not HAS_XGBOOST:
        warnings.warn(
            "xgboost is not installed — XGBoost models will be skipped. "
            "Install with: pip install xgboost",
            stacklevel=1,
        )

    results:     list[dict] = []
    pat_summary: list[dict] = []

    for label_col, ep_label in ENDPOINTS:
        print(f"\n--- Endpoint: {ep_label} ({label_col}) ---")

        df_cores, _ = load_dataset(DATA_PATH, target_col=label_col)
        df_cores = prepare_features(df_cores)

        assert_split_unique(df_cores)

        # Build patient-level table (indexed by PATIENT_COL)
        pat = build_patient_table(df_cores, label_col)

        # Split into train / val / test patient sets
        pat_train = pat[pat["split"] == "train"].drop(columns=["split", "n_cores"])
        pat_val   = pat[pat["split"] == "val"].drop(columns=["split", "n_cores"])
        pat_test  = pat[pat["split"] == "test"].drop(columns=["split", "n_cores"])

        n_tr, n_va, n_te = len(pat_train), len(pat_val), len(pat_test)
        pos_tr = int(pat_train["patient_label"].sum())
        pos_va = int(pat_val["patient_label"].sum())
        pos_te = int(pat_test["patient_label"].sum())

        print(f"  train : {n_tr} patients, {pos_tr} positive ({pos_tr/n_tr*100:.1f}%)")
        print(f"  val   : {n_va} patients, {pos_va} positive ({pos_va/n_va*100:.1f}%)")
        print(f"  test  : {n_te} patients, {pos_te} positive ({pos_te/n_te*100:.1f}%)")

        pat_summary.append({
            "ep_label":  ep_label,
            "label_col": label_col,
            "train_n":   n_tr,  "train_pos": pos_tr, "train_prev": pos_tr / n_tr,
            "val_n":     n_va,  "val_pos":   pos_va, "val_prev":   pos_va / n_va,
            "test_n":    n_te,  "test_pos":  pos_te, "test_prev":  pos_te / n_te,
        })

        model_names = [name for name, _, _ in get_model_specs(pat_train["patient_label"])]
        available   = set(pat_train.columns)

        for fs_name, fs_cols_raw in PAT_FEATURE_SETS.items():
            feature_cols = [c for c in fs_cols_raw if c in available]
            missing      = [c for c in fs_cols_raw if c not in available]
            if missing:
                warnings.warn(
                    f"[{label_col}/{fs_name}] missing columns: {missing}", stacklevel=1
                )
            if not feature_cols:
                continue

            for model_name in model_names:
                print(f"  [{fs_name}] {model_name} ...", end=" ", flush=True)
                row = run_one(
                    label_col, fs_name, feature_cols, model_name,
                    pat_train, pat_val, pat_test,
                )
                if row is None:
                    print("skipped")
                    continue
                results.append(row)
                print(
                    f"ROC-AUC={_f(row['test_roc_auc'])}  "
                    f"PR-AUC={_f(row['test_pr_auc'])}"
                )

    if not results:
        print("\nNo results produced — check data and dependencies.")
        return

    write_reports(results, pat_summary)
    print("\nDone.")


if __name__ == "__main__":
    main()
