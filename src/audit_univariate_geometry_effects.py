"""
Univariate interpretability audit for target geometry and clinical features.

For each (label, feature) pair this script:
  1. Computes Spearman correlation with the label on all filtered rows.
  2. Fits a univariate logistic regression on the train split.
  3. Reports the standardised coefficient (numeric features are z-scored;
     boolean features are left as 0/1 and their coefficient is unscaled).
  4. Evaluates ROC-AUC and PR-AUC on the test split.

This is an interpretability audit to check marginal feature directions,
NOT a predictive model. Multicollinearity and confounding are ignored.

Dataset filter (full_geometry_dataset, same as MODE A in the main experiments):
  label_join_status == "coord_match"
  chosen label is not null
  split in {"train", "val", "test"}

Usage
-----
    python src/audit_univariate_geometry_effects.py

Outputs
-------
    reports/univariate_geometry_effects.csv
    reports/univariate_geometry_effects.md
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

REPO_ROOT   = Path(__file__).resolve().parent.parent
DATA_PATH   = REPO_ROOT / "data" / "share" / "needle_features_v1.csv"
REPORTS_DIR = REPO_ROOT / "reports"

PATIENT_COL  = "patient_number"
SPLIT_COL    = "split"
VALID_SPLITS = ("train", "val", "test")

LABEL_COLS = ["binary_label_int", "binary_label_gg3plus_int"]
LABEL_DESC = {
    "binary_label_int":         "GG2+ / csPCa (Gleason ≥ 3+4=7)",
    "binary_label_gg3plus_int": "GG3+ / high-grade (Gleason ≥ 4+3=7)",
}

# Boolean features: imputed by most_frequent, never standard-scaled.
# Coefficient interpretation: log-odds change for a 0→1 transition.
BOOLEAN_FEATURES = {"trajectory_intersects_target", "midpoint_inside_prostate"}

FEATURES = [
    "log_psa_ng_ml",
    "psa_density",
    "prostate_volume_cc",
    "distance_midpoint_to_target_centroid_mm",
    "distance_midpoint_to_target_surface_mm",
    "trajectory_intersects_target",
    "approximate_fraction_of_centerline_inside_target",
    "core_length_mm",
    "midpoint_inside_prostate",
]


# ── Data loading & feature derivation ────────────────────────────────────────

def _to_binary(series: pd.Series) -> pd.Series:
    """Map True/False-like values to float 0/1, NaN otherwise."""
    truthy = {True, "True", "TRUE", "true", 1, "1", 1.0}
    falsy  = {False, "False", "FALSE", "false", 0, "0", 0.0}
    out = pd.Series(np.nan, index=series.index, dtype=float)
    out[series.isin(truthy)] = 1.0
    out[series.isin(falsy)]  = 0.0
    return out


def load_and_prepare(label_col: str) -> pd.DataFrame:
    raw = pd.read_csv(DATA_PATH)

    mask = (
        (raw["label_join_status"] == "coord_match")
        & raw[label_col].notna()
        & raw[SPLIT_COL].isin(VALID_SPLITS)
    )
    df = raw[mask].copy()
    df[label_col] = df[label_col].astype(int)

    # Derived clinical features
    df["log_psa_ng_ml"] = np.log1p(df["psa_ng_ml"])
    valid_vol = df["prostate_volume_cc"].notna() & (df["prostate_volume_cc"] > 0)
    df["psa_density"] = np.where(
        valid_vol & df["psa_ng_ml"].notna(),
        df["psa_ng_ml"] / df["prostate_volume_cc"],
        np.nan,
    )

    # Normalise boolean columns to 0/1 float
    for col in BOOLEAN_FEATURES:
        if col in df.columns:
            df[col] = _to_binary(df[col])

    return df.reset_index(drop=True)


# ── Univariate analysis ───────────────────────────────────────────────────────

def spearman(series: pd.Series, label: pd.Series) -> tuple[float, float]:
    """Spearman ρ and p-value between one feature and the label (pairwise complete)."""
    valid = series.notna() & label.notna()
    if valid.sum() < 5:
        return np.nan, np.nan
    r, p = stats.spearmanr(series[valid], label[valid])
    return float(r), float(p)


def univariate_lr(
    feature: str,
    df: pd.DataFrame,
    label_col: str,
) -> dict:
    """
    Fit a univariate logistic regression on the train split, evaluate on test.

    Returns a dict of metrics, or a dict of NaNs if the feature is not usable.
    """
    nan_row = dict(
        n_train=np.nan, n_train_pos=np.nan,
        n_test=np.nan,  n_test_pos=np.nan,
        test_prevalence=np.nan,
        lr_coef=np.nan, lr_intercept=np.nan,
        test_roc_auc=np.nan, test_pr_auc=np.nan,
    )

    train = df[df[SPLIT_COL] == "train"]
    test  = df[df[SPLIT_COL] == "test"]

    # Need at least one non-NaN value in train to fit
    if train[feature].isna().all():
        warnings.warn(f"[{label_col}/{feature}] all-NaN in train — skipping LR.")
        return nan_row

    # Need both classes in train
    if len(train[label_col].unique()) < 2:
        warnings.warn(f"[{label_col}/{feature}] only one class in train — skipping LR.")
        return nan_row

    is_bool = feature in BOOLEAN_FEATURES

    if is_bool:
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=2000)),
        ])
    else:
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("clf",     LogisticRegression(class_weight="balanced", max_iter=2000)),
        ])

    X_train = train[[feature]]
    y_train = train[label_col]
    X_test  = test[[feature]]
    y_test  = test[label_col]

    pipe.fit(X_train, y_train)

    coef      = float(pipe.named_steps["clf"].coef_[0, 0])
    intercept = float(pipe.named_steps["clf"].intercept_[0])

    n_test     = len(y_test)
    n_test_pos = int(y_test.sum())

    if len(y_test.unique()) < 2 or n_test_pos == 0:
        roc_auc = np.nan
        pr_auc  = np.nan
    else:
        probs   = pipe.predict_proba(X_test)[:, 1]
        roc_auc = float(roc_auc_score(y_test, probs))
        pr_auc  = float(average_precision_score(y_test, probs))

    return dict(
        n_train=len(y_train),
        n_train_pos=int(y_train.sum()),
        n_test=n_test,
        n_test_pos=n_test_pos,
        test_prevalence=n_test_pos / n_test if n_test > 0 else np.nan,
        lr_coef=coef,
        lr_intercept=intercept,
        test_roc_auc=roc_auc,
        test_pr_auc=pr_auc,
    )


# ── Reporting ─────────────────────────────────────────────────────────────────

def _fmt(v, dec: int = 3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{dec}f}"
    return str(v)


def _sign_arrow(v: float) -> str:
    if np.isnan(v):
        return "—"
    return "▲" if v >= 0 else "▼"


def write_csv(rows: list[dict], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Saved -> {path.relative_to(REPO_ROOT)}")


def write_md(rows: list[dict], path: Path) -> None:
    df_all = pd.DataFrame(rows)

    lines = [
        "# Univariate Geometry & Clinical Feature Effects",
        "",
        "Marginal associations between individual features and each binary label.",
        "**Not** a multivariate model: interactions and confounding are not captured.",
        "",
        "**Coefficient** = standardised log-odds per 1 σ (numeric features, z-scored  ",
        "before LR) or per 0→1 transition (boolean features, unscaled).  ",
        "**ROC-AUC / PR-AUC** evaluated on the test split (threshold-free).  ",
        "**Spearman ρ** computed on all filtered rows (train + val + test).",
        "",
    ]

    for label_col in LABEL_COLS:
        desc  = LABEL_DESC[label_col]
        sub   = df_all[df_all["label_col"] == label_col].copy()
        if sub.empty:
            continue

        # Sort by absolute coefficient descending (NaN last)
        sub["_abs_coef"] = sub["lr_coef"].abs()
        sub = sub.sort_values("_abs_coef", ascending=False).drop(columns="_abs_coef")

        # Quick summary: test prevalence
        prev_example = sub["test_prevalence"].dropna()
        prev_str = _fmt(prev_example.iloc[0]) if not prev_example.empty else "n/a"

        lines += [
            f"## {label_col} — {desc}",
            "",
            f"Test-set prevalence: **{prev_str}**  ",
            f"(train n = {sub['n_train'].dropna().astype(int).iloc[0]:,}  "
            f"· test n = {sub['n_test'].dropna().astype(int).iloc[0]:,})",
            "",
            "| Feature | Scaled? | Spearman ρ | Spearman p | "
            "Coef (std LR) | Dir | ROC-AUC | PR-AUC |",
            "|---|---|---|---|---|---|---|---|",
        ]

        for _, r in sub.iterrows():
            scaled = "std-scaled" if not r["is_boolean"] else "0/1 flag"
            lines.append(
                f"| `{r['feature']}` | {scaled} "
                f"| {_fmt(r['spearman_r'])} | {_fmt(r['spearman_p'])} "
                f"| {_fmt(r['lr_coef'])} | {_sign_arrow(r['lr_coef'])} "
                f"| {_fmt(r['test_roc_auc'])} | {_fmt(r['test_pr_auc'])} |"
            )

        lines.append("")

        # Highlight strongest associations
        pos = sub[sub["lr_coef"] > 0].head(3)
        neg = sub[sub["lr_coef"] < 0].head(3)

        if not pos.empty:
            lines += [
                "**Strongest positive associations** (▲ → higher probability of label = 1):",
                "",
            ]
            for _, r in pos.iterrows():
                lines.append(
                    f"- `{r['feature']}`: coef = {_fmt(r['lr_coef'])}, "
                    f"ROC-AUC = {_fmt(r['test_roc_auc'])}, "
                    f"Spearman ρ = {_fmt(r['spearman_r'])}"
                )
            lines.append("")

        if not neg.empty:
            lines += [
                "**Strongest negative associations** (▼ → lower probability of label = 1):",
                "",
            ]
            for _, r in neg.iterrows():
                lines.append(
                    f"- `{r['feature']}`: coef = {_fmt(r['lr_coef'])}, "
                    f"ROC-AUC = {_fmt(r['test_roc_auc'])}, "
                    f"Spearman ρ = {_fmt(r['spearman_r'])}"
                )
            lines.append("")

    path.write_text("\n".join(lines) + "\n")
    print(f"Saved -> {path.relative_to(REPO_ROOT)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== Univariate geometry & clinical effects audit ===\n")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    for label_col in LABEL_COLS:
        print(f"Label: {label_col}")
        df = load_and_prepare(label_col)
        n_pos = int((df[label_col] == 1).sum())
        print(f"  {len(df):,} rows, {n_pos:,} positives ({n_pos/len(df)*100:.1f}%)")

        for feat in FEATURES:
            if feat not in df.columns:
                print(f"  [skip] '{feat}' not in dataset")
                continue

            rho, pval = spearman(df[feat], df[label_col])
            lr_metrics = univariate_lr(feat, df, label_col)

            row = dict(
                label_col=label_col,
                feature=feat,
                is_boolean=(feat in BOOLEAN_FEATURES),
                n_total_filtered=len(df),
                n_feature_nonmissing=int(df[feat].notna().sum()),
                spearman_r=rho,
                spearman_p=pval,
                **lr_metrics,
            )
            all_rows.append(row)

            coef_str = _fmt(lr_metrics["lr_coef"])
            roc_str  = _fmt(lr_metrics["test_roc_auc"])
            print(f"  {feat:<55s}  coef={coef_str:>8}  ROC-AUC={roc_str}  ρ={_fmt(rho)}")

        print()

    write_csv(all_rows, REPORTS_DIR / "univariate_geometry_effects.csv")
    write_md(all_rows,  REPORTS_DIR / "univariate_geometry_effects.md")


if __name__ == "__main__":
    main()
