import warnings
import json
import gzip
import joblib
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import concurrent.futures
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

import data_processing as dp
import cptac_data_processing as cdp

warnings.filterwarnings("ignore")


def compute_time_aucs(y_train_rec, y_val_df, y_pred, time_points=[12.0, 36.0, 60.0]):
    """
    Computes time-dependent cumulative dynamic AUCs for 1-Yr (12m), 3-Yr (36m), 5-Yr (60m).
    Uses y_train_rec (CPTAC training records) as baseline reference.
    """
    y_val_rec = np.empty(len(y_val_df), dtype=[("event", bool), ("duration", float)])
    y_val_rec["event"] = y_val_df["event"].astype(bool).values
    y_val_rec["duration"] = y_val_df["duration"].astype(float).values

    min_time = y_val_rec["duration"].min()
    max_time = y_val_rec["duration"].max()

    valid_times = []
    labels = []
    for t in time_points:
        if min_time < t < max_time:
            n_events_before = (y_val_rec["event"] & (y_val_rec["duration"] <= t)).sum()
            n_follow_after = (y_val_rec["duration"] > t).sum()
            if n_events_before > 0 and n_follow_after > 0:
                valid_times.append(t)
                labels.append(f"{int(t/12)}Yr")

    if len(valid_times) == 0:
        return {}

    try:
        va_auc, _ = cumulative_dynamic_auc(y_train_rec, y_val_rec, y_pred, np.array(valid_times))
        return {label: f"{auc_val:.3f}" for label, auc_val in zip(labels, va_auc)}
    except Exception:
        return {}


def compute_km_logrank_hr(y_val_df, y_pred):
    """
    Computes median risk-score stratified Kaplan-Meier log-rank p-value and Hazard Ratio (HR with 95% CI).
    """
    df = y_val_df.copy()
    df["risk_score"] = y_pred
    median_risk = df["risk_score"].median()
    df["group"] = (df["risk_score"] > median_risk).astype(int)

    high = df[df["group"] == 1]
    low = df[df["group"] == 0]

    if len(high) == 0 or len(low) == 0:
        return {"logrank_p": "N/A", "HR": "N/A", "HR_CI": "N/A"}

    try:
        lr_res = logrank_test(high["duration"], low["duration"], event_observed_A=high["event"], event_observed_B=low["event"])
        p_val = lr_res.p_value

        cph = CoxPHFitter(penalizer=0.01)
        cph.fit(df[["duration", "event", "group"]], duration_col="duration", event_col="event")
        summary = cph.summary.loc["group"]
        hr = summary["exp(coef)"]
        ci_lower = summary["exp(coef) lower 95%"]
        ci_upper = summary["exp(coef) upper 95%"]

        return {
            "logrank_p": f"{p_val:.4e}" if p_val < 0.001 else f"{p_val:.4f}",
            "HR": f"{hr:.2f}",
            "HR_CI": f"[{ci_lower:.2f}-{ci_upper:.2f}]"
        }
    except Exception:
        return {"logrank_p": "N/A", "HR": "N/A", "HR_CI": "N/A"}


def load_tcga_external(genes):
    print("Loading external TCGA-KIRC validation set...")
    clinical = dp.load_clean_kirc_clinical()
    tcga_expr = dp.get_tcga_expr("gene_name", "tpm")

    omic_w_clinical = clinical[["event", "duration", "Gender"]].join(tcga_expr, how="inner").groupby(level=0).first()

    avail_genes = [g for g in genes if g in omic_w_clinical.columns]
    missing_genes = [g for g in genes if g not in avail_genes]
    if missing_genes:
        print(f"TCGA missing {len(missing_genes)} features. Filling with 0.0.")

    X = omic_w_clinical.reindex(columns=genes, fill_value=0.0)
    y = omic_w_clinical[["event", "duration"]].copy()
    y["duration"] = y["duration"] / 30.437  # Convert days to months
    y["event"] = y["event"].astype(bool)

    return X, y, omic_w_clinical["Gender"]


def load_icgc_external(genes):
    print("Loading external ICGC RECA-EU validation set...")
    from eval_sex_agnostic import load_icgc_dataset
    df_icgc = load_icgc_dataset(genes)
    if df_icgc.empty:
        return None, None, None

    avail_genes = [g for g in genes if g in df_icgc.columns]
    X = df_icgc.reindex(columns=genes, fill_value=0.0)
    y = df_icgc[["event", "duration"]].copy()
    y["event"] = y["event"].astype(bool)

    return X, y, df_icgc["Gender"]


def load_gse22541_external(genes):
    print("Loading external GSE22541 validation set...")
    from eval_sex_agnostic import load_gse22541_dataset
    df_gse = load_gse22541_dataset(genes)
    if df_gse.empty:
        return None, None, None

    X = df_gse.reindex(columns=genes, fill_value=0.0)
    y = df_gse[["event", "duration"]].copy()
    y["event"] = y["event"].astype(bool)

    return X, y, df_gse["Gender"]


def evaluate_cohort(cohort_name, X_val, y_val, scaler, models, y_train_rec):
    print(f"\n=======================================================")
    print(f" EVALUATING CPTAC-TRAINED MODEL ON {cohort_name.upper()} ({len(X_val)} patients, {y_val['event'].sum()} deaths)")
    print(f"=======================================================")

    # Apply FROZEN CPTAC scaler (NO refitting on validation data!)
    X_val_scaled = scaler.transform(X_val)
    X_val_scaled_df = pd.DataFrame(X_val_scaled, columns=X_val.columns)

    results = []

    for m_label, model in models.items():
        if model is None:
            continue
        try:
            # Predict using FROZEN CPTAC model (NO refitting on validation data!)
            y_pred = model.predict(X_val_scaled_df)

            c_index = concordance_index_censored(y_val["event"], y_val["duration"], y_pred)[0]
            auc_dict = compute_time_aucs(y_train_rec, y_val, y_pred)
            km_stats = compute_km_logrank_hr(y_val, y_pred)

            results.append({
                "Validation Cohort": cohort_name,
                "Model": m_label,
                "C-Index": f"{c_index:.4f}",
                "1-Yr AUC": auc_dict.get("1Yr", "N/A"),
                "3-Yr AUC": auc_dict.get("3Yr", "N/A"),
                "5-Yr AUC": auc_dict.get("5Yr", "N/A"),
                "Log-Rank p": km_stats["logrank_p"],
                "HR (95% CI)": f"{km_stats['HR']} {km_stats['HR_CI']}"
            })
        except Exception as e:
            print(f"Error evaluating {m_label} on {cohort_name}: {e}")

    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))
    return df_res


def main():
    bundle_path = Path("./aggregated_results/cptac_frozen_models.joblib")
    if not bundle_path.exists():
        print(f"Error: Frozen CPTAC model bundle not found at {bundle_path}.")
        print("Please run cptac_model_freezer.py first.")
        return

    bundle = joblib.load(bundle_path)
    genes = bundle["genes"]
    scaler = bundle["scaler"]
    models = bundle["models"]

    # Reconstruct CPTAC baseline training records for time AUC reference
    clinical = cdp.get_cptac_survival_data()
    y_cptac = clinical[["event", "duration"]].copy()
    y_cptac["duration"] = y_cptac["duration"] / 30.437
    y_cptac["event"] = y_cptac["event"].astype(bool)
    y_train_rec = y_cptac.to_records(index=False)

    print(f"Loaded frozen CPTAC bundle: {len(genes)} signature genes, {len(models)} trained models.")

    all_eval_rows = []

    # 1. External TCGA Evaluation
    X_tcga, y_tcga, _ = load_tcga_external(genes)
    if X_tcga is not None and not X_tcga.empty:
        df_tcga_res = evaluate_cohort("TCGA-KIRC", X_tcga, y_tcga, scaler, models, y_train_rec)
        all_eval_rows.append(df_tcga_res)

    # 2. External GSE22541 Evaluation
    X_gse, y_gse, _ = load_gse22541_external(genes)
    if X_gse is not None and not X_gse.empty:
        df_gse_res = evaluate_cohort("GSE22541", X_gse, y_gse, scaler, models, y_train_rec)
        all_eval_rows.append(df_gse_res)

    # 3. External ICGC RECA-EU Evaluation
    X_icgc, y_icgc, _ = load_icgc_external(genes)
    if X_icgc is not None and not X_icgc.empty:
        df_icgc_res = evaluate_cohort("ICGC RECA-EU", X_icgc, y_icgc, scaler, models, y_train_rec)
        all_eval_rows.append(df_icgc_res)

    if all_eval_rows:
        final_df = pd.concat(all_eval_rows, ignore_index=True)
        out_csv = Path("./aggregated_results/cptac_external_validation_results.csv")
        final_df.to_csv(out_csv, index=False)
        print(f"\nSaved all CPTAC-trained external validation metrics to {out_csv}")


if __name__ == "__main__":
    main()
