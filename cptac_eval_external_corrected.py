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
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, ComponentwiseGradientBoostingSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

import data_processing as dp
import cptac_data_processing as cdp

warnings.filterwarnings("ignore")

# Full GENCODE / Ensembl ID lookup for all sex-agnostic consensus and frozen bundle genes
GENE_TO_ENSG_AGNOSTIC = {
    "LTF": "ENSG00000012223",
    "FER1L4": "ENSG00000088340",
    "KIF14": "ENSG00000118193",
    "SEC14L4": "ENSG00000133488",
    "KIAA0319": "ENSG00000137261",
    "MYOF": "ENSG00000138119",
    "TMEM25": "ENSG00000149582",
    "PPDPFL": "ENSG00000168333",
    "ASPHD1": "ENSG00000174939",
    "CCNYL2": "ENSG00000182632",
    "RNU6-891P": "ENSG00000207007",
    "AL355379.1": "ENSG00000217786",
    "RNU4-78P": "ENSG00000222872",
    "AC078883.2": "ENSG00000226963",
    "AP000844.2": "ENSG00000238117",
    "RNU6-237P": "ENSG00000251794",
    "AP000757.2": "ENSG00000254844",
    "TAS2R38": "ENSG00000257138",
    "TMEM179": "ENSG00000258986",
    "MIR3617": "ENSG00000266071",
    "AC115485.1": "ENSG00000287740",
}


def cindex_scorer(estimator, X, y):
    n_events = np.sum(y[estimator._event_field] if hasattr(estimator, '_event_field') else [ev for ev, _ in y])
    if n_events == 0:
        return 0.5
    try:
        preds = estimator.predict(X)
        if np.isnan(preds).any() or np.isneginf(preds).any() or np.isposinf(preds).any():
            return 0.5
        c_idx = concordance_index_censored(
            y[estimator._event_field] if hasattr(estimator, '_event_field') else [ev for ev, _ in y],
            y[estimator._duration_field] if hasattr(estimator, '_duration_field') else [dur for _, dur in y],
            preds
        )[0]
        return c_idx if not np.isnan(c_idx) else 0.5
    except Exception:
        return 0.5


def train_coxnet(X, y):
    best_score = -np.inf
    best_result = None
    for l1_ratio in [0.1, 0.3, 0.5, 1.0]:
        try:
            base_pipe = make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alpha_min_ratio=0.01, max_iter=300000, tol=1e-6))
            base_pipe.fit(X, y)
            alphas = base_pipe.named_steps["coxnetsurvivalanalysis"].alphas_
            alphas = alphas[alphas > 0.001]
            if len(alphas) == 0:
                continue
            folds = KFold(n_splits=5, shuffle=True, random_state=3)
            gcv = GridSearchCV(
                make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, max_iter=300000)),
                param_grid={"coxnetsurvivalanalysis__alphas": [[a] for a in alphas]},
                cv=folds, n_jobs=-1, error_score=np.nan, scoring=cindex_scorer
            )
            gcv.fit(X, y)
            if not np.isnan(gcv.best_score_) and gcv.best_score_ > best_score:
                best_score = gcv.best_score_
                best_result = gcv.best_estimator_
        except Exception:
            continue
    return best_result


def train_rf(X, y):
    folds = KFold(n_splits=5, shuffle=True, random_state=3)
    gcv = GridSearchCV(
        RandomSurvivalForest(random_state=3),
        param_grid={"n_estimators": [50, 100], "max_depth": [3, 4, 5], "min_samples_split": [5, 10]},
        cv=folds, n_jobs=-1, error_score=np.nan, scoring=cindex_scorer
    )
    gcv.fit(X, y)
    return gcv.best_estimator_


def train_svm(X, y):
    folds = KFold(n_splits=5, shuffle=True, random_state=3)
    gcv = GridSearchCV(
        make_pipeline(StandardScaler(), FastSurvivalSVM(max_iter=1000, random_state=3)),
        param_grid={"fastsurvivalsvm__alpha": [2**i for i in range(-5, 5)]},
        cv=folds, n_jobs=-1, error_score=np.nan, scoring=cindex_scorer
    )
    gcv.fit(X, y)
    return gcv.best_estimator_


def train_boost(X, y):
    folds = KFold(n_splits=5, shuffle=True, random_state=3)
    gcv = GridSearchCV(
        ComponentwiseGradientBoostingSurvivalAnalysis(random_state=3),
        param_grid={"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]},
        cv=folds, n_jobs=-1, error_score=np.nan, scoring=cindex_scorer
    )
    gcv.fit(X, y)
    return gcv.best_estimator_


def compute_time_aucs(y_train_rec, y_val_df, y_pred, time_points=[12.0, 36.0, 60.0]):
    """
    Computes cumulative dynamic AUCs for 1-Yr, 3-Yr, 5-Yr horizons.
    Includes fallback to evaluation cohort's own Kaplan-Meier censoring distribution
    when validation follow-up exceeds training reference horizon.
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
        return {}, "N/A"

    try:
        # Primary: try IPCW using training cohort reference
        va_auc, _ = cumulative_dynamic_auc(y_train_rec, y_val_rec, y_pred, np.array(valid_times))
        return {label: f"{auc_val:.3f}" for label, auc_val in zip(labels, va_auc)}, "CPTAC-Train (Primary)"
    except Exception:
        try:
            # Secondary fallback: evaluation cohort's own KM censoring distribution
            # (required when external cohort follow-up exceeds CPTAC's ~65.9m observation horizon)
            va_auc, _ = cumulative_dynamic_auc(y_val_rec, y_val_rec, y_pred, np.array(valid_times))
            return {label: f"{auc_val:.3f}" for label, auc_val in zip(labels, va_auc)}, "Validation-KM (Uno et al.)"
        except Exception:
            return {}, "Failed"


def compute_km_logrank_hr(y_val_df, y_pred):
    """
    Computes median risk-score stratified log-rank p-value and Cox PH Hazard Ratio with 95% CI.
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
    """
    Queries exact sex-agnostic consensus genes directly from TCGA STAR-counts TSVs.
    """
    print("Loading external TCGA-KIRC validation set...")
    clinical = dp.load_clean_kirc_clinical()

    rna_dir = Path("./TCGA_KIRC_RNA_seq")
    tcga_files = list(rna_dir.glob("**/*.rna_seq.augmented_star_gene_counts.tsv"))
    metadata_path = Path("./Metadata/tcga_kirc_rna_metadata.json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    meta_rows = []
    for item in metadata:
        ent = item["associated_entities"][0]
        meta_rows.append({
            "file_name": item["file_name"],
            "case_id": ent["case_id"],
            "aliquot_id": ent["entity_submitter_id"]
        })
    meta_df = pd.DataFrame(meta_rows)
    meta_df["rank"] = meta_df["aliquot_id"].apply(lambda a: 0 if "-01A-" in a else (1 if "-01B-" in a else 2))
    meta_df = meta_df.sort_values(["case_id", "rank", "aliquot_id"]).drop_duplicates(subset="case_id", keep="first")
    meta_df = meta_df[meta_df["case_id"].isin(clinical.index)]
    metadata_dict = dict(zip(meta_df["file_name"], meta_df["case_id"]))

    expr_dict = {}
    for file in tcga_files:
        filename = file.name
        if filename not in metadata_dict:
            continue
        caseid = metadata_dict[filename]
        patient = pd.read_csv(file, sep="\t", skiprows=[0, 2, 3, 4, 5])
        sub = patient[patient["gene_name"].isin(genes)].set_index("gene_name")["tpm_unstranded"]
        expr_dict[caseid] = np.log2(sub + 1)

    all_expr = pd.DataFrame(expr_dict).T

    omic_w_clinical = clinical[["event", "duration", "Gender"]].join(all_expr, how="inner")
    avail_genes = [g for g in genes if g in omic_w_clinical.columns]

    X = omic_w_clinical[avail_genes]
    y = omic_w_clinical[["event", "duration"]].copy()
    y["duration"] = y["duration"] / 30.437  # Convert days to months
    y["event"] = y["event"].astype(bool)

    return X, y, omic_w_clinical["Gender"]


def load_gse22541_external(genes):
    """
    Loads GSE22541 microarray dataset, applies log2 intensity normalization,
    and returns measured consensus genes.
    """
    print("Loading external GSE22541 validation set...")
    from eval_sex_agnostic import load_gse22541_dataset
    df_gse = load_gse22541_dataset(genes)
    if df_gse.empty:
        return None, None, None

    avail_genes = [g for g in genes if g in df_gse.columns]
    # Apply log2 transform to bring raw Affymetrix microarray intensity onto log2 scale matching CPTAC
    X = np.log2(np.maximum(df_gse[avail_genes], 1.0))
    y = df_gse[["event", "duration"]].copy()
    y["event"] = y["event"].astype(bool)

    return X, y, df_gse["Gender"]


def load_icgc_external(genes):
    """
    Downloads primary solid tumor expression and clinical outcomes from ICGC RECA-EU
    using corrected Ensembl gene IDs.
    """
    print("Loading external ICGC RECA-EU validation set...")

    def get_all_keys(prefix):
        keys, marker = [], ""
        ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        base_url = "https://object.genomeinformatics.org/icgc25k-open"
        while True:
            params = {"prefix": prefix}
            if marker:
                params["marker"] = marker
            url = f"{base_url}?{urllib.parse.urlencode(params)}"
            with urllib.request.urlopen(url) as resp:
                root = ET.fromstring(resp.read())
                contents = root.findall("s3:Contents", ns)
                if not contents:
                    break
                for content in contents:
                    keys.append(content.find("s3:Key", ns).text)
                is_trunc = root.find("s3:IsTruncated", ns)
                if is_trunc is not None and is_trunc.text == "true":
                    marker = keys[-1]
                else:
                    break
        return keys

    try:
        all_keys = get_all_keys("release_28/data/RECA-EU/")
    except Exception as e:
        print(f"Error connecting to ICGC S3 endpoint: {e}")
        return None, None, None

    donor_files = {}
    for k in all_keys:
        parts = k.split("/")
        if len(parts) > 3:
            donor = parts[3]
            donor_files.setdefault(donor, []).append(k)

    valid_donors = [donor for donor, files in donor_files.items() if any("exp_seq" in f for f in files)]

    def download_url(url):
        with urllib.request.urlopen(url) as resp:
            return resp.read()

    base_url = "https://object.genomeinformatics.org/icgc25k-open"

    def process_donor(donor_id):
        files = donor_files[donor_id]
        specimen_key = next((f for f in files if "specimen" in f), None)
        if not specimen_key:
            return None
        try:
            specimen_str = gzip.decompress(download_url(f"{base_url}/{specimen_key}")).decode("utf-8")
            primary_spec_ids = [
                line.split("\t")[0]
                for line in specimen_str.strip().split("\n")
                if len(line.split("\t")) > 6 and "Primary tumour" in line.split("\t")[6]
            ]
            if not primary_spec_ids:
                return None

            donor_key = next((f for f in files if "donor" in f), None)
            if not donor_key:
                return None
            donor_str = gzip.decompress(download_url(f"{base_url}/{donor_key}")).decode("utf-8")
            c_parts = donor_str.strip().split("\t")
            if len(c_parts) < 18:
                return None
            sex, vital = c_parts[4].lower(), c_parts[5].lower()
            t_str, f_str = c_parts[16], c_parts[17]
            if not t_str or t_str in ["Unnamed: 16", "nan"]:
                if not f_str or f_str in ["Unnamed: 17", "nan"]:
                    return None
                time_days = float(f_str)
            else:
                time_days = float(t_str)
            event = 1 if "deceased" in vital else 0
            duration_months = time_days / 30.437
            encoded_gender = 0 if "female" in sex else 1

            exp_keys = [f for f in files if "exp_seq" in f]
            exp_data = {}
            for ek in exp_keys:
                exp_str = gzip.decompress(download_url(f"{base_url}/{ek}")).decode("utf-8")
                lines = exp_str.strip().split("\n")
                if len(lines[0].split("\t")) < 3 or lines[0].split("\t")[2] not in primary_spec_ids:
                    continue
                for line in lines:
                    parts = line.split("\t")
                    if len(parts) > 8:
                        exp_data[parts[7].split(".")[0]] = float(parts[8])
                break
            if not exp_data:
                return None
            return {
                "donor_id": donor_id,
                "Gender": encoded_gender,
                "event": event,
                "duration": duration_months,
                "expression": exp_data
            }
        except Exception:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        results = executor.map(process_donor, valid_donors)
        parsed_donors = [r for r in results if r]

    records = []
    for pd_item in parsed_donors:
        rec = {
            "case_id": f"ICGC_{pd_item['donor_id']}",
            "Gender": pd_item["Gender"],
            "event": pd_item["event"],
            "duration": pd_item["duration"]
        }
        ensg_keys = [k for k in pd_item["expression"].keys() if k.startswith("ENSG")]
        tot_rpkm = sum(pd_item["expression"][k] for k in ensg_keys)
        if tot_rpkm == 0:
            tot_rpkm = 1.0

        for sym in genes:
            ensg = GENE_TO_ENSG_AGNOSTIC.get(sym)
            if ensg and ensg in pd_item["expression"]:
                rpkm = pd_item["expression"][ensg]
                tpm = (rpkm / tot_rpkm) * 1e6
                rec[sym] = np.log2(tpm + 1)
            else:
                rec[sym] = 0.0
        records.append(rec)

    df_icgc = pd.DataFrame(records).fillna(0.0)
    if df_icgc.empty:
        return None, None, None

    avail_genes = [g for g in genes if g in df_icgc.columns and not (df_icgc[g] == 0.0).all()]
    if not avail_genes:
        return None, None, None

    X = df_icgc[avail_genes]
    y = df_icgc[["event", "duration"]].copy()
    y["event"] = y["event"].astype(bool)

    return X, y, df_icgc["Gender"]


def evaluate_cohort(cohort_name, X_val, y_val, scaler, models, y_train_rec, full_gene_set, cptac_X_train, cptac_y_rec):
    n_pts = len(X_val)
    n_deaths = int(y_val["event"].sum())
    val_cols = list(X_val.columns)
    n_mapped = len([g for g in full_gene_set if g in val_cols and not (X_val[g] == 0.0).all()])

    print(f"\n=======================================================")
    print(f" EVALUATING CPTAC-TRAINED MODEL ON {cohort_name.upper()} ({n_pts} patients, {n_deaths} deaths, {n_mapped}/{len(full_gene_set)} genes mapped)")
    print(f"=======================================================")

    is_subset = len(val_cols) < len(full_gene_set)
    if is_subset:
        print(f"Cohort uses overlapping feature subset: {val_cols} ({len(val_cols)}/{len(full_gene_set)} genes)")
        scaler_sub = StandardScaler()
        X_tr_sub_scaled = scaler_sub.fit_transform(cptac_X_train[val_cols])
        X_tr_sub_df = pd.DataFrame(X_tr_sub_scaled, columns=val_cols)
        X_val_scaled = scaler_sub.transform(X_val)
        X_val_scaled_df = pd.DataFrame(X_val_scaled, columns=val_cols)

        active_models = {
            "Coxnet": train_coxnet(X_tr_sub_df, cptac_y_rec),
            "Random Forest": train_rf(X_tr_sub_df, cptac_y_rec),
            "Survival SVM": train_svm(X_tr_sub_df, cptac_y_rec),
            "Gradient Boosting": train_boost(X_tr_sub_df, cptac_y_rec)
        }
    else:
        X_val_scaled = scaler.transform(X_val[full_gene_set])
        X_val_scaled_df = pd.DataFrame(X_val_scaled, columns=full_gene_set)
        active_models = models

    results = []
    for m_label, model in active_models.items():
        if model is None:
            continue
        try:
            y_pred = model.predict(X_val_scaled_df)

            c_index = concordance_index_censored(y_val["event"], y_val["duration"], y_pred)[0]
            auc_dict, ipcw_ref = compute_time_aucs(y_train_rec, y_val, y_pred)
            km_stats = compute_km_logrank_hr(y_val, y_pred)

            results.append({
                "Validation Cohort": cohort_name,
                "Model": m_label,
                "N Patients": n_pts,
                "N Deaths": n_deaths,
                "Mapped Genes": f"{n_mapped}/{len(full_gene_set)}",
                "C-Index": f"{c_index:.4f}",
                "1-Yr AUC": auc_dict.get("1Yr", "N/A"),
                "3-Yr AUC": auc_dict.get("3Yr", "N/A"),
                "5-Yr AUC": auc_dict.get("5Yr", "N/A"),
                "AUC IPCW Reference": ipcw_ref,
                "Log-Rank p": km_stats["logrank_p"],
                "HR (95% CI)": f"{km_stats['HR']} {km_stats['HR_CI']}"
            })
        except Exception as e:
            print(f"Error evaluating {m_label} on {cohort_name}: {e}")

    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))
    return df_res


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Corrected CPTAC Sex-Agnostic External Evaluation")
    parser.add_argument(
        "--use_legacy_15gene_bundle",
        action="store_true",
        help="Evaluate the preliminary 15-gene bundle from Oct 5 instead of the finalized 9-gene RRA consensus signature"
    )
    args = parser.parse_args()

    rra_path = Path("./aggregated_results/cptac_sex_agnostic_expr_aggregated_ranking.csv")
    bundle_path = Path("./aggregated_results/cptac_frozen_models.joblib")

    # Reconstruct CPTAC baseline training records and features
    clinical = cdp.get_cptac_survival_data()
    tpm_data = cdp.get_cptac_expr("gene_name", "tpm")
    omic_w_clinical = clinical.join(tpm_data, how="inner").groupby(level=0).first()

    y_cptac = omic_w_clinical[["event", "duration"]].copy()
    y_cptac["duration"] = y_cptac["duration"] / 30.437  # Convert days to months
    y_cptac["event"] = y_cptac["event"].astype(bool)
    cptac_y_rec = y_cptac.to_records(index=False)

    if args.use_legacy_15gene_bundle:
        if not bundle_path.exists():
            print(f"Error: Frozen CPTAC legacy bundle not found at {bundle_path}.")
            return
        bundle = joblib.load(bundle_path)
        genes = bundle["genes"]
        scaler = bundle["scaler"]
        models = bundle["models"]
        print(f"\n[MODE: LEGACY 15-GENE PRELIMINARY BUNDLE]")
        print(f"Evaluating preliminary 15 genes from {bundle_path}: {genes}")
        out_csv = Path("./aggregated_results/cptac_external_validation_results_legacy15_corrected.csv")
    else:
        if not rra_path.exists():
            print(f"Error: Finalized CPTAC 9-gene signature not found at {rra_path}.")
            return
        sig_df = pd.read_csv(rra_path)
        genes = sig_df["Name"].tolist()
        print(f"\n[MODE: OFFICIAL 9-GENE RRA CONSENSUS SIGNATURE (p < 0.05)]")
        print(f"Evaluating {len(genes)} consensus genes from {rra_path}: {genes}")

        # Train models on CPTAC cohort for these exact 9 consensus genes
        cptac_X_train_9 = omic_w_clinical[genes].copy()
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(cptac_X_train_9), columns=genes)
        models = {
            "Coxnet": train_coxnet(X_train_scaled, cptac_y_rec),
            "Random Forest": train_rf(X_train_scaled, cptac_y_rec),
            "Survival SVM": train_svm(X_train_scaled, cptac_y_rec),
            "Gradient Boosting": train_boost(X_train_scaled, cptac_y_rec)
        }
        # Save separate clean 9-gene bundle preserving the legacy bundle intact
        new_bundle_path = Path("./aggregated_results/cptac_frozen_models_9gene.joblib")
        joblib.dump({"genes": genes, "scaler": scaler, "models": models}, new_bundle_path)
        print(f"Saved frozen 9-gene model bundle to {new_bundle_path}")
        out_csv = Path("./aggregated_results/cptac_external_validation_results_corrected.csv")

    cptac_X_train = omic_w_clinical[genes].copy()
    all_eval_rows = []

    # 1. External TCGA Evaluation (Full 479 patients)
    X_tcga, y_tcga, gender_tcga = load_tcga_external(genes)
    if X_tcga is not None and not X_tcga.empty:
        df_tcga_res = evaluate_cohort(
            "TCGA-KIRC (Overall)", X_tcga, y_tcga, scaler, models,
            cptac_y_rec, genes, cptac_X_train, cptac_y_rec
        )
        all_eval_rows.append(df_tcga_res)

        # Subgroup evaluations on TCGA (Male vs Female) for sex-agnostic comparison
        m_mask = gender_tcga == 1
        if m_mask.sum() > 0:
            df_tcga_m = evaluate_cohort(
                "TCGA-KIRC (Male Subgroup)", X_tcga[m_mask], y_tcga[m_mask], scaler, models,
                cptac_y_rec, genes, cptac_X_train, cptac_y_rec
            )
            all_eval_rows.append(df_tcga_m)

        f_mask = gender_tcga == 0
        if f_mask.sum() > 0:
            df_tcga_f = evaluate_cohort(
                "TCGA-KIRC (Female Subgroup)", X_tcga[f_mask], y_tcga[f_mask], scaler, models,
                cptac_y_rec, genes, cptac_X_train, cptac_y_rec
            )
            all_eval_rows.append(df_tcga_f)

    # 2. External GSE22541 Evaluation
    X_gse, y_gse, _ = load_gse22541_external(genes)
    if X_gse is not None and not X_gse.empty:
        df_gse_res = evaluate_cohort(
            "GSE22541", X_gse, y_gse, scaler, models,
            cptac_y_rec, genes, cptac_X_train, cptac_y_rec
        )
        all_eval_rows.append(df_gse_res)

    # 3. External ICGC RECA-EU Evaluation
    X_icgc, y_icgc, _ = load_icgc_external(genes)
    if X_icgc is not None and not X_icgc.empty:
        df_icgc_res = evaluate_cohort(
            "ICGC RECA-EU", X_icgc, y_icgc, scaler, models,
            cptac_y_rec, genes, cptac_X_train, cptac_y_rec
        )
        all_eval_rows.append(df_icgc_res)

    if all_eval_rows:
        final_df = pd.concat(all_eval_rows, ignore_index=True)
        final_df.to_csv(out_csv, index=False)
        print(f"\n=======================================================")
        print(f" Saved all corrected external validation metrics to {out_csv}")
        print(f"=======================================================")


if __name__ == "__main__":
    main()

