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
from cptac_model_freezer_sex_specific import train_coxnet, train_rf, train_svm, train_boost

warnings.filterwarnings("ignore")

# Full GENCODE v36 Ensembl ID lookup table for all 13 consensus genes
GENE_TO_ENSG = {
    "PAEP": "ENSG00000122133",
    "H19": "ENSG00000130600",
    "FGG": "ENSG00000171557",
    "FGA": "ENSG00000171560",
    "FGB": "ENSG00000171564",
    "SNORD31B": "ENSG00000201847",
    "RNU6-171P": "ENSG00000207082",
    "IGHD5-5": "ENSG00000211928",
    "AC005162.1": "ENSG00000233517",
    "RNU6-237P": "ENSG00000251794",
    "TMEM179": "ENSG00000258986",
    "BNIP3P15": "ENSG00000270716",
    "AC107385.2": "ENSG00000287770"
}


def compute_time_aucs(y_train_rec, y_val_df, y_pred, time_points=[12.0, 36.0, 60.0]):
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
        # Primary: try IPCW using training cohort reference
        va_auc, _ = cumulative_dynamic_auc(y_train_rec, y_val_rec, y_pred, np.array(valid_times))
        return {label: f"{auc_val:.3f}" for label, auc_val in zip(labels, va_auc)}
    except Exception:
        try:
            # Secondary: fallback to evaluation cohort's own censoring distribution
            # (required when external validation cohort follow-up exceeds training cohort support)
            va_auc, _ = cumulative_dynamic_auc(y_val_rec, y_val_rec, y_pred, np.array(valid_times))
            return {label: f"{auc_val:.3f}" for label, auc_val in zip(labels, va_auc)}
        except Exception:
            return {}


def compute_km_logrank_hr(y_val_df, y_pred):
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


def load_tcga_sex_cohort(genes, sex_code):
    """
    Queries exact signature genes directly from raw TCGA TSV files for specified sex cohort.
    sex_code: 1 = Male, 0 = Female
    """
    clinical = dp.load_clean_kirc_clinical()
    sex_clinical = clinical[clinical["Gender"] == sex_code].copy()

    rna_dir = Path("./TCGA_KIRC_RNA_seq")
    tcga_files = list(rna_dir.glob("**/*.rna_seq.augmented_star_gene_counts.tsv"))
    with open(Path("./Metadata/tcga_kirc_rna_metadata.json"), "r") as f:
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
    meta_df = meta_df[meta_df["case_id"].isin(sex_clinical.index)]
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

    omic_w_clinical = sex_clinical[["event", "duration"]].join(all_expr, how="inner")
    X = omic_w_clinical.reindex(columns=genes, fill_value=0.0)
    y = omic_w_clinical[["event", "duration"]].copy()
    y["duration"] = y["duration"] / 30.437
    y["event"] = y["event"].astype(bool)

    return X, y


def load_gse22541_sex_cohort(genes, sex_code):
    from eval_sex_agnostic import load_gse22541_dataset
    df_gse = load_gse22541_dataset(genes)
    if df_gse.empty:
        return None, None

    sex_df = df_gse[df_gse["Gender"] == sex_code].copy()
    if sex_df.empty:
        return None, None

    # Preserve only columns present in GSE22541 (overlapping features strategy matching old pipeline)
    avail_genes = [g for g in genes if g in sex_df.columns]
    # Apply log2 transform to bring raw Affymetrix microarray intensity onto log2 scale matching CPTAC
    X = np.log2(np.maximum(sex_df[avail_genes], 1.0))
    y = sex_df[["event", "duration"]].copy()
    y["event"] = y["event"].astype(bool)

    return X, y


def load_icgc_sex_cohort(genes, sex_code):
    def get_all_keys(prefix):
        keys, marker = [], ""
        ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        base_url = "https://object.genomeinformatics.org/icgc25k-open"
        while True:
            params = {"prefix": prefix}
            if marker: params["marker"] = marker
            url = f"{base_url}?{urllib.parse.urlencode(params)}"
            with urllib.request.urlopen(url) as resp:
                root = ET.fromstring(resp.read())
                contents = root.findall("s3:Contents", ns)
                if not contents: break
                for content in contents: keys.append(content.find("s3:Key", ns).text)
                is_trunc = root.find("s3:IsTruncated", ns)
                if is_trunc is not None and is_trunc.text == "true": marker = keys[-1]
                else: break
        return keys

    try:
        all_keys = get_all_keys("release_28/data/RECA-EU/")
    except Exception:
        return None, None

    donor_files = {}
    for k in all_keys:
        parts = k.split("/")
        if len(parts) > 3:
            donor = parts[3]
            donor_files.setdefault(donor, []).append(k)

    valid_donors = [donor for donor, files in donor_files.items() if any("exp_seq" in f for f in files)]

    def download_url(url):
        with urllib.request.urlopen(url) as resp: return resp.read()

    def process_donor(donor_id):
        files = donor_files[donor_id]
        specimen_key = next((f for f in files if "specimen" in f), None)
        if not specimen_key: return None
        base_url = "https://object.genomeinformatics.org/icgc25k-open"
        try:
            specimen_str = gzip.decompress(download_url(f"{base_url}/{specimen_key}")).decode("utf-8")
            primary_spec_ids = [line.split("\t")[0] for line in specimen_str.strip().split("\n") if len(line.split("\t")) > 6 and "Primary tumour" in line.split("\t")[6]]
            if not primary_spec_ids: return None

            donor_key = next((f for f in files if "donor" in f), None)
            if not donor_key: return None
            donor_str = gzip.decompress(download_url(f"{base_url}/{donor_key}")).decode("utf-8")
            c_parts = donor_str.strip().split("\t")
            if len(c_parts) < 18: return None
            sex, vital = c_parts[4].lower(), c_parts[5].lower()
            t_str, f_str = c_parts[16], c_parts[17]
            if not t_str or t_str in ["Unnamed: 16", "nan"]:
                if not f_str or f_str in ["Unnamed: 17", "nan"]: return None
                time_days = float(f_str)
            else: time_days = float(t_str)
            event = 1 if "deceased" in vital else 0
            duration_months = time_days / 30.437
            encoded_gender = 0 if "female" in sex else 1

            if encoded_gender != sex_code:
                return None

            exp_keys = [f for f in files if "exp_seq" in f]
            exp_data = {}
            for ek in exp_keys:
                exp_str = gzip.decompress(download_url(f"{base_url}/{ek}")).decode("utf-8")
                lines = exp_str.strip().split("\n")
                if len(lines[0].split("\t")) < 3 or lines[0].split("\t")[2] not in primary_spec_ids: continue
                for line in lines:
                    parts = line.split("\t")
                    if len(parts) > 8:
                        exp_data[parts[7].split(".")[0]] = float(parts[8])
                break
            if not exp_data: return None
            return {"donor_id": donor_id, "Gender": encoded_gender, "event": event, "duration": duration_months, "expression": exp_data}
        except Exception: return None

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
        if tot_rpkm == 0: tot_rpkm = 1.0

        for sym in genes:
            ensg = GENE_TO_ENSG.get(sym)
            if ensg and ensg in pd_item["expression"]:
                rpkm = pd_item["expression"][ensg]
                tpm = (rpkm / tot_rpkm) * 1e6
                rec[sym] = np.log2(tpm + 1)
        records.append(rec)

    df_icgc = pd.DataFrame(records)
    if df_icgc.empty:
        return None, None

    # Preserve only measured columns matching old pipeline overlapping feature strategy
    avail_genes = [g for g in genes if g in df_icgc.columns and not (df_icgc[g] == 0.0).all()]
    if not avail_genes:
        return None, None

    X = df_icgc[avail_genes]
    y = df_icgc[["event", "duration"]].copy()
    y["event"] = y["event"].astype(bool)

    return X, y


def evaluate_cohort(cohort_name, X_val, y_val, scaler, models, y_train_rec, full_gene_set, cptac_X_train, cptac_y_rec):
    n_pts = len(X_val)
    n_deaths = int(y_val["event"].sum())
    val_cols = list(X_val.columns)
    n_mapped = len([g for g in full_gene_set if g in val_cols and not (X_val[g] == 0.0).all()])

    print(f"\n=======================================================")
    print(f" EVALUATING CPTAC-TRAINED MODEL ON {cohort_name.upper()} ({n_pts} patients, {n_deaths} deaths, {n_mapped}/{len(full_gene_set)} genes mapped)")
    print(f"=======================================================")

    # Handle overlapping feature strategy if evaluation cohort has fewer mapped features (e.g. GSE22541 microarray)
    is_subset = len(val_cols) < len(full_gene_set)
    if is_subset:
        print(f"Cohort uses overlapping feature subset: {val_cols} ({len(val_cols)}/{len(full_gene_set)} genes)")
        scaler_sub = StandardScaler()
        X_tr_sub_scaled = scaler_sub.fit_transform(cptac_X_train[val_cols])
        X_tr_sub_df = pd.DataFrame(X_tr_sub_scaled, columns=val_cols)
        X_val_scaled = scaler_sub.transform(X_val)
        X_val_scaled_df = pd.DataFrame(X_val_scaled, columns=val_cols)

        # Refit models on overlapping subset strictly matching old pipeline convention
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
            auc_dict = compute_time_aucs(y_train_rec, y_val, y_pred)
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
                "Log-Rank p": km_stats["logrank_p"],
                "HR (95% CI)": f"{km_stats['HR']} {km_stats['HR_CI']}"
            })
        except Exception as e:
            print(f"Error evaluating {m_label} on {cohort_name}: {e}")

    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))
    return df_res


def run_external_validation_for_sex(sex_label, sex_code):
    bundle_path = Path(f"./aggregated_results/cptac_frozen_models_{sex_label.lower()}.joblib")
    if not bundle_path.exists():
        print(f"Error: Frozen bundle not found at {bundle_path}.")
        return None

    bundle = joblib.load(bundle_path)
    genes = bundle["genes"]
    scaler = bundle["scaler"]
    models = bundle["models"]

    # Load raw CPTAC training matrix for subset fitting if needed
    clinical = cdp.get_cptac_survival_data()
    tpm_data = cdp.get_cptac_expr("gene_name", "tpm")
    omic_w_clinical = clinical.join(tpm_data, how="inner").groupby(level=0).first()
    sex_clinical = omic_w_clinical[omic_w_clinical["Gender"] == sex_code].copy()
    cptac_X_train = sex_clinical[genes]
    y_cptac = sex_clinical[["event", "duration"]].copy()
    y_cptac["duration"] = y_cptac["duration"] / 30.437
    y_cptac["event"] = y_cptac["event"].astype(bool)
    y_train_rec = y_cptac.to_records(index=False)

    print(f"\n=== Running External Validation for CPTAC {sex_label.upper()} Pipeline ({len(genes)} genes) ===")

    all_rows = []

    # 1. External TCGA Evaluation
    X_tcga, y_tcga = load_tcga_sex_cohort(genes, sex_code)
    if X_tcga is not None and not X_tcga.empty:
        res_tcga = evaluate_cohort(f"TCGA-KIRC ({sex_label.capitalize()})", X_tcga, y_tcga, scaler, models, y_train_rec, genes, cptac_X_train, y_train_rec)
        all_rows.append(res_tcga)

    # 2. External GSE22541 Evaluation
    X_gse, y_gse = load_gse22541_sex_cohort(genes, sex_code)
    if X_gse is not None and not X_gse.empty:
        res_gse = evaluate_cohort(f"GSE22541 ({sex_label.capitalize()})", X_gse, y_gse, scaler, models, y_train_rec, genes, cptac_X_train, y_train_rec)
        all_rows.append(res_gse)

    # 3. External ICGC RECA-EU Evaluation
    X_icgc, y_icgc = load_icgc_sex_cohort(genes, sex_code)
    if X_icgc is not None and not X_icgc.empty:
        res_icgc = evaluate_cohort(f"ICGC RECA-EU ({sex_label.capitalize()})", X_icgc, y_icgc, scaler, models, y_train_rec, genes, cptac_X_train, y_train_rec)
        all_rows.append(res_icgc)

    if all_rows:
        return pd.concat(all_rows, ignore_index=True)
    return None


def main():
    print("=== CPTAC Sex-Specific External Validation Pipeline ===")
    df_male = run_external_validation_for_sex("male", 1)
    df_female = run_external_validation_for_sex("female", 0)

    all_res = []
    if df_male is not None and not df_male.empty: all_res.append(df_male)
    if df_female is not None and not df_female.empty: all_res.append(df_female)

    if all_res:
        final_df = pd.concat(all_res, ignore_index=True)
        out_csv = Path("./aggregated_results/cptac_external_validation_results_sex_specific.csv")
        final_df.to_csv(out_csv, index=False)
        print(f"\nSaved all sex-specific external validation metrics to {out_csv}")


if __name__ == "__main__":
    main()
