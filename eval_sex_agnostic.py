import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import gzip
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import concurrent.futures
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, ComponentwiseGradientBoostingSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning

# Import data_processing canonical module
import data_processing as dp

warnings.filterwarnings("ignore")

def cindex_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    return concordance_index_censored(y["event"], y["duration"], y_pred)[0]

def compute_time_aucs(y_train, y_val_df, y_pred, time_points=[12.0, 36.0, 60.0]):
    y_val_rec = np.empty(len(y_val_df), dtype=[("event", bool), ("duration", float)])
    y_val_rec["event"] = y_val_df["event"].astype(bool).values
    y_val_rec["duration"] = y_val_df["duration"].astype(float).values
    
    if isinstance(y_train, pd.DataFrame):
        y_train_rec = np.empty(len(y_train), dtype=[("event", bool), ("duration", float)])
        y_train_rec["event"] = y_train["event"].astype(bool).values
        y_train_rec["duration"] = y_train["duration"].astype(float).values
    else:
        y_train_rec = np.empty(len(y_train), dtype=[("event", bool), ("duration", float)])
        y_train_rec["event"] = y_train["event"].astype(bool)
        y_train_rec["duration"] = y_train["duration"].astype(float)
        if y_train_rec["duration"].max() > 300:
            y_train_rec["duration"] = y_train_rec["duration"] / 30.437

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
    except Exception as e:
        return {}

def coxnet_predict(X, y):
    best_score = -np.inf
    best_result = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=FitFailedWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        for l1_ratio in [0.1, 0.3, 0.5, 1.0]:
            try:
                base_pipe = make_pipeline(
                    StandardScaler(),
                    CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alpha_min_ratio=0.01, max_iter=300000, tol=1e-6)
                )
                base_pipe.fit(X, y)
                estimated_alphas = base_pipe.named_steps["coxnetsurvivalanalysis"].alphas_
                estimated_alphas = estimated_alphas[estimated_alphas > 0.001]
                if len(estimated_alphas) == 0:
                    continue
                folds = KFold(n_splits=5, shuffle=True, random_state=3)
                gcv = GridSearchCV(
                    make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, max_iter=300000)),
                    param_grid={"coxnetsurvivalanalysis__alphas": [[a] for a in estimated_alphas]},
                    cv=folds, n_jobs=-1, error_score=np.nan, scoring=cindex_scorer
                )
                gcv.fit(X, y)
                if gcv.best_score_ > best_score:
                    best_score = gcv.best_score_
                    best_result = gcv.best_estimator_
            except Exception:
                continue
    return best_result

def rf_survival(X, y):
    folds = KFold(n_splits=5, shuffle=True, random_state=3)
    gcv = GridSearchCV(
        RandomSurvivalForest(random_state=3),
        param_grid={"n_estimators": [50, 100], "max_depth": [3, 4, 5], "min_samples_split": [5, 10]},
        cv=folds, n_jobs=-1, scoring=cindex_scorer
    )
    gcv.fit(X, y)
    return gcv.best_estimator_

def svm_survival(X, y):
    folds = KFold(n_splits=5, shuffle=True, random_state=3)
    gcv = GridSearchCV(
        make_pipeline(StandardScaler(), FastSurvivalSVM(max_iter=1000, random_state=3)),
        param_grid={"fastsurvivalsvm__alpha": [0.1, 1.0, 10.0, 100.0]},
        cv=folds, n_jobs=-1, scoring=cindex_scorer
    )
    gcv.fit(X, y)
    return gcv.best_estimator_

def survival_boost(X, y):
    folds = KFold(n_splits=5, shuffle=True, random_state=3)
    gcv = GridSearchCV(
        ComponentwiseGradientBoostingSurvivalAnalysis(random_state=3),
        param_grid={"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]},
        cv=folds, n_jobs=-1, scoring=cindex_scorer
    )
    gcv.fit(X, y)
    return gcv.best_estimator_

# ------------------------------------------------------------------
# TCGA Data Loader
# ------------------------------------------------------------------
def get_tcga_data(genes):
    clinical = dp.load_clean_kirc_clinical()
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
        expr_dict[caseid] = sub

    tcga_sig_df = pd.DataFrame(expr_dict).groupby(level=0).mean().T
    tcga_sig_df = np.log2(tcga_sig_df + 1)
    tcga_sig_df["Gender"] = tcga_sig_df.index.map(lambda cid: clinical.loc[cid, "Gender"])
    tcga_full = clinical[["event", "duration"]].join(tcga_sig_df, how="inner").groupby(level=0).first()

    avail_genes = [g for g in genes if g in tcga_full.columns]
    X = tcga_full[avail_genes]
    y = tcga_full[["event", "duration"]].copy()
    y["duration"] = y["duration"] / 30.437
    y["event"] = y["event"].astype(bool)
    genders = tcga_full["Gender"]
    return X, y, genders, avail_genes

# ------------------------------------------------------------------
# External Data Loaders
# ------------------------------------------------------------------
def load_cptac_dataset(genes):
    df_clinical = dp.load_clean_cptac_clinical()
    cptac_dir = Path("./CPTAC_KIRC")
    cptac_files = list(cptac_dir.glob("*.tsv"))
    records = []
    for fpath in cptac_files:
        case_id = fpath.stem.split(".")[0]
        if case_id not in df_clinical.index:
            continue
        df_expr = pd.read_csv(fpath, sep="\t", skiprows=[0, 2, 3, 4, 5])
        df_expr_sig = df_expr[df_expr["gene_name"].isin(genes)]
        patient_record = {
            "case_id": case_id,
            "event": df_clinical.loc[case_id, "event"],
            "duration": df_clinical.loc[case_id, "duration"] / 30.437,
            "Gender": df_clinical.loc[case_id, "Gender"]
        }
        for _, row in df_expr_sig.iterrows():
            gname = row["gene_name"]
            tpm = float(row["tpm_unstranded"])
            patient_record[gname] = np.log2(tpm + 1)
        records.append(patient_record)
    return pd.DataFrame(records)

def load_icgc_dataset(genes):
    gene_map = {
        "ENSG00000140391": "HS3ST1", "ENSG00000169139": "RNF183", "ENSG00000240476": "LINC00973",
        "ENSG00000244734": "HBB", "ENSG00000124762": "CDKN1A", "ENSG00000149948": "HMGA2",
        "ENSG00000171243": "HS3ST3A1", "ENSG00000151136": "ADAM8", "ENSG00000139874": "SSTR1",
        "ENSG00000102575": "ACP5", "ENSG00000181634": "TNFSF15", "ENSG00000165030": "NUDT2",
        "ENSG00000147403": "ACTN2", "ENSG00000137449": "CTTNBP2"
    }
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

    all_keys = get_all_keys("release_28/data/RECA-EU/")
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

        for ensg, sym in gene_map.items():
            if sym in genes and ensg in pd_item["expression"]:
                rpkm = pd_item["expression"][ensg]
                tpm = (rpkm / tot_rpkm) * 1e6
                rec[sym] = np.log2(tpm + 1)
        records.append(rec)

    return pd.DataFrame(records)

# ------------------------------------------------------------------
# Evaluation Procedures
# ------------------------------------------------------------------
def evaluate_tcga_internal(X, y, genders, genes):
    y_rec = y.to_records(index=False)
    n_patients = len(X)
    
    print("\n==========================================================================================")
    print(f" 1. TCGA-KIRC SEX-AGNOSTIC INTERNAL CV ({n_patients} Patients: 318 Males, 161 Females | {int(y['event'].sum())} Deaths)")
    print(f" Frozen Sex-Agnostic Signature ({len(genes)} Genes): {', '.join(genes)}")
    print("==========================================================================================")

    models = {
        "Coxnet (Primary)": coxnet_predict,
        "Gradient Boosting": survival_boost,
        "Survival SVM": svm_survival,
        "Random Forest": rf_survival,
    }

    outer_folds = KFold(n_splits=5, shuffle=True, random_state=3)
    summary_rows = []

    for m_label, train_func in models.items():
        oof_preds = np.zeros(n_patients)
        for fold_idx, (train_idx, val_idx) in enumerate(outer_folds.split(X, y_rec)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y_rec[train_idx], y_rec[val_idx]
            
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=genes)
            X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=genes)
            
            model = train_func(X_train_scaled, y_train)
            oof_preds[val_idx] = model.predict(X_val_scaled)

        # Overall C-index
        c_all = concordance_index_censored(y["event"], y["duration"], oof_preds)[0]
        auc_dict_all = compute_time_aucs(y_rec, y, oof_preds)

        # Male subgroup
        m_mask = (genders == 1).values
        c_male = concordance_index_censored(y["event"].values[m_mask], y["duration"].values[m_mask], oof_preds[m_mask])[0]

        # Female subgroup
        f_mask = (genders == 0).values
        c_female = concordance_index_censored(y["event"].values[f_mask], y["duration"].values[f_mask], oof_preds[f_mask])[0]

        summary_rows.append({
            "Model": m_label,
            "Overall C-Index": f"{c_all:.4f}",
            "Male Subgroup C-Index": f"{c_male:.4f}",
            "Female Subgroup C-Index": f"{c_female:.4f}",
            "1-Yr AUC": auc_dict_all.get("1Yr", "N/A"),
            "3-Yr AUC": auc_dict_all.get("3Yr", "N/A"),
            "5-Yr AUC": auc_dict_all.get("5Yr", "N/A")
        })

    df_summary = pd.DataFrame(summary_rows)
    print("\n" + df_summary.to_string(index=False))

def evaluate_external_method_a(cohort_name, df_val, X_train, y_train, genes):
    avail_genes = [g for g in genes if g in df_val.columns]
    df_val_clean = df_val.dropna(subset=avail_genes + ["event", "duration"]).copy()
    
    print("\n==========================================================================================")
    print(f" EXTERNAL VALIDATION (Method A Scaling): {cohort_name.upper()} ({len(df_val_clean)} Patients | {int(df_val_clean['event'].sum())} Deaths)")
    print(f" Signature Genes ({len(avail_genes)}): {', '.join(avail_genes)}")
    print("==========================================================================================")

    y_train_rec = y_train.to_records(index=False)
    X_val = df_val_clean[avail_genes]
    y_val = df_val_clean[["event", "duration"]].copy()
    y_val["event"] = y_val["event"].astype(bool)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train[avail_genes]), columns=avail_genes)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=avail_genes)

    models = {
        "Coxnet (Primary)": coxnet_predict,
        "Gradient Boosting": survival_boost,
        "Survival SVM": svm_survival,
        "Random Forest": rf_survival,
    }

    summary_rows = []
    genders = df_val_clean["Gender"] if "Gender" in df_val_clean.columns else None

    for m_label, train_func in models.items():
        try:
            model = train_func(X_train_scaled, y_train_rec)
            y_pred = model.predict(X_val_scaled)

            c_all = concordance_index_censored(y_val["event"], y_val["duration"], y_pred)[0]
            auc_dict_all = compute_time_aucs(y_train_rec, y_val, y_pred)

            c_male, c_female = "N/A", "N/A"
            if genders is not None:
                m_mask = (genders == 1).values
                f_mask = (genders == 0).values
                if m_mask.sum() > 0:
                    c_male = f"{concordance_index_censored(y_val['event'].values[m_mask], y_val['duration'].values[m_mask], y_pred[m_mask])[0]:.4f}"
                if f_mask.sum() > 0:
                    c_female = f"{concordance_index_censored(y_val['event'].values[f_mask], y_val['duration'].values[f_mask], y_pred[f_mask])[0]:.4f}"

            summary_rows.append({
                "Model": m_label,
                "Overall C-Index": f"{c_all:.4f}",
                "Male Subgroup C-Index": c_male,
                "Female Subgroup C-Index": c_female,
                "1-Yr AUC": auc_dict_all.get("1Yr", "N/A"),
                "3-Yr AUC": auc_dict_all.get("3Yr", "N/A"),
                "5-Yr AUC": auc_dict_all.get("5Yr", "N/A")
            })
        except Exception as e:
            print(f"[{m_label}] Failed: {e}")

    df_summary = pd.DataFrame(summary_rows)
    print("\n" + df_summary.to_string(index=False))

# ------------------------------------------------------------------
# Main Orchestrator
# ------------------------------------------------------------------
def main():
    ranking_file = Path("./aggregated_results/sex_agnostic_aggregated_ranking.csv")
    if not ranking_file.exists():
        print(f"Error: {ranking_file} not found. Run execution_sex_agnostic.py first.")
        return

    rank_df = pd.read_csv(ranking_file)
    top_8_genes = rank_df["Name"].head(8).tolist()

    X_tcga, y_tcga, genders_tcga, avail_genes = get_tcga_data(top_8_genes)
    
    # 1. TCGA Internal CV Evaluation
    evaluate_tcga_internal(X_tcga, y_tcga, genders_tcga, avail_genes)

    # 2. CPTAC-3 External Validation
    print("\nLoading CPTAC-3 dataset...")
    df_cptac = load_cptac_dataset(avail_genes)
    if len(df_cptac) > 0:
        evaluate_external_method_a("CPTAC-3 RNA-seq", df_cptac, X_tcga, y_tcga, avail_genes)

    # 3. ICGC RECA-EU External Validation
    print("\nLoading ICGC RECA-EU dataset...")
    df_icgc = load_icgc_dataset(avail_genes)
    if len(df_icgc) > 0:
        evaluate_external_method_a("ICGC RECA-EU RNA-seq", df_icgc, X_tcga, y_tcga, avail_genes)

if __name__ == "__main__":
    main()
