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
from sklearn.base import BaseEstimator, TransformerMixin

# Import data_processing without modifying it
import data_processing as dp

warnings.filterwarnings("ignore")

# Define frozen sex-specific 8-gene signatures
male_genes = ["HS3ST1", "RNF183", "LINC00973", "HBB", "CDKN1A", "HMGA2", "HS3ST3A1", "ADAM8"]
female_genes = ["SSTR1", "ACP5", "TNFSF15", "NUDT2", "ACTN2", "CTTNBP2", "SNTG2-AS1", "RNY4P34"]

male_gene_map = {
    "ENSG00000140391": "HS3ST1",
    "ENSG00000169139": "RNF183",
    "ENSG00000240476": "LINC00973",
    "ENSG00000244734": "HBB",
    "ENSG00000124762": "CDKN1A",
    "ENSG00000149948": "HMGA2",
    "ENSG00000171243": "HS3ST3A1",
    "ENSG00000151136": "ADAM8"
}

female_gene_map = {
    "ENSG00000139874": "SSTR1",
    "ENSG00000102575": "ACP5",
    "ENSG00000181634": "TNFSF15",
    "ENSG00000165030": "NUDT2",
    "ENSG00000147403": "ACTN2",
    "ENSG00000137449": "CTTNBP2"
}

tcga_cache = {}

def cindex_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    return concordance_index_censored(y["event"], y["duration"], y_pred)[0]

def get_tcga_training_data(encoded_gender, genes):
    global tcga_cache
    if "tcga_full" not in tcga_cache:
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

        all_sig_genes = set(male_genes + female_genes)
        expr_dict = {}
        for file in tcga_files:
            filename = file.name
            if filename not in metadata_dict:
                continue
            caseid = metadata_dict[filename]
            patient = pd.read_csv(file, sep="\t", skiprows=[0, 2, 3, 4, 5])
            sub = patient[patient["gene_name"].isin(all_sig_genes)].set_index("gene_name")["tpm_unstranded"]
            expr_dict[caseid] = sub

        tcga_sig_df = pd.DataFrame(expr_dict).groupby(level=0).mean().T
        tcga_sig_df = np.log2(tcga_sig_df + 1)
        tcga_sig_df["Gender"] = tcga_sig_df.index.map(lambda cid: clinical.loc[cid, "Gender"])
        tcga_full = clinical[["event", "duration"]].join(tcga_sig_df, how="inner").groupby(level=0).first()
        tcga_cache["tcga_full"] = tcga_full

    tcga_full = tcga_cache["tcga_full"]
    cohort = tcga_full[tcga_full["Gender"] == encoded_gender].drop("Gender", axis=1)
    avail_genes = [g for g in genes if g in cohort.columns]
    X = cohort[avail_genes]
    y = cohort[["event", "duration"]].copy()
    y["duration"] = y["duration"] / 30.437
    y["event"] = y["event"].astype(bool)
    return X, y.to_records(index=False)

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
        print("AUC ERR:", e)
        return {}

# -------------------------------------------------------------
# ICGC RECA-EU Data Parser
# -------------------------------------------------------------
def load_icgc_dataset():
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
    combined_gene_map = {**male_gene_map, **female_gene_map}

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

        for ensg, sym in combined_gene_map.items():
            if ensg in pd_item["expression"]:
                rpkm = pd_item["expression"][ensg]
                tpm = (rpkm / tot_rpkm) * 1e6
                rec[sym] = np.log2(tpm + 1)
        records.append(rec)

    return pd.DataFrame(records)

# -------------------------------------------------------------
# GSE22541 Microarray Data Parser
# -------------------------------------------------------------
class QuantileNorm(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.rank_means = None

    def fit(self, X, y=None):
        if isinstance(X, np.ndarray): X = pd.DataFrame(X)
        sort_in_sample = np.sort(X.values, axis=1)
        self.rank_means = np.mean(sort_in_sample, axis=0)
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray): X = pd.DataFrame(X)
        ranked_x = X.rank(axis=1, method='min').astype(int) - 1
        mapped_means = np.array(self.rank_means)[ranked_x.values]
        return pd.DataFrame(mapped_means, index=ranked_x.index, columns=ranked_x.columns)

def load_gse22541_dataset():
    gpl_path = "GSE22541_RAW/GPL570_annotation.txt"
    probe_to_gene = {}
    with open(gpl_path, "r", encoding="utf-8", errors="ignore") as f:
        found_table = False
        for line in f:
            if line.startswith("!platform_table_begin"):
                found_table = True
                continue
            if line.startswith("!platform_table_end"): break
            if found_table:
                parts = line.strip().split("\t")
                if len(parts) > 10:
                    probe_id = parts[0]
                    gene_symbol = parts[10].strip()
                    if gene_symbol and gene_symbol != "Gene Symbol":
                        symbol = gene_symbol.split("///")[0].strip()
                        probe_to_gene[probe_id] = symbol

    matrix_path = "GSE22541_RAW/GSE22541_series_matrix.txt.gz"
    samples_meta = []
    expression_lines = []
    with gzip.open(matrix_path, "rt", encoding="utf-8", errors="ignore") as f:
        found_data = False
        for line in f:
            if line.startswith("!Sample_geo_accession"):
                parts = line.strip().split("\t")
                geo_ids = [v.strip("\"") for v in parts[1:]]
                for geo in geo_ids: samples_meta.append({"geo_accession": geo})
            elif line.startswith("!Sample_characteristics_ch1"):
                parts = line.strip().split("\t")
                vals = [v.strip("\"") for v in parts[1:]]
                for idx, val in enumerate(vals):
                    if ":" in val:
                        key, value = val.split(":", 1)
                        samples_meta[idx][key.strip().lower()] = value.strip()
                    elif "=" in val:
                        key, value = val.split("=", 1)
                        samples_meta[idx][key.strip().lower()] = value.strip()
            elif line.startswith("!series_matrix_table_begin"):
                found_data = True
                continue
            elif line.startswith("!series_matrix_table_end"): break
            elif found_data: expression_lines.append(line.strip())

    header_parts = expression_lines[0].split("\t")
    sample_columns = [h.strip("\"") for h in header_parts[1:]]
    
    data_rows, row_names = [], []
    for line in expression_lines[1:]:
        parts = line.split("\t")
        row_names.append(parts[0].strip("\""))
        data_rows.append([float(x) for x in parts[1:]])
    
    ex_df = pd.DataFrame(data_rows, index=row_names, columns=sample_columns)
    ex_df = ex_df.loc[ex_df.index.isin(probe_to_gene.keys())]
    ex_df = ex_df.rename(index=probe_to_gene)
    ex_df = ex_df.groupby(level=0).mean()
    ex = ex_df.T

    ex[ex <= 0] = 1
    ex_log = np.log2(ex)
    ex_log_qn = QuantileNorm().fit_transform(ex_log)

    df_meta = pd.DataFrame(samples_meta).set_index("geo_accession")
    primary_meta = df_meta[df_meta["tissue"] == "primary clear-cell renal cell carcinoma"].copy()

    clinical_rows = []
    for geo_id, row in primary_meta.iterrows():
        gender_raw = str(row.get("gender", "")).lower()
        if "female" in gender_raw or gender_raw == "f": sex = 0
        elif "male" in gender_raw or gender_raw == "m": sex = 1
        else: continue
            
        dfs_val = str(row.get("dfs/follow-up", ""))
        if "DFS =" in dfs_val:
            event = 1
            duration = float(dfs_val.split("=")[1].split()[0])
        elif "Follow-up =" in dfs_val:
            event = 0
            duration = float(dfs_val.split("=")[1].split()[0])
        else: continue
            
        clinical_rows.append({
            "case_id": f"GSE22541_{geo_id}",
            "Gender": sex,
            "event": event,
            "duration": duration
        })

    clinical = pd.DataFrame(clinical_rows).set_index("case_id")
    ex_log_qn.index = [f"GSE22541_{idx}" for idx in ex_log_qn.index]
    merged = clinical.join(ex_log_qn, how="inner").reset_index()
    return merged

# -------------------------------------------------------------
# Combined Evaluation with Joe Pizzi's Independent Z-Score Normalization
# -------------------------------------------------------------
def evaluate_combined_cohort(df_icgc, df_gse, genes, encoded_gender):
    gender_str = "Male" if encoded_gender == 1 else "Female"
    
    # Filter datasets by gender
    icgc_sub = df_icgc[df_icgc["Gender"] == encoded_gender].copy()
    gse_sub = df_gse[df_gse["Gender"] == encoded_gender].copy()
    
    avail_genes = [g for g in genes if g in icgc_sub.columns and g in gse_sub.columns]
    
    print(f"\n==========================================================================================")
    print(f" COMBINED ICGC + GSE22541 {gender_str.upper()} COHORT EVALUATION (Pizzi et al. Z-Score Batch Normalization)")
    print(f" ICGC: {len(icgc_sub)} patients | GSE22541: {len(gse_sub)} patients | Total Combined: {len(icgc_sub) + len(gse_sub)} patients ({int(icgc_sub['event'].sum() + gse_sub['event'].sum())} deaths)")
    print(f" Frozen Signature Genes ({len(avail_genes)}): {', '.join(avail_genes)}")
    print(f"==========================================================================================")

    # 1. Independent Z-Score Normalization per dataset (Joe Pizzi's Method)
    scaler_icgc = StandardScaler()
    icgc_scaled_feats = pd.DataFrame(scaler_icgc.fit_transform(icgc_sub[avail_genes]), columns=avail_genes, index=icgc_sub.index)
    icgc_combined = pd.concat([icgc_sub[["Gender", "event", "duration"]], icgc_scaled_feats], axis=1)

    scaler_gse = StandardScaler()
    gse_scaled_feats = pd.DataFrame(scaler_gse.fit_transform(gse_sub[avail_genes]), columns=avail_genes, index=gse_sub.index)
    gse_combined = pd.concat([gse_sub[["Gender", "event", "duration"]], gse_scaled_feats], axis=1)

    # 2. Concatenate the normalized datasets
    combined_df = pd.concat([icgc_combined, gse_combined], axis=0, ignore_index=True)

    # 3. Get TCGA Training Data
    X_train, y_train = get_tcga_training_data(encoded_gender, avail_genes)
    
    scaler_tcga = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler_tcga.fit_transform(X_train), columns=avail_genes)
    X_val_combined = combined_df[avail_genes]
    y_val_combined = combined_df[["event", "duration"]].copy()

    models = {
        "Coxnet (Primary)": coxnet_predict,
        "Gradient Boosting": survival_boost,
        "Survival SVM": svm_survival,
        "Random Forest": rf_survival,
    }

    summary_rows = []
    for m_label, train_func in models.items():
        try:
            model = train_func(X_train_scaled, y_train)
            y_pred = model.predict(X_val_combined)
            c_index = concordance_index_censored(y_val_combined["event"].astype(bool), y_val_combined["duration"], y_pred)[0]
            auc_dict = compute_time_aucs(y_train, y_val_combined, y_pred)
            summary_rows.append({
                "Model": m_label,
                "# Genes": len(avail_genes),
                "C-Index": f"{c_index:.4f}",
                "1-Yr AUC": auc_dict.get("1Yr", "N/A"),
                "3-Yr AUC": auc_dict.get("3Yr", "N/A"),
                "5-Yr AUC": auc_dict.get("5Yr", "N/A")
            })
        except Exception as e:
            print(f"[{m_label}] Failed: {e}")

    df_summary = pd.DataFrame(summary_rows)
    print("\n" + df_summary.to_string(index=False))

if __name__ == "__main__":
    print("Loading ICGC RECA-EU dataset...")
    df_icgc = load_icgc_dataset()
    
    print("Loading GSE22541 microarray dataset...")
    df_gse = load_gse22541_dataset()
    
    # Run evaluation on combined Male cohort (8 genes)
    evaluate_combined_cohort(df_icgc, df_gse, male_genes, 1)
    
    # Run evaluation on combined Female cohort (6 coding genes)
    evaluate_combined_cohort(df_icgc, df_gse, female_genes, 0)
