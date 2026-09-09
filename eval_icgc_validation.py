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

# Import data_processing without modifying it
import data_processing as dp

warnings.filterwarnings("ignore")

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
        # Load clean KIRC clinical data (479 cases: 318 male, 161 female)
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

# Model training functions using cross-validation (identical to eval_cptac_validation.py)
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

    print("Listing S3 bucket files for RECA-EU...")
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

    print("Downloading and parsing ICGC donor files in parallel...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        results = executor.map(process_donor, valid_donors)
        parsed_donors = [r for r in results if r]

    print(f"Successfully parsed {len(parsed_donors)} primary tumor samples.")
    records = []
    combined_gene_map = {**male_gene_map, **female_gene_map}

    for pd_item in parsed_donors:
        rec = {
            "case_id": pd_item["donor_id"],
            "Gender": pd_item["Gender"],
            "event": pd_item["event"],
            "duration": pd_item["duration"]
        }
        # RPKM to TPM conversion
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

def evaluate_icgc_cohort(df_val, genes, encoded_gender):
    gender_str = "Male" if encoded_gender == 1 else "Female"
    df_val_cohort = df_val[df_val["Gender"] == encoded_gender]
    
    avail_genes = [g for g in genes if g in df_val_cohort.columns]
    if len(df_val_cohort) == 0 or len(avail_genes) == 0:
        print(f"No records/genes found for {gender_str} cohort.")
        return

    print(f"\n=======================================================")
    print(f" EVALUATING ICGC RECA-EU {gender_str.upper()} COHORT ({len(df_val_cohort)} patients, {int(df_val_cohort['event'].sum())} deaths, {len(avail_genes)} genes)")
    print(f"=======================================================")

    X_train, y_train = get_tcga_training_data(encoded_gender, avail_genes)
    X_val = df_val_cohort[avail_genes]
    y_val = df_val_cohort[["event", "duration"]].copy()

    models = {
        "Coxnet (Primary)": coxnet_predict,
        "Gradient Boosting": survival_boost,
        "Survival SVM": svm_survival,
        "Random Forest": rf_survival,
    }

    scaler_pipe = StandardScaler()
    X_train_scaled = scaler_pipe.fit_transform(X_train)
    X_val_scaled = scaler_pipe.transform(X_val)
    X_train_df = pd.DataFrame(X_train_scaled, columns=avail_genes)
    X_val_df = pd.DataFrame(X_val_scaled, columns=avail_genes)

    summary_rows = []
    for m_label, train_func in models.items():
        try:
            model = train_func(X_train_df, y_train)
            y_pred = model.predict(X_val_df)
            c_index = concordance_index_censored(y_val["event"].astype(bool), y_val["duration"], y_pred)[0]
            auc_dict = compute_time_aucs(y_train, y_val, y_pred)
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
    df_icgc = load_icgc_dataset()
    evaluate_icgc_cohort(df_icgc, male_genes, 1)
    evaluate_icgc_cohort(df_icgc, female_genes, 0)
