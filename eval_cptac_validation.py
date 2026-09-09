import os
import json
import warnings
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
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning

# Import KIRC data processing functions
import data_processing as dp

warnings.filterwarnings("ignore")

# Define signature genes
male_genes = ["HS3ST1", "RNF183", "LINC00973", "HBB", "CDKN1A", "HMGA2", "HS3ST3A1", "ADAM8"]
female_genes = ["SSTR1", "ACP5", "TNFSF15", "NUDT2", "ACTN2", "CTTNBP2", "SNTG2-AS1", "RNY4P34"]

# Map genes to their Ensembl IDs where necessary, but we will extract by gene_name directly from CPTAC files!
male_gene_set = set(male_genes)
female_gene_set = set(female_genes)

# Cache for TCGA expression data
tcga_cache = {}

def cindex_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    return concordance_index_censored(y["event"], y["duration"], y_pred)[0]

def get_tcga_training_data(encoded_gender, genes):
    global tcga_cache
    clinical = dp.get_survival_data()
    if "tpm" not in tcga_cache:
        tcga_cache["tpm"] = dp.get_tcga_expr("gene_name", "tpm")
    data = tcga_cache["tpm"]
    omic_w_clinical = clinical.join(data, how="inner")
    omic_w_clinical = omic_w_clinical.groupby(omic_w_clinical.index).first()
    data = omic_w_clinical[omic_w_clinical["Gender"] == encoded_gender].drop("Gender", axis=1)
    
    avail_genes = [g for g in genes if g in data.columns]
    X = data[avail_genes]
    y = data[["event", "duration"]].copy()
    y["duration"] = y["duration"] / 30.437
    y["event"] = y["event"].astype(bool)
    y = y.to_records(index=False)
    return X, y

# Model training functions using cross-validation (same as original pipeline)
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

def load_cptac_dataset():
    # 1. Load clinical data
    clinical_path = Path("./CPTAC_KIRC/Metadata/clinical.tsv")
    df_clinical = dp.load_clean_kirc_clinical(clinical_path)
    clean_case_ids = set(df_clinical.index)
    
    # 2. Load biospecimen map
    with open("./CPTAC_KIRC/Metadata/cptac_biospecimen_map.json", "r") as f:
        biospec_map = json.load(f)
        
    # Map case IDs to expression files (picking the first alphabetically if duplicates exist)
    case_to_file = {}
    for fname, info in biospec_map.items():
        case_id = info["case_id"]
        if case_id in clean_case_ids:
            if case_id not in case_to_file:
                case_to_file[case_id] = fname
            else:
                case_to_file[case_id] = min(case_to_file[case_id], fname)
                
    # 3. Parse expression data for signature genes
    expression_dir = Path("./CPTAC_KIRC/CPTAC_RNA_seq")
    records = []
    
    all_sig_genes = male_gene_set.union(female_gene_set)
    
    for case_id, fname in case_to_file.items():
        fpath = expression_dir / fname
        if not fpath.exists():
            continue
            
        df_expr = pd.read_csv(fpath, sep="\t", skiprows=1)
        # Filter for signature genes
        df_expr_sig = df_expr[df_expr["gene_name"].isin(all_sig_genes)]
        
        # Build patient expression record
        patient_record = {
            "case_id": case_id,
            "event": df_clinical.loc[case_id, "event"],
            "duration": df_clinical.loc[case_id, "duration"] / 30.437, # convert days to months
            "Gender": df_clinical.loc[case_id, "Gender"]
        }
        
        for _, row in df_expr_sig.iterrows():
            gname = row["gene_name"]
            tpm = float(row["tpm_unstranded"])
            # Apply log2(TPM + 1)
            patient_record[gname] = np.log2(tpm + 1)
            
        records.append(patient_record)
        
    df_val = pd.DataFrame(records)
    print(f"Successfully compiled expression matrix for {len(df_val)} matching CPTAC cases.")
    return df_val

def evaluate_cptac_cohort(df_val, genes, encoded_gender):
    gender_str = "Male" if encoded_gender == 1 else "Female"
    
    # Stratify validation cohort by gender
    df_val_cohort = df_val[df_val["Gender"] == encoded_gender]
    if len(df_val_cohort) == 0:
        print(f"No records found for {gender_str} cohort.")
        return
        
    print(f"\n=======================================================")
    print(f" EVALUATING {gender_str.upper()} COHORT ({len(df_val_cohort)} patients, {int(df_val_cohort['event'].sum())} deaths)")
    print(f"=======================================================")
    
    # Load TCGA training data
    X_train, y_train = get_tcga_training_data(encoded_gender, genes)
    X_val = df_val_cohort[genes]
    y_val = df_val_cohort[["event", "duration"]].copy()
    y_val["event"] = y_val["event"].astype(bool)
    
    models = {
        "Coxnet (Primary)": coxnet_predict,
        "Gradient Boosting": survival_boost,
        "Survival SVM": svm_survival,
        "Random Forest": rf_survival,
    }
    
    scaler_pipe = StandardScaler()
    X_train_scaled_p = scaler_pipe.fit_transform(X_train)
    X_val_scaled_p = scaler_pipe.transform(X_val)
    X_train_df_p = pd.DataFrame(X_train_scaled_p, columns=genes)
    X_val_df_p = pd.DataFrame(X_val_scaled_p, columns=genes)

    summary_rows = []

    for m_label, train_func in models.items():
        try:
            model = train_func(X_train_df_p, y_train)
            y_pred = model.predict(X_val_df_p)
                
            c_index = concordance_index_censored(y_val["event"], y_val["duration"], y_pred)[0]
            
            # Compute Time AUCs
            auc_dict = compute_time_aucs(y_train, y_val, y_pred)
            
            summary_rows.append({
                "Model": m_label,
                "Scaling Method": "Method A (Pipeline)",
                "C-Index": f"{c_index:.4f}",
                "1-Yr AUC": auc_dict.get("1Yr", "N/A"),
                "3-Yr AUC": auc_dict.get("3Yr", "N/A"),
                "5-Yr AUC": auc_dict.get("5Yr", "N/A")
            })
        except Exception as e:
            print(f"[{m_label}] Failed: {e}")

    # Print clean ASCII table
    df_summary = pd.DataFrame(summary_rows)
    print("\n" + df_summary.to_string(index=False))

def compute_time_aucs(y_train, y_val_df, y_pred, time_points=[12.0, 36.0, 60.0]):
    # Convert y_val
    y_val_rec = np.empty(len(y_val_df), dtype=[("event", bool), ("duration", float)])
    y_val_rec["event"] = y_val_df["event"].astype(bool).values
    y_val_rec["duration"] = y_val_df["duration"].astype(float).values
    
    # Convert y_train
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




if __name__ == "__main__":
    df_cptac = load_cptac_dataset()
    evaluate_cptac_cohort(df_cptac, male_genes, 1)
    evaluate_cptac_cohort(df_cptac, female_genes, 0)


