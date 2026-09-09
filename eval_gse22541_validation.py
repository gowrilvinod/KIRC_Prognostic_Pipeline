import argparse 
import gzip
from data_processing import *
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning
import warnings 
import pandas as pd 
from sklearn.pipeline import make_pipeline 
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import GridSearchCV, KFold
from ensemble import cindex_scorer
import numpy as np 
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
from sklearn.base import BaseEstimator, TransformerMixin
from sksurv.ensemble import RandomSurvivalForest
from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM

tcga_cache = {}

def eval_gse22541_cohort(encoded_gender):
    global tcga_cache
    gender_str = "Male" if encoded_gender == 1 else "Female"

    # get survival information for TCGA (vital status and survival time in months)
    clinical = get_survival_data()

    # load expression matrix
    if "expr" not in tcga_cache:
        tcga_cache["expr"] = get_tcga_expr("gene_name", "tpm")
    data = tcga_cache["expr"]

    # merge clinical and omics data
    omic_w_clinical = clinical.join(data, how="inner")
    omic_w_clinical = omic_w_clinical.groupby(omic_w_clinical.index).first()

    # split by gender
    data = omic_w_clinical[omic_w_clinical["Gender"] == encoded_gender].drop("Gender", axis=1)

    # drop survival information from omics data
    X = data.drop(["duration", "event"], axis=1)
    y = data[["event", "duration"]].copy()
    y["duration"] = y["duration"] / 30.437
    y["event"] = y["event"].astype(bool)
    y = y.to_records(index=False)

    # get aggregated rankings from csv file 
    if encoded_gender == 1: 
        aggy = pd.read_csv(Path(f"./aggregated_results/male_expr_aggregated_ranking.csv"))
    else: 
        aggy = pd.read_csv(Path(f"./aggregated_results/female_expr_aggregated_ranking.csv"))

    selected_genes = aggy["Name"].to_list()

    # load external dataset
    X_ex, y_ex = get_external_expr(encoded_gender)

    # get overlapping genes while preserving rank order 
    overlap = [g for g in selected_genes if g in X_ex.columns]
    num_genes = len(overlap)
    subset = overlap[:num_genes]

    X_selected = X[subset]
    X_ex_sub = X_ex[subset]

    models = {
        "Coxnet (Primary)": "coxnet",
        "Gradient Boosting": "boost",
        "Survival SVM": "svm",
        "Random Forest": "rf",
    }

    summary_rows = []

    for m_label, eval_model in models.items():
        if eval_model == "coxnet": 
            best_pipe = coxnet_predict(X_selected, y)
        elif eval_model == "rf": 
            best_pipe = rf_survival(X_selected, y)
        elif eval_model == "svm": 
            best_pipe = svm_survival(X_selected, y)
        elif eval_model == "boost": 
            best_pipe = survival_boost(X_selected, y)

        y_pred = best_pipe.predict(X_ex_sub)

        c_index = concordance_index_censored(y_ex["event"], y_ex["duration"], y_pred)[0]
        auc, mean_auc = cumulative_dynamic_auc(y, y_ex, y_pred, np.array([12, 36, 60]))

        summary_rows.append({
            "Model": m_label,
            "# Genes": num_genes,
            "C-Index": f"{c_index:.4f}",
            "1-Yr AUC": f"{auc[0]:.3f}",
            "3-Yr AUC": f"{auc[1]:.3f}",
            "5-Yr AUC": f"{auc[2]:.3f}"
        })

    print(f"\n=======================================================")
    print(f" EVALUATING GSE22541 {gender_str.upper()} COHORT ({len(X_ex_sub)} patients, {int(y_ex['event'].sum())} deaths, {num_genes} genes)")
    print(f"=======================================================")
    df_summary = pd.DataFrame(summary_rows)
    print("\n" + df_summary.to_string(index=False))


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
                    CoxnetSurvivalAnalysis(
                        l1_ratio=l1_ratio,
                        alpha_min_ratio=0.01,
                        max_iter=300000,
                        tol=1e-6
                    )
                )
                base_pipe.fit(X, y)

                estimated_alphas = base_pipe.named_steps["coxnetsurvivalanalysis"].alphas_
                estimated_alphas = estimated_alphas[estimated_alphas > 0.001]  

                if len(estimated_alphas) == 0:
                    continue  

                folds = KFold(n_splits=5, shuffle=True, random_state=3)

                gcv = GridSearchCV(
                    make_pipeline(
                        StandardScaler(),
                        CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, max_iter=300000)
                    ),
                    param_grid={"coxnetsurvivalanalysis__alphas": [[a] for a in estimated_alphas]},
                    cv=folds,
                    n_jobs=-1,
                    error_score=np.nan,  
                    scoring=cindex_scorer
                )

                gcv.fit(X, y)

                if not np.isnan(gcv.best_score_) and gcv.best_score_ > best_score:
                    best_score = gcv.best_score_
                    best_result = gcv
                    
            except (ArithmeticError, ValueError) as e:
                continue

    best_pipe = best_result.best_estimator_
    return best_pipe


def rf_survival(X, y): 
    folds = KFold(n_splits=5, shuffle=True, random_state=3)

    rsf = RandomSurvivalForest(
        n_estimators=500,
        n_jobs=18,
        random_state=folds.random_state,
        bootstrap=True
    )

    pipe = make_pipeline(StandardScaler(), rsf)

    param_grid = {
        "randomsurvivalforest__min_samples_split": [5, 10, 20],
        "randomsurvivalforest__min_samples_leaf": [5, 10, 20],
        "randomsurvivalforest__max_features": ["sqrt", 0.3, 0.5],
    }

    gcv = GridSearchCV(
        pipe,
        param_grid,
        cv=folds,
        n_jobs=1,
        scoring=cindex_scorer
    )

    gcv.fit(X, y)
    best_rsf = gcv.best_estimator_
    return best_rsf


def svm_survival(X, y): 
    folds = KFold(n_splits=5, shuffle=True, random_state=3)

    ssvm = make_pipeline(
        StandardScaler(), 
        FastSurvivalSVM(
            rank_ratio=1.0,
            max_iter=1000,
            tol=1e-5,
            random_state=folds.random_state
        )
    )

    param_grid = {
        "fastsurvivalsvm__alpha": [2.0**v for v in range(-8, 9, 2)],
    }

    gcv = GridSearchCV(
        ssvm,
        param_grid,
        cv=folds,
        n_jobs=18, 
        scoring=cindex_scorer
    )
    
    gcv.fit(X, y)
    best_svm = gcv.best_estimator_
    return best_svm


def survival_boost(X, y): 
    folds = KFold(n_splits=5, shuffle=True, random_state=3)

    est = make_pipeline(StandardScaler(), ComponentwiseGradientBoostingSurvivalAnalysis(
        loss="coxph",
        random_state=folds.random_state
    ))

    param_grid = {
        "componentwisegradientboostingsurvivalanalysis__learning_rate": [0.05, 0.1],
        "componentwisegradientboostingsurvivalanalysis__n_estimators": [100, 200, 400],
        "componentwisegradientboostingsurvivalanalysis__subsample": [0.5, 1.0],
        "componentwisegradientboostingsurvivalanalysis__dropout_rate": [0.0, 0.1],
    }

    gcv = GridSearchCV(
        est,
        param_grid=param_grid,
        cv=folds,
        n_jobs=18,
        scoring=cindex_scorer
    )

    gcv.fit(X, y)
    best_model = gcv.best_estimator_
    return best_model


class QuantileNorm(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.rank_means = None

    def fit(self, X, y=None):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        elif not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be numpy array or pandas dataframe...")
        sort_in_sample = np.sort(X.values, axis=1)
        self.rank_means = np.mean(sort_in_sample, axis=0)
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        elif not isinstance(X, pd.DataFrame):
            raise ValueError("Input should be a ndarray or dataframe")
        ranked_x = X.rank(axis=1, method='min').astype(int) - 1
        mapped_means = np.array(self.rank_means)[ranked_x.values]
        normed_x = pd.DataFrame(mapped_means, index=ranked_x.index, columns=ranked_x.columns)
        return normed_x


def get_external_expr(encoded_gender): 
    gpl_path = "GSE22541_RAW/GPL570_annotation.txt"
    probe_to_gene = {}
    with open(gpl_path, "r", encoding="utf-8", errors="ignore") as f:
        found_table = False
        for line in f:
            if line.startswith("!platform_table_begin"):
                found_table = True
                continue
            if line.startswith("!platform_table_end"):
                break
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
                for geo in geo_ids:
                    samples_meta.append({"geo_accession": geo})
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
            elif line.startswith("!series_matrix_table_end"):
                break
            elif found_data:
                expression_lines.append(line.strip())

    header_parts = expression_lines[0].split("\t")
    sample_columns = [h.strip("\"") for h in header_parts[1:]]
    
    data_rows = []
    row_names = []
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
        if "female" in gender_raw or gender_raw == "f":
            sex = "F"
        elif "male" in gender_raw or gender_raw == "m":
            sex = "M"
        else:
            continue
            
        dfs_val = str(row.get("dfs/follow-up", ""))
        if "DFS =" in dfs_val:
            event = True
            duration = float(dfs_val.split("=")[1].split()[0])
        elif "Follow-up =" in dfs_val:
            event = False
            duration = float(dfs_val.split("=")[1].split()[0])
        else:
            continue
            
        clinical_rows.append({
            "geo_accession": geo_id,
            "sex": sex,
            "event": event,
            "duration": duration
        })

    clinical = pd.DataFrame(clinical_rows).set_index("geo_accession")
    merged = clinical.join(ex_log_qn, how="inner")
    target_sex = "M" if encoded_gender == 1 else "F"
    cohort = merged[merged["sex"] == target_sex]

    X = cohort.drop(["sex", "event", "duration"], axis=1)
    y = cohort[["event", "duration"]].copy()
    y["event"] = y["event"].astype(bool)
    y = y.to_records(index=False)

    return X, y


if __name__ == "__main__":
    eval_gse22541_cohort(1)
    eval_gse22541_cohort(0)
