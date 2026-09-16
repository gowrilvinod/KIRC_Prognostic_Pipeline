import os
import glob
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import beta
from sklearn.model_selection import KFold
from lifelines import CoxPHFitter

# Import canonical modules
import data_processing as dp
from ensemble import coxnet, rf_survival, survival_booster, survival_svm

warnings.filterwarnings("ignore")

def fit_single_gene_cox(args):
    gene, duration, event, gene_vals = args
    try:
        df_sub = pd.DataFrame({"duration": duration, "event": event, "gene": gene_vals})
        cph = CoxPHFitter(penalizer=0.01)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cph.fit(df_sub, duration_col="duration", event_col="event")
        p_val = cph.summary.loc["gene", "p"]
        return {"variable": gene, "p_value": p_val}
    except Exception:
        return None

def univariate_cox(train_df):
    """
    Parallelized univariate Cox proportional hazards regression for candidate genes.
    """
    duration = train_df["duration"].values
    event = train_df["event"].values.astype(bool)
    feature_cols = [c for c in train_df.columns if c not in ["duration", "event"]]
    
    tasks = [(g, duration, event, train_df[g].values) for g in feature_cols]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(fit_single_gene_cox, tasks))
        
    valid_results = [r for r in results if r is not None]
    return pd.DataFrame(valid_results)

def robust_rank_aggregation(rank_lists, all_items):
    """
    Kolde's Robust Rank Aggregation (RRA) algorithm.
    """
    m = len(rank_lists)
    if m == 0:
        return pd.DataFrame(columns=["Name", "Score"])
        
    scores = []
    normalized_ranks = {item: [] for item in all_items}
        
    for r_list in rank_lists:
        L = len(r_list)
        rank_lookup = {item: idx+1 for idx, item in enumerate(r_list)}
        for item in all_items:
            if item in rank_lookup:
                normalized_ranks[item].append(rank_lookup[item] / L)
            else:
                normalized_ranks[item].append(1.0)
                
    for item in all_items:
        r_vals = sorted(normalized_ranks[item])
        min_p = 1.0
        for k in range(1, m + 1):
            u_k = r_vals[k - 1]
            p_k = beta.cdf(u_k, k, m - k + 1)
            if p_k < min_p:
                min_p = p_k
                
        adjusted_p = min(1.0, min_p * m)
        scores.append({"Name": item, "Score": adjusted_p})
        
    result_df = pd.DataFrame(scores).sort_values(by="Score", ascending=True)
    return result_df

def run_sex_agnostic_discovery(n_repeats=20):
    print(f"=======================================================================")
    print(f" TCGA-KIRC SEX-AGNOSTIC DISCOVERY (Canonical Filtering & Preprocessing Parity)")
    print(f" {n_repeats} Repeats x 5 Folds = {n_repeats*5} Fold-Iterations")
    print(f"=======================================================================")

    # 1. Load exact canonical clean KIRC clinical data and TPM expression data
    clinical = dp.load_clean_kirc_clinical()
    tpm_data = dp.get_tcga_expr("gene_name", "tpm")
    omic_w_clinical = clinical.join(tpm_data, how="inner")
    omic_w_clinical = omic_w_clinical.groupby(omic_w_clinical.index).first()

    # Combine ALL 479 patients (318 males + 161 females) without sex splitting or sex features
    data = omic_w_clinical.drop(["Gender"], axis=1, errors="ignore")
    X = data.drop(["duration", "event"], axis=1)
    y_df = data[["event", "duration"]].copy()
    y_df["duration"] = y_df["duration"] / 30.437  # Convert days to months (canonical overall survival endpoint)
    y_df["event"] = y_df["event"].astype(bool)
    y_rec = y_df.to_records(index=False)

    print(f"Total Combined Patients: {len(X)} (Males: 318, Females: 161) | Deaths: {int(y_df['event'].sum())}")
    print(f"Total Candidate Features: {X.shape[1]}")

    rank_lists = []
    all_unique_items = set()

    for repeat_idx in range(n_repeats):
        print(f"\n---> Running Discovery Repeat {repeat_idx + 1}/{n_repeats}...")
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=repeat_idx)

        for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X, y_rec)):
            X_train, y_train = X.iloc[train_idx], y_rec[train_idx]
            
            # 2. Univariate Cox pre-filtering (p < 0.05) on training set fold only
            train_df = pd.DataFrame(y_train)
            train_df = pd.concat([train_df[["duration", "event"]].reset_index(drop=True), X_train.reset_index(drop=True)], axis=1)
            
            uni_res = univariate_cox(train_df)
            sig_df = uni_res[uni_res["p_value"] < 0.05].sort_values("p_value")
            
            # Retain top 200 univariate significant genes (exact canonical 200-gene cap per fold)
            top_200_genes = sig_df["variable"].head(200).tolist()
            if len(top_200_genes) == 0:
                print(f"  Fold {fold_idx}: No significant univariate genes found. Skipping fold.")
                continue

            X_train_f = X_train[top_200_genes]
            inner_cv = KFold(n_splits=5, shuffle=True, random_state=repeat_idx)

            # 3. Execute exact 4 canonical ensemble feature selection models
            c_genes = coxnet(X_train_f, y_train, inner_cv)
            rf_g = rf_survival(X_train_f, y_train, inner_cv)
            boost_g = survival_booster(X_train_f, y_train, inner_cv)
            svm_g = survival_svm(X_train_f, y_train, inner_cv)

            for m_name, g_list in [("coxnet", c_genes), ("rf", rf_g), ("boost", boost_g), ("svm", svm_g)]:
                if len(g_list) > 0:
                    rank_lists.append(g_list)
                    all_unique_items.update(g_list)

    print(f"\nAggregating {len(rank_lists)} ranking voter lists across folds via Robust Rank Aggregation (RRA)...")
    rra_results = robust_rank_aggregation(rank_lists, list(all_unique_items))
    rra_results["ConsensusRank"] = range(1, len(rra_results) + 1)

    out_dir = Path("./aggregated_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "sex_agnostic_aggregated_ranking.csv"
    rra_results.to_csv(out_file, index=False)
    
    print(f"Successfully saved sex-agnostic rankings to {out_file}")
    print("\nTop 15 Sex-Agnostic Genes:")
    print(rra_results.head(15).to_string(index=False))

if __name__ == "__main__":
    run_sex_agnostic_discovery(n_repeats=20)
