import os
import glob
import warnings
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from lifelines import CoxPHFitter
import concurrent.futures

# Import canonical modules
import data_processing as dp
from ensemble import coxnet, rf_survival, survival_booster, survival_svm

warnings.filterwarnings("ignore")

# Parse CLI arguments matching execution.py
parser = argparse.ArgumentParser(description="Sex-Agnostic SLURM Array Discovery Task")
parser.add_argument("--rando", type=int, required=True, help="Random seed / SLURM array task ID (0-19)")
parser.add_argument("--data_type", type=str, default="expr", help="Data type (default: expr)")
args = parser.parse_args()

rando = args.rando
data_type = args.data_type

def fit_single_gene_cox(args_tuple):
    gene, duration, event, gene_vals = args_tuple
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
    Parallelized univariate Cox proportional hazards regression for candidate genes on training fold data.
    """
    duration = train_df["duration"].values
    event = train_df["event"].values.astype(bool)
    feature_cols = [c for c in train_df.columns if c not in ["duration", "event"]]
    
    tasks = [(g, duration, event, train_df[g].values) for g in feature_cols]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(fit_single_gene_cox, tasks))
        
    valid_results = [r for r in results if r is not None]
    return pd.DataFrame(valid_results)

def run_sex_agnostic_fold_discovery(rando):
    print(f"=======================================================================")
    print(f" TCGA-KIRC SEX-AGNOSTIC DISCOVERY (SLURM Array Task: rando={rando})")
    print(f"=======================================================================")

    # 1. Load clean KIRC clinical survival data (event, duration, Gender) and TPM expression data
    clinical = dp.get_survival_data()
    tpm_data = dp.get_tcga_expr("gene_name", "tpm")
    omic_w_clinical = clinical.join(tpm_data, how="inner")
    omic_w_clinical = omic_w_clinical.groupby(omic_w_clinical.index).first()

    # Combine ALL 479 patients (318 males + 161 females) without sex features or sex splitting
    data = omic_w_clinical.drop(["Gender"], axis=1, errors="ignore")
    X = data.drop(["duration", "event"], axis=1)
    y_df = data[["event", "duration"]].copy()
    y_df["duration"] = y_df["duration"] / 30.437  # Convert days to months
    y_df["event"] = y_df["event"].astype(bool)
    y_rec = y_df.to_records(index=False)

    out_dir = Path("./aggregated_results/folded_ranks")
    out_dir.mkdir(parents=True, exist_ok=True)

    outer_cv = KFold(n_splits=5, shuffle=True, random_state=rando)

    for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X, y_rec)):
        out_file = out_dir / f"sex_agnostic_fold_{fold_idx}_rando{rando}.csv"
        if out_file.exists() and out_file.stat().st_size > 0:
            print(f"Fold {fold_idx} (rando={rando}) already completed, skipping.")
            continue

        X_train, y_train = X.iloc[train_idx], y_rec[train_idx]
        
        # 2. Univariate Cox pre-filtering (p < 0.05) on training fold only
        train_df = pd.DataFrame(y_train)
        train_df = pd.concat([train_df[["duration", "event"]].reset_index(drop=True), X_train.reset_index(drop=True)], axis=1)
        
        uni_res = univariate_cox(train_df)
        sig_df = uni_res[uni_res["p_value"] < 0.05].sort_values("p_value")
        top_200_genes = sig_df["variable"].head(200).tolist()
        if len(top_200_genes) == 0:
            print(f"Fold {fold_idx}: No univariate Cox significant genes found. Skipping.")
            continue

        X_train_f = X_train[top_200_genes]
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=rando)

        # 3. Fit 4 canonical ML ensemble feature selection models
        c_genes = coxnet(X_train_f, y_train, inner_cv)
        rf_g = rf_survival(X_train_f, y_train, inner_cv)
        boost_g = survival_booster(X_train_f, y_train, inner_cv)
        svm_g = survival_svm(X_train_f, y_train, inner_cv)

        rra_rows = []
        qid = "Q1"

        for model_name, ranked_genes in [("coxnet", c_genes), ("rf", rf_g), ("boost", boost_g), ("svm", svm_g)]:
            voter = f"{model_name}_fold{fold_idx}_iter{rando}"
            n_genes = len(ranked_genes)
            if n_genes == 0:
                continue
            for rank, gene in enumerate(ranked_genes, start=1):
                score = 1.0 - ((rank - 1) / n_genes)
                rra_rows.append((qid, voter, gene, rank, score, "ensemble"))

        rra_df = pd.DataFrame(rra_rows, columns=["Query", "Voter", "ItemID", "Rank", "Score", "Algorithm"])
        rra_df.to_csv(out_file, index=False)
        print(f"Saved fold {fold_idx} rankings (rando={rando}) to {out_file}")

if __name__ == "__main__":
    run_sex_agnostic_fold_discovery(rando)
