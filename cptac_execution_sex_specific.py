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

# Import canonical modules and CPTAC-specific module
import data_processing as dp
import cptac_data_processing as cdp
from ensemble import coxnet, rf_survival, survival_booster, survival_svm

warnings.filterwarnings("ignore")

# Parse CLI arguments matching execution_sex_specific setup
parser = argparse.ArgumentParser(description="CPTAC Sex-Specific Discovery Task")
parser.add_argument("--rando", type=int, default=3, help="Random seed / array task ID (default: 3)")
parser.add_argument("--data_type", type=str, default="expr", help="Data type (default: expr)")
parser.add_argument("--sex", type=str, default="both", choices=["both", "male", "female"], help="Sex cohort to run: 'male', 'female', or 'both' (default: 'both')")
args = parser.parse_args()

rando = args.rando
data_type = args.data_type
sex_arg = args.sex


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
    Parallelized univariate Cox proportional hazards regression using ProcessPoolExecutor for maximum speed across cores.
    """
    duration = train_df["duration"].values
    event = train_df["event"].values.astype(bool)
    feature_cols = [c for c in train_df.columns if c not in ["duration", "event"]]

    tasks = [(g, duration, event, train_df[g].values) for g in feature_cols]
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(fit_single_gene_cox, tasks, chunksize=50))

    valid_results = [r for r in results if r is not None]
    return pd.DataFrame(valid_results)


def run_cptac_single_sex_discovery(sex_label, sex_code, omic_w_clinical, rando):
    print(f"\n=======================================================================")
    print(f" CPTAC-KIRC SEX-SPECIFIC DISCOVERY: {sex_label.upper()} COHORT (Task: rando={rando})")
    print(f"=======================================================================")

    # Filter cohort by sex (Gender: 1 = Male, 0 = Female)
    sex_df = omic_w_clinical[omic_w_clinical["Gender"] == sex_code].copy()
    data = sex_df.drop(["Gender"], axis=1, errors="ignore")

    X = data.drop(["duration", "event"], axis=1)
    y_df = data[["event", "duration"]].copy()
    y_df["duration"] = y_df["duration"] / 30.437  # Convert days to months
    y_df["event"] = y_df["event"].astype(bool)
    y_rec = y_df.to_records(index=False)

    n_samples = len(X)
    n_deaths = int(y_df["event"].sum())
    print(f"Cohort Stats ({sex_label}): {n_samples} patients, {n_deaths} deaths ({n_deaths/n_samples*100:.1f}% event rate)")

    out_dir = Path(f"./aggregated_results/cptac_sex_specific_{sex_label.lower()}_folded_ranks")
    out_dir.mkdir(parents=True, exist_ok=True)

    outer_cv = KFold(n_splits=5, shuffle=True, random_state=rando)

    for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X, y_rec)):
        out_file = out_dir / f"cptac_sex_specific_{sex_label.lower()}_fold_{fold_idx}_rando{rando}.csv"
        if out_file.exists() and out_file.stat().st_size > 0:
            print(f"[{sex_label.upper()}] Fold {fold_idx} (rando={rando}) already completed, skipping.")
            continue

        print(f"[{sex_label.upper()}] Processing Fold {fold_idx}/5...")
        X_train, y_train = X.iloc[train_idx], y_rec[train_idx]
        n_train_deaths = int(y_train["event"].sum())
        print(f"[{sex_label.upper()}] Fold {fold_idx} training set: {len(X_train)} samples, {n_train_deaths} deaths")

        # 1. Univariate Cox pre-filtering (p < 0.05) on training fold only
        train_df = pd.DataFrame(y_train)
        train_df = pd.concat([train_df[["duration", "event"]].reset_index(drop=True), X_train.reset_index(drop=True)], axis=1)

        uni_res = univariate_cox(train_df)
        if uni_res.empty:
            print(f"[{sex_label.upper()}] WARNING: Fold {fold_idx}: Univariate Cox failed to converge for any gene. Skipping fold.")
            continue

        sig_df = uni_res[uni_res["p_value"] < 0.05].sort_values("p_value")
        top_200_genes = sig_df["variable"].head(200).tolist()
        if len(top_200_genes) == 0:
            print(f"[{sex_label.upper()}] WARNING: Fold {fold_idx}: No univariate Cox significant genes found (p < 0.05). Skipping fold.")
            continue

        print(f"[{sex_label.upper()}] Fold {fold_idx}: Selected top {len(top_200_genes)} univariate Cox genes. Fitting ensemble models...")
        X_train_f = X_train[top_200_genes]
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=rando)

        # 2. Fit 4 canonical ML ensemble feature selection models (log warnings explicitly)
        print(f"[{sex_label.upper()}] Fold {fold_idx}: Fitting CoxNet...")
        c_genes = coxnet(X_train_f, y_train, inner_cv)
        if len(c_genes) == 0:
            print(f"[{sex_label.upper()}] WARNING: Fold {fold_idx}: CoxNet configurations failed or returned 0 features.")

        print(f"[{sex_label.upper()}] Fold {fold_idx}: Fitting Random Survival Forest...")
        rf_g = rf_survival(X_train_f, y_train, inner_cv)
        if len(rf_g) == 0:
            print(f"[{sex_label.upper()}] WARNING: Fold {fold_idx}: Random Survival Forest returned 0 features.")

        print(f"[{sex_label.upper()}] Fold {fold_idx}: Fitting Gradient Boosting...")
        boost_g = survival_booster(X_train_f, y_train, inner_cv)
        if len(boost_g) == 0:
            print(f"[{sex_label.upper()}] WARNING: Fold {fold_idx}: Gradient Boosting returned 0 features.")

        print(f"[{sex_label.upper()}] Fold {fold_idx}: Fitting Survival SVM...")
        svm_g = survival_svm(X_train_f, y_train, inner_cv)
        if len(svm_g) == 0:
            print(f"[{sex_label.upper()}] WARNING: Fold {fold_idx}: Survival SVM returned 0 features.")

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
        print(f"[{sex_label.upper()}] Saved fold {fold_idx} rankings (rando={rando}) to {out_file} ({len(rra_df)} entries from {len(set(rra_df['Voter'])) if not rra_df.empty else 0} active voters)")


def run_cptac_sex_specific_fold_discovery(rando, sex_mode="both"):
    print(f"=======================================================================")
    print(f" CPTAC-KIRC SEX-SPECIFIC DISCOVERY PIPELINE (Task: rando={rando}, Mode: {sex_mode})")
    print(f"=======================================================================")

    # Load clean CPTAC clinical survival data (event, duration, Gender) and TPM expression data
    clinical = cdp.get_cptac_survival_data()
    tpm_data = cdp.get_cptac_expr("gene_name", "tpm")
    omic_w_clinical = clinical.join(tpm_data, how="inner")
    omic_w_clinical = omic_w_clinical.groupby(omic_w_clinical.index).first()

    # Run male cohort discovery if requested
    if sex_mode in ["both", "male"]:
        run_cptac_single_sex_discovery("male", 1, omic_w_clinical, rando)

    # Run female cohort discovery if requested
    if sex_mode in ["both", "female"]:
        run_cptac_single_sex_discovery("female", 0, omic_w_clinical, rando)


if __name__ == "__main__":
    run_cptac_sex_specific_fold_discovery(rando, sex_arg)
