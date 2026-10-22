import warnings
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, ComponentwiseGradientBoostingSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning

import cptac_data_processing as cdp

warnings.filterwarnings("ignore")


def cindex_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    return concordance_index_censored(y["event"], y["duration"], y_pred)[0]


def train_coxnet(X, y):
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


def train_rf(X, y):
    folds = KFold(n_splits=5, shuffle=True, random_state=3)
    gcv = GridSearchCV(
        RandomSurvivalForest(random_state=3),
        param_grid={"n_estimators": [50, 100], "max_depth": [3, 4, 5], "min_samples_split": [5, 10]},
        cv=folds, n_jobs=-1, scoring=cindex_scorer
    )
    gcv.fit(X, y)
    return gcv.best_estimator_


def train_svm(X, y):
    folds = KFold(n_splits=5, shuffle=True, random_state=3)
    gcv = GridSearchCV(
        make_pipeline(StandardScaler(), FastSurvivalSVM(max_iter=1000, random_state=3)),
        param_grid={"fastsurvivalsvm__alpha": [2**i for i in range(-5, 5)]},
        cv=folds, n_jobs=-1, scoring=cindex_scorer
    )
    gcv.fit(X, y)
    return gcv.best_estimator_


def train_boost(X, y):
    folds = KFold(n_splits=5, shuffle=True, random_state=3)
    gcv = GridSearchCV(
        ComponentwiseGradientBoostingSurvivalAnalysis(random_state=3),
        param_grid={"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]},
        cv=folds, n_jobs=-1, scoring=cindex_scorer
    )
    gcv.fit(X, y)
    return gcv.best_estimator_


def main():
    print("=== Training & Freezing CPTAC-Derived Prognostic Models ===")

    # 1. Load CPTAC signature genes from RRA aggregation
    rra_path = Path("./aggregated_results/cptac_sex_agnostic_expr_aggregated_ranking.csv")
    raw_rra_path = Path("./aggregated_results/cptac_sex_agnostic_expr_raw_rra.csv")

    sig_genes = []
    if rra_path.exists():
        rra_df = pd.read_csv(rra_path)
        sig_genes = rra_df["Name"].tolist()

    if len(sig_genes) < 5 and raw_rra_path.exists():
        raw_df = pd.read_csv(raw_rra_path)
        sig_genes = raw_df.sort_values("ConsensusRank")["Name"].head(15).tolist()
        print(f"Loaded top {len(sig_genes)} consensus-ranked CPTAC signature genes from {raw_rra_path}")
    else:
        print(f"Loaded {len(sig_genes)} consensus signature genes from {rra_path}")

    # 2. Load CPTAC dataset
    clinical = cdp.get_cptac_survival_data()
    tpm_data = cdp.get_cptac_expr("gene_name", "tpm")
    omic_w_clinical = clinical.join(tpm_data, how="inner").groupby(level=0).first()

    data = omic_w_clinical.drop(["Gender"], axis=1, errors="ignore")

    avail_genes = [g for g in sig_genes if g in data.columns]
    X_train = data[avail_genes]

    y_df = data[["event", "duration"]].copy()
    y_df["duration"] = y_df["duration"] / 30.437  # Convert days to months
    y_df["event"] = y_df["event"].astype(bool)
    y_train_rec = y_df.to_records(index=False)

    print(f"Fitting CPTAC training set: {len(X_train)} samples, {y_df['event'].sum()} deaths, {len(avail_genes)} features: {avail_genes}")

    # 3. Fit CPTAC Scaler (strictly on CPTAC data)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=avail_genes)

    # 4. Fit canonical models
    models = {
        "Coxnet": train_coxnet(X_train_scaled_df, y_train_rec),
        "Random Forest": train_rf(X_train_scaled_df, y_train_rec),
        "Survival SVM": train_svm(X_train_scaled_df, y_train_rec),
        "Gradient Boosting": train_boost(X_train_scaled_df, y_train_rec)
    }

    # 5. Freeze and save trained pipeline bundle
    bundle = {
        "genes": avail_genes,
        "scaler": scaler,
        "models": models,
        "cptac_train_shape": X_train.shape,
        "cptac_events": int(y_df["event"].sum())
    }

    out_file = Path("./aggregated_results/cptac_frozen_models.joblib")
    joblib.dump(bundle, out_file)
    print(f"Successfully frozen CPTAC models and scaler parameters to {out_file}")


if __name__ == "__main__":
    main()
