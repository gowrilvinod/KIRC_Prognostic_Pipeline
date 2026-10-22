import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, ComponentwiseGradientBoostingSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test

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
                if not np.isnan(gcv.best_score_) and gcv.best_score_ > best_score:
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


def compute_time_aucs(y_train_rec, y_val_df, y_pred, time_points=[12.0, 36.0, 60.0]):
    """
    Computes time-dependent cumulative dynamic AUCs for 1-Yr (12m), 3-Yr (36m), 5-Yr (60m).
    """
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
        va_auc, _ = cumulative_dynamic_auc(y_train_rec, y_val_rec, y_pred, np.array(valid_times))
        return {label: f"{auc_val:.3f}" for label, auc_val in zip(labels, va_auc)}
    except Exception:
        return {}


def compute_km_logrank_hr(y_val_df, y_pred):
    """
    Computes median risk-score stratified Kaplan-Meier log-rank p-value and Hazard Ratio (HR with 95% CI).
    """
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


def main():
    print("=======================================================================")
    print(" CPTAC-KIRC INTERNAL OUT-OF-FOLD (OOF) VALIDATION")
    print("=======================================================================")

    # 1. Load finalized 9-gene signature from RRA ranking
    sig_path = Path("./aggregated_results/cptac_sex_agnostic_expr_aggregated_ranking.csv")
    if not sig_path.exists():
        raise FileNotFoundError(f"CPTAC signature file not found at {sig_path}")

    sig_df = pd.read_csv(sig_path)
    sig_genes = sig_df["Name"].tolist()
    print(f"Loaded {len(sig_genes)} finalized CPTAC signature genes: {sig_genes}")

    # 2. Load CPTAC dataset
    clinical = cdp.get_cptac_survival_data()
    tpm_data = cdp.get_cptac_expr("gene_name", "tpm")
    omic_w_clinical = clinical.join(tpm_data, how="inner").groupby(level=0).first()

    data = omic_w_clinical.drop(["Gender"], axis=1, errors="ignore")
    avail_genes = [g for g in sig_genes if g in data.columns]
    
    X = data[avail_genes]
    y_df = data[["event", "duration"]].copy()
    y_df["duration"] = y_df["duration"] / 30.437  # Convert days to months
    y_df["event"] = y_df["event"].astype(bool)
    y_rec = y_df.to_records(index=False)

    print(f"CPTAC Cohort: n={len(X)} patients, {y_df['event'].sum()} deaths, {len(avail_genes)} signature features.")
    print("Procedure: 5-Fold Cross-Validation Out-Of-Fold (OOF) Prediction (Zero Data Leakage)")

    # 3. Perform 5-fold Out-Of-Fold Cross-Validation
    cv = KFold(n_splits=5, shuffle=True, random_state=3)

    model_trainers = {
        "Coxnet": train_coxnet,
        "Random Forest": train_rf,
        "Survival SVM": train_svm,
        "Gradient Boosting": train_boost
    }

    oof_predictions = {m_name: np.full(len(X), np.nan) for m_name in model_trainers.keys()}
    fold_failures = {m_name: 0 for m_name in model_trainers.keys()}

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y_rec)):
        X_tr, y_tr = X.iloc[train_idx], y_rec[train_idx]
        X_te = X.iloc[test_idx]

        # Fit scaler ONLY on training fold
        scaler = StandardScaler()
        X_tr_scaled = pd.DataFrame(scaler.fit_transform(X_tr), columns=avail_genes, index=X_tr.index)
        X_te_scaled = pd.DataFrame(scaler.transform(X_te), columns=avail_genes, index=X_te.index)

        for m_name, trainer_func in model_trainers.items():
            try:
                fitted_model = trainer_func(X_tr_scaled, y_tr)
                if fitted_model is not None:
                    preds = fitted_model.predict(X_te_scaled)
                    oof_predictions[m_name][test_idx] = preds
                else:
                    fold_failures[m_name] += 1
                    print(f"Fold {fold_idx}: {m_name} optimization failed (returned None).")
            except Exception as e:
                fold_failures[m_name] += 1
                print(f"Fold {fold_idx}: {m_name} threw exception: {e}")

    # 4. Evaluate OOF Predictions for each model
    results = []

    for m_name, preds in oof_predictions.items():
        valid_mask = ~np.isnan(preds)
        n_valid = valid_mask.sum()

        if n_valid < len(X) * 0.5:
            print(f"Warning: {m_name} failed on too many folds ({fold_failures[m_name]}/5 folds failed).")
            results.append({
                "Cohort": "CPTAC-KIRC (Internal OOF)",
                "Model": m_name,
                "CV Procedure": "5-Fold OOF CV",
                "C-Index": "N/A",
                "1-Yr AUC": "N/A",
                "3-Yr AUC": "N/A",
                "5-Yr AUC": "N/A",
                "Log-Rank p": "N/A",
                "HR (95% CI)": "N/A",
                "Failed Folds": f"{fold_failures[m_name]}/5"
            })
            continue

        y_val_sub = y_df.iloc[valid_mask]
        preds_sub = preds[valid_mask]

        c_index = concordance_index_censored(y_val_sub["event"], y_val_sub["duration"], preds_sub)[0]
        auc_dict = compute_time_aucs(y_rec, y_val_sub, preds_sub)
        km_stats = compute_km_logrank_hr(y_val_sub, preds_sub)

        results.append({
            "Cohort": "CPTAC-KIRC (Internal OOF)",
            "Model": m_name,
            "CV Procedure": "5-Fold OOF CV",
            "C-Index": f"{c_index:.4f}",
            "1-Yr AUC": auc_dict.get("1Yr", "N/A"),
            "3-Yr AUC": auc_dict.get("3Yr", "N/A"),
            "5-Yr AUC": auc_dict.get("5Yr", "N/A"),
            "Log-Rank p": km_stats["logrank_p"],
            "HR (95% CI)": f"{km_stats['HR']} {km_stats['HR_CI']}",
            "Failed Folds": f"{fold_failures[m_name]}/5"
        })

    res_df = pd.DataFrame(results)

    print("\n=======================================================================")
    print(" INTERNAL VALIDATION RESULTS (CPTAC 5-FOLD OUT-OF-FOLD PREDICTIONS)")
    print("=======================================================================")
    print(res_df.to_string(index=False))

    out_csv = Path("./aggregated_results/cptac_internal_validation_results.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved internal validation results to {out_csv}")


if __name__ == "__main__":
    main()
