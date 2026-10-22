import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, ComponentwiseGradientBoostingSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

import cptac_data_processing as cdp

warnings.filterwarnings("ignore")


def cindex_scorer(estimator, X, y):
    if isinstance(y, pd.DataFrame):
        n_events = y["event"].sum()
    else:
        n_events = y["event"].astype(bool).sum()

    if n_events == 0:
        return 0.5

    try:
        y_pred = estimator.predict(X)
        return concordance_index_censored(y["event"], y["duration"], y_pred)[0]
    except Exception:
        return 0.5


def compute_time_aucs(y_train_rec, y_val_df, y_pred, time_points=[12.0, 36.0, 60.0]):
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


def train_single_fold_model(model_type, X_tr, y_tr):
    try:
        if model_type == "Coxnet":
            best_score = -np.inf
            best_m = None
            for l1_ratio in [0.1, 0.3, 0.5, 1.0]:
                try:
                    pipe = make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alpha_min_ratio=0.01, max_iter=300000, tol=1e-6))
                    pipe.fit(X_tr, y_tr)
                    alphas = pipe.named_steps["coxnetsurvivalanalysis"].alphas_
                    alphas = alphas[alphas > 0.001]
                    if len(alphas) == 0: continue
                    cv = KFold(n_splits=5, shuffle=True, random_state=3)
                    gcv = GridSearchCV(make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, max_iter=300000)),
                                       param_grid={"coxnetsurvivalanalysis__alphas": [[a] for a in alphas]},
                                       cv=cv, n_jobs=-1, error_score=np.nan, scoring=cindex_scorer)
                    gcv.fit(X_tr, y_tr)
                    if gcv.best_score_ > best_score:
                        best_score = gcv.best_score_
                        best_m = gcv.best_estimator_
                except Exception: continue
            return best_m
        elif model_type == "Random Forest":
            cv = KFold(n_splits=5, shuffle=True, random_state=3)
            gcv = GridSearchCV(RandomSurvivalForest(random_state=3),
                               param_grid={"n_estimators": [50, 100], "max_depth": [3, 4, 5], "min_samples_split": [5, 10]},
                               cv=cv, n_jobs=-1, error_score=np.nan, scoring=cindex_scorer)
            gcv.fit(X_tr, y_tr)
            return gcv.best_estimator_
        elif model_type == "Survival SVM":
            cv = KFold(n_splits=5, shuffle=True, random_state=3)
            gcv = GridSearchCV(make_pipeline(StandardScaler(), FastSurvivalSVM(max_iter=1000, random_state=3)),
                               param_grid={"fastsurvivalsvm__alpha": [2**i for i in range(-5, 5)]},
                               cv=cv, n_jobs=-1, error_score=np.nan, scoring=cindex_scorer)
            gcv.fit(X_tr, y_tr)
            return gcv.best_estimator_
        elif model_type == "Gradient Boosting":
            cv = KFold(n_splits=5, shuffle=True, random_state=3)
            gcv = GridSearchCV(ComponentwiseGradientBoostingSurvivalAnalysis(random_state=3),
                               param_grid={"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]},
                               cv=cv, n_jobs=-1, error_score=np.nan, scoring=cindex_scorer)
            gcv.fit(X_tr, y_tr)
            return gcv.best_estimator_
    except Exception:
        return None


def run_oof_validation_for_sex(sex_label, sex_code):
    print(f"\n=======================================================================")
    print(f" CPTAC {sex_label.upper()} INTERNAL 5-FOLD OUT-OF-FOLD (OOF) VALIDATION")
    print(f"=======================================================================")

    rra_path = Path(f"./aggregated_results/cptac_sex_specific_{sex_label.lower()}_expr_aggregated_ranking.csv")
    if not rra_path.exists():
        print(f"Aggregated signature file not found: {rra_path}")
        return []

    sig_genes = pd.read_csv(rra_path)["Name"].tolist()
    print(f"Evaluating {len(sig_genes)} CPTAC {sex_label} consensus genes: {sig_genes}")

    clinical = cdp.get_cptac_survival_data()
    tpm_data = cdp.get_cptac_expr("gene_name", "tpm")
    omic_w_clinical = clinical.join(tpm_data, how="inner").groupby(level=0).first()

    sex_df = omic_w_clinical[omic_w_clinical["Gender"] == sex_code].copy()
    data = sex_df.drop(["Gender"], axis=1, errors="ignore")

    avail_genes = [g for g in sig_genes if g in data.columns]
    X = data[avail_genes]
    y_df = data[["event", "duration"]].copy()
    y_df["duration"] = y_df["duration"] / 30.437  # Convert days to months
    y_df["event"] = y_df["event"].astype(bool)
    y_rec = y_df.to_records(index=False)

    n_pts = len(X)
    n_deaths = int(y_df["event"].sum())

    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    model_types = ["Coxnet", "Random Forest", "Survival SVM", "Gradient Boosting"]

    results = []

    for m_label in model_types:
        oof_preds = np.zeros(len(X))
        oof_mask = np.zeros(len(X), dtype=bool)
        failed_folds = 0

        for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X, y_rec)):
            X_tr, y_tr = X.iloc[train_idx], y_rec[train_idx]
            X_va = X.iloc[val_idx]

            scaler = StandardScaler()
            X_tr_scaled = pd.DataFrame(scaler.fit_transform(X_tr), columns=avail_genes)
            X_va_scaled = pd.DataFrame(scaler.transform(X_va), columns=avail_genes)

            model = train_single_fold_model(m_label, X_tr_scaled, y_tr)
            if model is not None:
                try:
                    preds = model.predict(X_va_scaled)
                    oof_preds[val_idx] = preds
                    oof_mask[val_idx] = True
                except Exception:
                    failed_folds += 1
            else:
                failed_folds += 1

        if oof_mask.sum() == 0:
            print(f"[{sex_label.upper()}] Model {m_label} failed on all folds.")
            continue

        valid_y_df = y_df.iloc[oof_mask]
        valid_preds = oof_preds[oof_mask]

        if valid_y_df["event"].sum() == 0:
            print(f"[{sex_label.upper()}] Model {m_label}: All OOF validation samples are censored (0 deaths). Skipping.")
            continue

        c_index = concordance_index_censored(valid_y_df["event"], valid_y_df["duration"], valid_preds)[0]
        auc_dict = compute_time_aucs(y_rec, valid_y_df, valid_preds)
        km_stats = compute_km_logrank_hr(valid_y_df, valid_preds)

        results.append({
            "Cohort": f"CPTAC-{sex_label.capitalize()} (Internal OOF)",
            "Model": m_label,
            "CV Procedure": "5-Fold OOF CV",
            "N Patients": n_pts,
            "N Deaths": n_deaths,
            "C-Index": f"{c_index:.4f}",
            "1-Yr AUC": auc_dict.get("1Yr", "N/A"),
            "3-Yr AUC": auc_dict.get("3Yr", "N/A"),
            "5-Yr AUC": auc_dict.get("5Yr", "N/A"),
            "Log-Rank p": km_stats["logrank_p"],
            "HR (95% CI)": f"{km_stats['HR']} {km_stats['HR_CI']}",
            "Failed Folds": f"{failed_folds}/5"
        })

    return results


def main():
    print("=== CPTAC Sex-Specific Internal Validation (5-Fold OOF CV) ===")
    res_male = run_oof_validation_for_sex("male", 1)
    res_female = run_oof_validation_for_sex("female", 0)

    all_res = res_male + res_female
    if all_res:
        df_res = pd.DataFrame(all_res)
        out_csv = Path("./aggregated_results/cptac_internal_validation_results_sex_specific.csv")
        df_res.to_csv(out_csv, index=False)
        print("\n=======================================================================")
        print(" INTERNAL VALIDATION RESULTS (CPTAC SEX-SPECIFIC 5-FOLD OOF PREDICTIONS)")
        print("=======================================================================")
        print(df_res.to_string(index=False))
        print(f"\nSaved sex-specific internal validation metrics to {out_csv}")


if __name__ == "__main__":
    main()
