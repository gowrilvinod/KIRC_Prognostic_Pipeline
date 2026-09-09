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

# Import KIRC data processing module
import data_processing as dp

warnings.filterwarnings("ignore")

# Define frozen sex-specific 8-gene signatures
male_genes = ["HS3ST1", "RNF183", "LINC00973", "HBB", "CDKN1A", "HMGA2", "HS3ST3A1", "ADAM8"]
female_genes = ["SSTR1", "ACP5", "TNFSF15", "NUDT2", "ACTN2", "CTTNBP2", "SNTG2-AS1", "RNY4P34"]

def cindex_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    return concordance_index_censored(y["event"], y["duration"], y_pred)[0]

def get_tcga_cohort(encoded_gender, genes):
    clinical = dp.get_survival_data()
    tpm_data = dp.get_tcga_expr("gene_name", "tpm")
    omic_w_clinical = clinical.join(tpm_data, how="inner")
    omic_w_clinical = omic_w_clinical.groupby(omic_w_clinical.index).first()
    data = omic_w_clinical[omic_w_clinical["Gender"] == encoded_gender].drop("Gender", axis=1)
    
    avail_genes = [g for g in genes if g in data.columns]
    X = data[avail_genes].copy()
    y = data[["event", "duration"]].copy()
    y["duration"] = y["duration"] / 30.437  # Convert days to months
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

def evaluate_tcga_internal_cv(encoded_gender, genes):
    gender_str = "Male" if encoded_gender == 1 else "Female"
    X, y = get_tcga_cohort(encoded_gender, genes)
    
    n_patients = len(X)
    n_deaths = int(y["event"].sum())
    
    print(f"\n=======================================================")
    print(f" TCGA-KIRC 5-FOLD OUT-OF-FOLD INTERNAL VALIDATION: {gender_str.upper()} COHORT")
    print(f" Patients: {n_patients} | Deaths: {n_deaths} | Frozen Genes: {len(genes)}")
    print(f"=======================================================")
    
    models = {
        "Coxnet (Primary)": coxnet_predict,
        "Gradient Boosting": survival_boost,
        "Survival SVM": svm_survival,
        "Random Forest": rf_survival,
    }
    
    outer_folds = KFold(n_splits=5, shuffle=True, random_state=3)
    summary_rows = []
    
    for m_label, train_func in models.items():
        oof_preds = np.zeros(n_patients)
        
        for fold_idx, (train_idx, val_idx) in enumerate(outer_folds.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=genes)
            X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=genes)
            
            trained_model = train_func(X_train_scaled, y_train)
            if trained_model is None:
                raise RuntimeError(f"[{m_label}] Training failed on fold {fold_idx}")
            oof_preds[val_idx] = trained_model.predict(X_val_scaled)
            
        c_index = concordance_index_censored(y["event"], y["duration"], oof_preds)[0]
        auc_values, mean_auc = cumulative_dynamic_auc(y, y, oof_preds, np.array([12.0, 36.0, 60.0]))
        
        summary_rows.append({
            "Model": m_label,
            "C-Index": f"{c_index:.4f}",
            "1-Yr AUC": f"{auc_values[0]:.4f}",
            "3-Yr AUC": f"{auc_values[1]:.4f}",
            "5-Yr AUC": f"{auc_values[2]:.4f}",
            "Mean AUC": f"{mean_auc:.4f}"
        })
        
    df_summary = pd.DataFrame(summary_rows)
    print("\n" + df_summary.to_string(index=False))

if __name__ == "__main__":
    evaluate_tcga_internal_cv(1, male_genes)
    evaluate_tcga_internal_cv(0, female_genes)
