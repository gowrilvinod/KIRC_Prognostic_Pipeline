import pandas as pd 
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import make_pipeline 
from sklearn.inspection import permutation_importance
from sksurv.metrics import concordance_index_censored
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning
import warnings

# scoring function for grid searches using C index 
# evaluates whether risk scores are useful 
def cindex_scorer(model, X, y):
        prediction = model.predict(X)
        result = concordance_index_censored(
            y["event"], 
            y["duration"], 
            prediction
        )
        return result[0]


def coxnet(omic_df, survival_array, folds):
    
    best_score = -np.inf
    best_result = None

    # suppress warnings globally within this function
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=FitFailedWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        # defines how close model is to elastic net or LASSO 
        # L1 = 1.0 = LASSO 
        # L1 = 0.0 = L2 penalty 
        # All values in between are a mix of L1 and L2 
        for l1_ratio in [0.1, 0.3, 0.5, 1.0]:
            
            try:
                # first fit to get alphas
                base_pipe = make_pipeline(
                    StandardScaler(),
                    CoxnetSurvivalAnalysis(
                        l1_ratio=l1_ratio,
                        alpha_min_ratio=0.01,
                        max_iter=300000,
                        tol=1e-6
                    )
                )
                base_pipe.fit(omic_df, survival_array)
                estimated_alphas = base_pipe.named_steps["coxnetsurvivalanalysis"].alphas_
                
                # INCREASED minimum alpha to avoid numerical issues
                estimated_alphas = estimated_alphas[estimated_alphas > 0.001]  

                if len(estimated_alphas) == 0:
                    continue  # skip if no alphas survived threshold

                # GridSearchCV for hyperparameter optimization of the alpha space we just got from the initial fit 
                gcv = GridSearchCV(
                    make_pipeline(
                        StandardScaler(),
                        CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, max_iter=300000)
                    ),
                    param_grid={"coxnetsurvivalanalysis__alphas": [[a] for a in estimated_alphas]},
                    cv=folds,
                    n_jobs=18,
                    error_score=np.nan,  # Return NaN instead of crashing
                    scoring=cindex_scorer
                )
                gcv.fit(omic_df, survival_array)

                # keep best scoring model (check for valid score)
                if not np.isnan(gcv.best_score_) and gcv.best_score_ > best_score:
                    best_score = gcv.best_score_
                    best_result = gcv
                    
            except (ArithmeticError, ValueError) as e:
                # Numerical instability for this l1_ratio, try next one
                continue

    # catch case that all L1 ratio values failed 
    if best_result is None:
        print("  WARNING: All CoxNet configurations failed, returning empty list")
        return []  # no model succeeded

    # get coefficients from the best model 
    best_model = best_result.best_estimator_.named_steps["coxnetsurvivalanalysis"]
    coefs = best_model.coef_.ravel()
    coef_series = pd.Series(coefs, index=omic_df.columns)

    # keep only non-zero coefficients, sorted by magnitude
    coef_series = coef_series[coef_series != 0].sort_values(key=lambda x: x.abs(), ascending=False)

    # extract the top 100 
    return coef_series.index.to_list()[:100]



def rf_survival(omic_df, survival_array, folds):
    
    # instantiate random forest
    rsf = RandomSurvivalForest(
        n_estimators=500,
        n_jobs=18,
        random_state=folds.random_state,
        bootstrap=True,
        oob_score=True,
    )

    # set up parameter search space 
    param_grid = {
        "min_samples_split": [5, 10, 20],
        "min_samples_leaf": [5, 10, 20],
        "max_features": ["sqrt", 0.3, 0.5],
    }

    # define exhaustive grid search with c-index as the scoring metric 
    gcv = GridSearchCV(
        rsf,
        param_grid,
        cv=folds,
        n_jobs=1,
        scoring=cindex_scorer
    )

    # fit grid search and get the best estimator 
    gcv.fit(omic_df, survival_array)
    best_rsf = gcv.best_estimator_
    
    # execute permutation importance analysis with the best random forest model 
    # this is necessary because scikit-survival's random forest does not have a feature importance attribute 
    perm = permutation_importance(
        best_rsf,
        omic_df,  
        survival_array,
        n_repeats=10,
        random_state=folds.random_state,
        n_jobs=1,
        scoring=cindex_scorer  
    )

    # get the feature importances 
    importances = pd.Series(perm.importances_mean, index=omic_df.columns)
    importances = importances.abs()

    # extract the top 100 
    return (
        importances
        .sort_values(ascending=False)
        .head(100)
        .index
        .tolist()
    )





def survival_booster(omic_df, survival_array, folds): 

    # create component wise gradient boosting survival analysis 
    est = ComponentwiseGradientBoostingSurvivalAnalysis(
        loss="coxph",
        random_state=folds.random_state
    )

    # define parameter grid 
    param_grid = {
        "learning_rate": [0.05, 0.1],
        "n_estimators": [100, 200, 400],
        "subsample": [0.5, 1.0],
        "dropout_rate": [0.0, 0.1],
    }

    # define grid search with c-index as the scoring metric 
    gcv = GridSearchCV(
        est,
        param_grid=param_grid,
        cv=folds,
        n_jobs=18, 
        scoring = cindex_scorer
    )

    # fit grid search and extract best model 
    gcv.fit(omic_df, survival_array)
    best_model = gcv.best_estimator_

    # get coefficients 
    coefs = best_model.coef_
    
    # If coefs has one extra element, skip the first (intercept/baseline)
    if len(coefs) == len(omic_df.columns) + 1:
        coef_series = pd.Series(coefs[1:], index=omic_df.columns)
    else:
        coef_series = pd.Series(coefs, index=omic_df.columns)

    # exclude nonzero coefficients and sort by absolute magnitude 
    coef_series = (
        coef_series[coef_series != 0]
        .abs()
        .sort_values(ascending=False)
    )

    # get top 100 genes 
    return coef_series.head(100).index.tolist()


def survival_svm(omic_df, survival_array, folds): 

    # Use ranking objective (rank_ratio=1.0)
    ssvm = make_pipeline(
        StandardScaler(), 
        FastSurvivalSVM(
            rank_ratio=1.0,
            max_iter=1000,
            tol=1e-5,
            random_state=folds.random_state
        )
    )
    
    # parameter grid for alpha values 
    param_grid = {
        "fastsurvivalsvm__alpha": [2.0**v for v in range(-8, 9, 2)],
    }

    # define grid search with c-index scoring 
    gcv = GridSearchCV(
        ssvm,
        param_grid,
        cv=folds,
        n_jobs=18, 
        scoring=cindex_scorer
    )
    
    # execute grid search and get best pipeline 
    gcv.fit(omic_df, survival_array)
    best_svm = gcv.best_estimator_
    
    # Get feature coefficients (weights)
    coefs = best_svm.named_steps["fastsurvivalsvm"].coef_
    coef_series = pd.Series(coefs, index=omic_df.columns)
    
    # Sort by absolute value of coefficients
    coef_series = coef_series.abs().sort_values(ascending=False)
    
    # top 100 
    return coef_series.head(100).index.tolist()



    
