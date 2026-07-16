import argparse 
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



def eval_aggy(data_type, encoded_gender, eval_model="coxnet"):

    # get survival information for TCGA-BLCA (vital status and survival time in months)
    clinical = get_survival_data()

    # depending on the desired data type, load the omics matrix 
    if data_type == "expr": 
        data = get_tcga_expr("gene_name", "tpm")
    elif data_type == "methy": 
        data = get_tcga_methy()
    elif data_type == "mirna": 
        data = get_tcga_mirna()
    elif data_type == "protein":
        data = get_tcga_protein()
    else: 
        raise Exception("Input valid data type!")

    # merge clinical and omics data
    omic_w_clinical = clinical.join(data, how="inner")
    omic_w_clinical = omic_w_clinical.groupby(omic_w_clinical.index).first()

    # split by gender
    data = omic_w_clinical[omic_w_clinical["Gender"] == encoded_gender].drop("Gender", axis=1)

    # drop survival information from omics data, store it in separate variable then convert to record 
    X = data.drop(["duration", "event"], axis=1)
    y = data[["event", "duration"]].copy()
    y["event"] = y["event"].astype(bool)
    y = y.to_records(index=False)

    # get aggregated rankings from csv file 
    if encoded_gender == 1: 
        aggy = pd.read_csv(Path(f"./aggregated_results/male_{data_type}_aggregated_ranking.csv"))
    else: 
        aggy = pd.read_csv(Path(f"./aggregated_results/female_{data_type}_aggregated_ranking.csv"))

    # convert to list
    selected_genes = aggy["Name"].to_list()

    # load external dataset
    X_ex, y_ex = get_external_expr(encoded_gender)

    # get overlapping genes while preserving rank order 
    overlap = [g for g in selected_genes if g in X_ex.columns]

    # store # genes, c-index, and AUCs for best c-index when iterating through all # of genes 
    best_cindex = 0 
    num_genes = 0
    associated_aucs = 0

    # I would try to make a graph of number of genes (i) vs. c-index 
    # store c-index for each value of i and make a line graph 

    # iterate through all possible top N genes 
    for i in range(len(overlap), 0, -1):
    
        subset = overlap[:i]
        # limit training data to genes of interest
        X_selected = X[subset]
        X_ex_sub = X_ex[subset]
        

        # determine optimal hyperparameters and fit best pipeline on training data
        if eval_model == "coxnet": 

            best_pipe = coxnet_predict(X_selected, y)

        elif eval_model == "rf": 

            best_pipe = rf_survival(X_selected, y)
        
        elif eval_model == "svm": 

            best_pipe = svm_survival(X_selected, y)

        elif eval_model == "boost": 

            best_pipe = survival_boost(X_selected, y)

        # get risk score from best pipeline 
        y_pred = best_pipe.predict(X_ex_sub)

        # get performance metrics
        c_index = concordance_index_censored(y_ex["event"], y_ex["duration"], y_pred)
        # AUC at 1, 3, and 5 years 
        auc, mean_auc = cumulative_dynamic_auc(y, y_ex, y_pred, np.array([12, 36, 60]))
        
        # store info if c-index is the best encountered thus far
        if c_index[0] > best_cindex:
            best_cindex = c_index[0]
            num_genes = i
            associated_aucs = auc

    # print info after all # of genes have been tested 
    print(best_cindex)
    print(num_genes)
    print(associated_aucs)




def coxnet_predict(X, y): 

    # score storage for hardcoded parameter optimization - needed for scikit-survival 
    best_score = -np.inf
    best_result = None

    # suppress warnings globally within this function
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=FitFailedWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        # L1 ratio defines whether model leans toward Elastic net or LASSO
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
                base_pipe.fit(X, y)

                # get estimated alphas to define parameter search space 
                estimated_alphas = base_pipe.named_steps["coxnetsurvivalanalysis"].alphas_
                estimated_alphas = estimated_alphas[estimated_alphas > 0.001]  

                if len(estimated_alphas) == 0:
                    continue  # skip if no alphas survived threshold

                # 5-fold CV 
                folds = KFold(n_splits=5, shuffle=True, random_state=3)

                # GridSearchCV for hyperparameter optimization of alpha at a given L1 ratio
                gcv = GridSearchCV(
                    make_pipeline(
                        StandardScaler(),
                        CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, max_iter=300000)
                    ),
                    param_grid={"coxnetsurvivalanalysis__alphas": [[a] for a in estimated_alphas]},
                    cv=folds,
                    n_jobs=-1,
                    error_score=np.nan,  # Return NaN instead of crashing
                    scoring=cindex_scorer
                )

                # fit the grid search
                gcv.fit(X, y)

                # keep best scoring model (check for valid score)
                if not np.isnan(gcv.best_score_) and gcv.best_score_ > best_score:
                    best_score = gcv.best_score_
                    best_result = gcv
                    
            except (ArithmeticError, ValueError) as e:
                # Numerical instability for this l1_ratio, try next one
                continue

    # best_result is the CoxNet pipeline with the best performance on the TCGA set 
    best_pipe = best_result.best_estimator_

    return best_pipe




def rf_survival(X, y): 

    # define 5-fold scheme
    folds = KFold(n_splits=5, shuffle=True, random_state=3)

    # instantiate random forest survival model (with boostrapping and 500 estimators)
    rsf = RandomSurvivalForest(
        n_estimators=500,
        n_jobs=18,
        random_state=folds.random_state,
        bootstrap=True
    )

    # random forest doesn't typically need scaling, but our training data is RNA-seq and external is microarray 
    pipe = make_pipeline(StandardScaler(), rsf)

    # set up parameter grid for searching
    param_grid = {
        "randomsurvivalforest__min_samples_split": [5, 10, 20],
        "randomsurvivalforest__min_samples_leaf": [5, 10, 20],
        "randomsurvivalforest__max_features": ["sqrt", 0.3, 0.5],
    }

    # instantiate exhaustive grid search of the above parameter grid with c-index as the scoring metric 
    gcv = GridSearchCV(
        pipe,
        param_grid,
        cv=folds,
        n_jobs=1,
        scoring=cindex_scorer
    )

    # fit the grid search and extract the best pipeline, then return it 
    gcv.fit(X, y)
    best_rsf = gcv.best_estimator_

    return best_rsf



def svm_survival(X, y): 

    # 5-fold scheme
    folds = KFold(n_splits=5, shuffle=True, random_state=3)

    # create SVM pipeline with scaling included 
    ssvm = make_pipeline(
        StandardScaler(), 
        FastSurvivalSVM(
            rank_ratio=1.0,
            max_iter=1000,
            tol=1e-5,
            random_state=folds.random_state
        )
    )

    # parameter grid of alphas 
    param_grid = {
        "fastsurvivalsvm__alpha": [2.0**v for v in range(-8, 9, 2)],
    }

    # grid search to find the best alpha using c-index as the scoring metric 
    gcv = GridSearchCV(
        ssvm,
        param_grid,
        cv=folds,
        n_jobs=18, 
        scoring=cindex_scorer
    )
    
    # Fit grid search and extract the best model, then return it 
    gcv.fit(X, y)
    best_svm = gcv.best_estimator_

    return best_svm



def survival_boost(X, y): 

    # define fold scheme 
    folds = KFold(n_splits=5, shuffle=True, random_state=3)

    # gradient boosting pipeline with scaling, coxph loss 
    est = make_pipeline(StandardScaler(), ComponentwiseGradientBoostingSurvivalAnalysis(
        loss="coxph",
        random_state=folds.random_state
    ))

    # create parameter search space 
    param_grid = {
        "componentwisegradientboostingsurvivalanalysis__learning_rate": [0.05, 0.1],
        "componentwisegradientboostingsurvivalanalysis__n_estimators": [100, 200, 400],
        "componentwisegradientboostingsurvivalanalysis__subsample": [0.5, 1.0],
        "componentwisegradientboostingsurvivalanalysis__dropout_rate": [0.0, 0.1],
    }

    # set up exhaustive grid search 
    gcv = GridSearchCV(
        est,
        param_grid=param_grid,
        cv=folds,
        n_jobs=18,
        scoring=cindex_scorer
    )

    # execute the grid search and extract the best pipeline, then return it 
    gcv.fit(X, y)
    best_model = gcv.best_estimator_

    return best_model


# custom transformer for true quantile normalization, for later use once computational resources are not limited
class QuantileNorm(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.rank_means = None

    def fit(self, X, y=None):

        # ensure data is in dataframe format
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        elif not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be numpy array or pandas dataframe...")

        # calculate rank means
        sort_in_sample = np.sort(X.values, axis=1)

        # average the expression of each rank
        self.rank_means = np.mean(sort_in_sample, axis=0)

        return self

    def transform(self, X):

        # ensure input is dataframe for easier manipulation
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        elif not isinstance(X, pd.DataFrame):
            raise ValueError("Input should be a ndarray or dataframe")

        # rank within sample
        ranked_x = X.rank(axis=1, method='min').astype(int) - 1

        # replace each value in ranked_x with the rank means in the np.array
        mapped_means = np.array(self.rank_means)[ranked_x.values]

        # convert back to df
        normed_x = pd.DataFrame(mapped_means, index=ranked_x.index, columns=ranked_x.columns)

        return normed_x



def get_external_expr(encoded_gender): 

    # load data
    ex = pd.read_csv(".\\GSE13507_RAW\\GSE13507_illumina_raw.txt", sep="\t", index_col=0) 

    # load conversion key 
    gene_table = pd.read_csv(".\\GSE13507_RAW\GPL6102_Illumina_HumanWG-6_V2_0_R1_11223189_A.bgx", sep="\t", skiprows=8).set_index("Probe_Id")["ILMN_Gene"].to_dict()

    # change probes to be columns rather than rows, then convert to gene names
    ex = ex.T
    ex = ex.rename(columns=gene_table)
    
    # keep only columns that were in the dictionary
    ex = ex[[col for col in ex.columns if col in set(gene_table.values())]]

    # average duplicate gene columns
    ex = ex.T.groupby(level=0).mean().T

    # log2 transform with a floor - negative and zero values are converted to 1
    ex[ex <= 0] = 1 
    ex_log = np.log2(ex)

    # quantile normalize 
    ex_log_qn = QuantileNorm().fit_transform(ex_log)
    
    # now determine which samples are control/tumor by getting key and converting 
    sample_key = pd.read_csv(".\\GSE13507_RAW\\GSE13507_key.txt", sep="\t", header=None)
    tumor_meta = sample_key[sample_key[1].str.contains("bladder", case=False)]

    # convert to study ID rather than GEO ID for purposes of identifying sex
    id_map = dict(zip(tumor_meta[0], tumor_meta[1].str.extract(r'(BT\d+)')[0]))

    # only include tumor samples 
    tumor_data = ex_log_qn.loc[ex_log_qn.index.isin(tumor_meta[0])]
    tumor_data.index = tumor_data.index.map(id_map)
    # make the ID a column rather than the index
    tumor_data = tumor_data.reset_index()

    # now determine the sex of each tumor sample with the appropriate key 
    clinical = pd.read_csv(".\\GSE13507_RAW\\GSE13507_clinical_info.csv").set_index("Sample name")[["SEX", "overall survival", "survivalMonth"]]
    # change such that 1 = death and 0 = survived for OS 
    clinical["overall survival"] = clinical["overall survival"] - 1
    # merge clinical info with expression data 
    merged = clinical.merge(tumor_data, left_index=True, right_on='index', how='inner')
    
    # get gender of interest 
    if encoded_gender == 1: 
        merged = merged[merged["SEX"] == "M"]
    else:
        merged = merged[merged["SEX"] == "F"]

    merged = merged.drop(["SEX"], axis=1)

    # X is just the expression data
    X = merged.drop(["survivalMonth", "overall survival", "index"], axis=1)

    # y is survival info
    y = merged[["survivalMonth", "overall survival"]].copy()

    # rename to standard for prognostic modeling and make sure vital status is boolean 
    y = y.rename(columns={"survivalMonth":"duration", "overall survival":"event"})
    y["event"] = y["event"].astype(bool)
    y = y[["event", "duration"]]
    y = y.to_records(index=False)

    return X, y





if __name__ == "__main__":

    # iterate through all evaluation models and execute external validation on GSE13507 for both sexes 
    for model in ["coxnet", "rf", "svm", "boost"]:
        print(model)
        eval_aggy("expr", 1, model)
        eval_aggy("expr", 0, model)
