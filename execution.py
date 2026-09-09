from data_processing import * 
from ensemble import * 
import argparse 
import os 
from sklearn.model_selection import KFold


# include arguments for multiple calls of this script 
# one call = one run of 5-fold nested cross validation 
parser = argparse.ArgumentParser()
parser.add_argument("--rando", type=int, required=True)
parser.add_argument("--data_type", type=str, required=True)
parser.add_argument("--compare_after_cox", type=str, default="Y")
args = parser.parse_args() 

rando = args.rando
data_type = args.data_type
compare_after_cox = args.compare_after_cox


def nested_discovery(data_type, rando):

    # get survival information from TCGA: vital status and survival time 
    clinical = get_survival_data()

    # based on the desired type, get the TCGA omics data 
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
    male_data = omic_w_clinical[omic_w_clinical["Gender"] == 1].drop("Gender", axis=1)
    female_data = omic_w_clinical[omic_w_clinical["Gender"] == 0].drop("Gender", axis=1)

    results = {}

    # prepare data - create dictionary where data matrix and target matrix are stored for each sex 
    sex_data = {}
    for sex, df in {"male": male_data, "female": female_data}.items():

        # drop survival info from omics data
        X = df.drop(["duration", "event"], axis=1)
        # and store in y 
        y = df[["event", "duration"]].copy()
        # ensure vital status is boolean and convert y to a records object for sksurv compatibility 
        y["event"] = y["event"].astype(bool)
        y = y.to_records(index=False)

        # store in dictionary 
        sex_data[sex] = (X, y)

    # separate outer splits for each gender 
    outer_cv_splits = {
        sex: list(KFold(n_splits=5, shuffle=True, random_state=rando).split(sex_data[sex][0]))
        for sex in ["male", "female"]
    }

    # iterate over folds
    for fold in range(5):
        # Checkpoint resumption logic
        male_path = os.path.expanduser(
            f"~/spring_2026/results_folded_1/"
            f"{data_type}_male_fold_{fold}_rando{rando}.csv"
        )
        female_path = os.path.expanduser(
            f"~/spring_2026/results_folded_1/"
            f"{data_type}_female_fold_{fold}_rando{rando}.csv"
        )
        if (os.path.exists(male_path) and os.path.getsize(male_path) > 0 and 
            os.path.exists(female_path) and os.path.getsize(female_path) > 0):
            print(f"Fold {fold} already fully completed for both sexes, skipping.")
            continue

        uni_results = {}

        # run univariate cox separately per sex using its own splits
        for sex in ["male", "female"]:

            # get X and y for the desired sex
            X, y = sex_data[sex]

            # get training and testing indices 
            train_idx, test_idx = outer_cv_splits[sex][fold]

            # apply train test split via indices on both X and y 
            X_train = X.iloc[train_idx]
            X_test  = X.iloc[test_idx]
            y_train = y[train_idx]
            y_test  = y[test_idx]

            # reconcatenate X and y for univariate cox 
            train_df = pd.concat(
                [
                    pd.DataFrame(y_train)[["duration", "event"]].reset_index(drop=True),
                    X_train.reset_index(drop=True)
                ],
                axis=1
            )

            # insert variance filter here 

            # execute univariate cox selection process as a filtering stage (similar to DGEA in my other paper)
            uni = univariate_cox(train_df)

            # extract genes with a univariate p_value < 0.05 for further analysis 
            sig_genes = set(
                uni.loc[uni["p_value"] < 0.05, "variable"].tolist()
            )

            # store results in a dictionary with sex as the key 
            uni_results[sex] = (
                X_train,
                X_test,
                y_train,
                y_test,
                sig_genes,
                uni
            )

        # remove overlapping genes between males and females 
        # this acts as a sex-specific filter - remove genes that are tumor-related regardless of sex 
        male_specific_raw   = uni_results["male"][4]   - uni_results["female"][4]
        female_specific_raw = uni_results["female"][4] - uni_results["male"][4]

        # Sort by p-value in their respective univariate analysis, and keep top 200
        male_uni_df = uni_results["male"][5]
        female_uni_df = uni_results["female"][5]

        male_specific_sorted = male_uni_df[male_uni_df["variable"].isin(male_specific_raw)].sort_values("p_value")
        female_specific_sorted = female_uni_df[female_uni_df["variable"].isin(female_specific_raw)].sort_values("p_value")

        male_specific = set(male_specific_sorted["variable"].head(200).tolist())
        female_specific = set(female_specific_sorted["variable"].head(200).tolist())

        # determine overlap for fun 
        overlap = uni_results["male"][4] & uni_results["female"][4]

        # display results of sex-specific filtering 
        print(
            f"Fold {fold} - "
            f"Male specific: {len(male_specific)}, "
            f"Female specific: {len(female_specific)}, "
            f"Overlap removed: {len(overlap)}"
        )

        # ensemble step
        for sex, specific_genes in {
            "male": male_specific,
            "female": female_specific
        }.items():

            # for given sex get training, testing data 
            X_train, X_test, y_train, y_test, _, _ = uni_results[sex]

            # make sure there were sex-specific genes found 
            if len(specific_genes) == 0:
                print(f"Fold {fold} {sex}: no specific genes, skipping")
                continue

            # convert sex-specific genes to list and limit training data to them
            specific_genes = list(specific_genes)
            X_train_f = X_train[specific_genes]
            X_test_f  = X_test[specific_genes]

            # create inner 5-fold split for hyperparameter tuning 
            inner_cv = KFold(n_splits=5, shuffle=True, random_state=rando)

            # execute ensemble - fit four different prognostic models on the training data for this fold scheme 
            # extract the top genes based on model coefficients or importance 
            cox_genes   = coxnet(X_train_f, y_train, inner_cv)
            rf_genes    = rf_survival(X_train_f, y_train, inner_cv)
            boost_genes = survival_booster(X_train_f, y_train, inner_cv)
            svm_genes   = survival_svm(X_train_f, y_train, inner_cv)

            # insert internal validation here 

            # this method of storing the rankings is a bit antiquated 
            # it was meant for a python implementation of RRA
            # But the aggregate_R is written to be compatible with this format
            rra_rows = []
            qid = "Q1"

            # for each ranking model
            for model_name, ranked_genes in {
                "coxnet": cox_genes,
                "rf": rf_genes,
                "boost": boost_genes,
                "svm": svm_genes
            }.items():

                # voter includes the selection model, fold scheme #, and the current random state 
                voter = f"{model_name}_fold{fold}_iter{rando}"
                n_genes = len(ranked_genes)

                if n_genes == 0:
                    continue

                # score is just based on rank and number of total genes 
                for rank, gene in enumerate(ranked_genes, start=1):

                    score = 1.0 - ((rank - 1) / n_genes)

                    rra_rows.append(
                        (qid, voter, gene, rank, score, "ensemble")
                    )

            # create data frame of all rankings for this given fold scheme and random state 
            rra_df = pd.DataFrame(
                rra_rows,
                columns=["Query", "Voter", "ItemID", "Rank", "Score", "Algorithm"]
            )

            out_path = os.path.expanduser(
                f"~/spring_2026/results_folded_1/"
                f"{data_type}_{sex}_fold_{fold}_rando{rando}.csv"
            )
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            # 20 iterations of 5-fold nested CV = 100 rankings * 2 genders = 200 rankings in results folder 
            rra_df.to_csv(out_path, index=False)

    return results

# this function is antiquated, runs pipeline without comparing between genders 
# the result was most of the genes were the same 
def nested_discovery_no_cross(data_type, rando):

    clinical = get_survival_data()

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
    male_data = omic_w_clinical[omic_w_clinical["Gender"] == 1].drop("Gender", axis=1)
    female_data = omic_w_clinical[omic_w_clinical["Gender"] == 0].drop("Gender", axis=1)

    results = {}

    # prepare data
    sex_data = {}
    for sex, df in {"male": male_data, "female": female_data}.items():

        X = df.drop(["duration", "event"], axis=1)

        y = df[["event", "duration"]].copy()
        y["event"] = y["event"].astype(bool)
        y = y.to_records(index=False)

        sex_data[sex] = (X, y)

    # PRECOMPUTE OUTER SPLITS SEPARATELY FOR EACH SEX
    outer_cv_splits = {
        sex: list(KFold(n_splits=5, shuffle=True, random_state=rando).split(sex_data[sex][0]))
        for sex in ["male", "female"]
    }

    # iterate over folds
    for fold in range(5):
        # Checkpoint resumption logic
        male_path = os.path.expanduser(
            f"~/spring_2026/results_folded_2/"
            f"{data_type}_male_fold_{fold}_rando{rando}.csv"
        )
        female_path = os.path.expanduser(
            f"~/spring_2026/results_folded_2/"
            f"{data_type}_female_fold_{fold}_rando{rando}.csv"
        )
        if (os.path.exists(male_path) and os.path.getsize(male_path) > 0 and 
            os.path.exists(female_path) and os.path.getsize(female_path) > 0):
            print(f"Fold {fold} already fully completed for both sexes, skipping.")
            continue

        uni_results = {}

        # run univariate cox separately per sex using its own splits
        for sex in ["male", "female"]:

            X, y = sex_data[sex]

            train_idx, test_idx = outer_cv_splits[sex][fold]

            X_train = X.iloc[train_idx]
            X_test  = X.iloc[test_idx]

            y_train = y[train_idx]
            y_test  = y[test_idx]

            train_df = pd.concat(
                [
                    pd.DataFrame(y_train)[["duration", "event"]].reset_index(drop=True),
                    X_train.reset_index(drop=True)
                ],
                axis=1
            )

            uni = univariate_cox(train_df)

            sig_genes_raw = set(
                uni.loc[uni["p_value"] < 0.05, "variable"].tolist()
            )

            # Sort by p-value and keep top 200
            sig_genes_sorted = uni[uni["variable"].isin(sig_genes_raw)].sort_values("p_value")
            sig_genes = set(sig_genes_sorted["variable"].head(200).tolist())

            uni_results[sex] = (
                X_train,
                X_test,
                y_train,
                y_test,
                sig_genes
            )

        # ensemble step
        for sex in ["male", "female"]:

            X_train, X_test, y_train, y_test, sig_genes = uni_results[sex]

            if len(sig_genes) == 0:
                print(f"Fold {fold} {sex}: no significant genes, skipping")
                continue

            sig_genes = list(sig_genes)

            X_train_f = X_train[sig_genes]
            X_test_f  = X_test[sig_genes]

            inner_cv = KFold(n_splits=4, shuffle=True, random_state=rando)

            cox_genes   = coxnet(X_train_f, y_train, inner_cv)
            rf_genes    = rf_survival(X_train_f, y_train, inner_cv)
            boost_genes = survival_booster(X_train_f, y_train, inner_cv)
            svm_genes   = survival_svm(X_train_f, y_train, inner_cv)

            rra_rows = []
            qid = "Q1"

            for model_name, ranked_genes in {
                "coxnet": cox_genes,
                "rf": rf_genes,
                "boost": boost_genes,
                "svm": svm_genes
            }.items():

                voter = f"{model_name}_fold{fold}_iter{rando}"
                n_genes = len(ranked_genes)

                if n_genes == 0:
                    continue

                for rank, gene in enumerate(ranked_genes, start=1):

                    score = 1.0 - ((rank - 1) / n_genes)

                    rra_rows.append(
                        (qid, voter, gene, rank, score, "ensemble")
                    )

            rra_df = pd.DataFrame(
                rra_rows,
                columns=["Query", "Voter", "ItemID", "Rank", "Score", "Algorithm"]
            )

            out_path = os.path.expanduser(
                f"~/spring_2026/results_folded_2/"
                f"{data_type}_{sex}_fold_{fold}_rando{rando}.csv"
            )
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            rra_df.to_csv(out_path, index=False)

    return results


    

    
if __name__ == "__main__":

    if compare_after_cox == "Y":
        nested_discovery(data_type, rando)
    else: 
        nested_discovery_no_cross(data_type, rando)