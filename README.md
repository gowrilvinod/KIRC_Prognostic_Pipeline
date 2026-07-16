# Kidney Renal Clear Cell Carcinoma (KIRC) Prognostic Multi-Omics Analysis

This repository contains python and R code for running survival analysis and machine learning-based prognostic models on multi-omics cancer data, focusing on Kidney Renal Clear Cell Carcinoma (KIRC) and related datasets.

## Project Structure

* **`execution.py`**: The main execution driver script for nested cross-validation, feature discovery, and survival model evaluations.
* **`data_processing.py`**: Helper module containing functions to parse and preprocess clinical, expression, miRNA, methylation, and protein data.
* **`ensemble.py`**: Implementation of ensemble prognostic model building and integration.
* **`eval.py`**: Code for scoring models, evaluating performance metrics, and performing survival curves analyses.
* **`aggregate_R.R`**: Statistical scripts for aggregating feature weights and ranks across folds.
* **`multi_prog_folded_select.sh`**: Bash pipeline script for running multiple model folds and random seeds in parallel/batch mode.
* **`bca_env.yml`**: Conda virtual environment configuration file with all necessary python packages and dependencies.

## Installation & Setup

1. Make sure you have [Conda](https://docs.conda.io/en/latest/) installed.
2. Recreate the environment using the provided environment file:
   ```bash
   conda env create -f bca_env.yml
   ```
3. Activate the new conda environment:
   ```bash
   conda activate multi_omics_S26
   ```

## Usage

To run a nested cross-validation fold for prognostic feature discovery, execute the main script with custom parameters:
```bash
python execution.py --rando 42 --data_type expr --compare_after_cox yes
```

### Script Arguments:
* `--rando`: Random seed/state integer (used for shuffling splits).
* `--data_type`: Type of omics data to analyze (`expr` for gene expression, `methy` for methylation, `mirna` for miRNA, `protein` for proteomics).
* `--compare_after_cox`: Option (`yes`/`no`) to run downstream comparative evaluation after Cox filtering.
