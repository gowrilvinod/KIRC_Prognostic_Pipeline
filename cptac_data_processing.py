import pandas as pd
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import warnings

import data_processing as dp


def get_cptac_survival_data(clinical_path=Path("./CPTAC_KIRC/Metadata/clinical.tsv")):
    """
    Loads and cleans CPTAC clinical survival data (event, duration, Gender) indexed by case_id.
    Duration is in days, matching TCGA clinical processing in data_processing.py.
    """
    df_clinical = dp.load_clean_kirc_clinical(clinical_path)
    return df_clinical[["event", "duration", "Gender"]]


def get_cptac_expr(ids="gene_name", format="tpm", limit=False):
    """
    Loads CPTAC-KIRC RNA-seq gene expression data aligned with the clean CPTAC clinical cohort.
    Applies log2(TPM + 1) transformation and top 10,000 highest-variance gene filtering,
    matching the exact methodology used for TCGA in data_processing.py.
    """
    clinical_path = Path("./CPTAC_KIRC/Metadata/clinical.tsv")
    clean_clinical = dp.load_clean_kirc_clinical(clinical_path)

    biospec_path = Path("./CPTAC_KIRC/Metadata/cptac_biospecimen_map.json")
    if not biospec_path.exists():
        raise FileNotFoundError(f"CPTAC biospecimen map not found at {biospec_path}")

    with open(biospec_path, "r") as f:
        biospec_map = json.load(f)

    clean_case_ids = set(clean_clinical.index)
    case_to_file = {}
    for fname, info in biospec_map.items():
        case_id = info["case_id"]
        if case_id in clean_case_ids:
            if case_id not in case_to_file:
                case_to_file[case_id] = fname
            else:
                case_to_file[case_id] = min(case_to_file[case_id], fname)

    expression_dir = Path("./CPTAC_KIRC/CPTAC_RNA_seq")
    value_col = "unstranded" if format == "counts" else "tpm_unstranded"

    expr_dict = {}
    print(f"Loading CPTAC expression data for {len(case_to_file)} patients...")
    for case_id, fname in tqdm(case_to_file.items()):
        fpath = expression_dir / fname
        if not fpath.exists():
            continue
        patient = pd.read_csv(fpath, sep="\t", skiprows=[0, 2, 3, 4, 5])
        expr_series = patient.set_index(ids)[value_col]
        expr_dict[case_id] = expr_series

    all_expr = pd.DataFrame(expr_dict)

    if format == "counts":
        all_expr = all_expr.groupby(all_expr.index).sum()
    else:
        all_expr = all_expr.groupby(all_expr.index).mean()

    # Transpose to have patients as rows
    all_expr = all_expr.T

    if format == "tpm":
        # Log2 transform with pseudocount and filter top 10,000 variance genes
        all_expr = np.log2(all_expr + 1)
        gene_var = all_expr.var(axis=0)
        top_genes = gene_var.nlargest(10000).index
        all_expr = all_expr[top_genes]

    return all_expr
