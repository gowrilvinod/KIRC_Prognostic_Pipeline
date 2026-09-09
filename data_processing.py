import pandas as pd
from pathlib import Path
import json 
import numpy as np 
from lifelines import CoxPHFitter 
from tqdm import tqdm
import warnings 
from sklearn.impute import KNNImputer 


def get_paths(directory):
    """Utility to list sorted file paths within a directory."""
    directory = Path(directory)
    return sorted([p for p in directory.iterdir() if p.is_file()])


def load_clean_kirc_clinical(clinical_path=Path("./TCGA_KIRC_Clinical/clinical.tsv")):
    """
    Loads and cleans TCGA-KIRC clinical metadata adhering to exact cohort filtering:
      1. Primary disease diagnosis rows only (deduplicating case IDs).
      2. AJCC Pathologic Stage I-IV.
      3. Tumor Grade G1-G4.
      4. Metastasis M0 and M1.
      5. Valid survival duration (> 0 days).
      6. Exclusion of short-term follow-up (duration > 30 days).
      7. Valid binary gender mapping (male: 1, female: 0).
    """
    clinical_path = Path(clinical_path)
    if not clinical_path.exists():
        raise FileNotFoundError(f"Clinical file not found at {clinical_path}")
        
    df = pd.read_csv(clinical_path, sep="\t")
    
    # 1. Filter for primary disease diagnosis rows
    if 'diagnoses.diagnosis_is_primary_disease' in df.columns:
        primary_df = df[df['diagnoses.diagnosis_is_primary_disease'].astype(str).str.lower() == 'true']
    else:
        primary_df = df
        
    clean_df = primary_df.groupby('cases.case_id').first().copy()
    
    # 2. Retain AJCC Stage I-IV
    if 'diagnoses.ajcc_pathologic_stage' in clean_df.columns:
        clean_df = clean_df[clean_df['diagnoses.ajcc_pathologic_stage'].isin(['Stage I', 'Stage II', 'Stage III', 'Stage IV'])]
    
    # 3. Retain Grade G1-G4
    if 'diagnoses.tumor_grade' in clean_df.columns:
        clean_df = clean_df[clean_df['diagnoses.tumor_grade'].isin(['G1', 'G2', 'G3', 'G4'])]
    
    # 4. Retain M0 and M1 patients
    if 'diagnoses.ajcc_pathologic_m' in clean_df.columns:
        clean_df = clean_df[clean_df['diagnoses.ajcc_pathologic_m'].isin(['M0', 'M1'])]
    
    # 5. Parse survival data
    clean_df["demographic.days_to_death"] = pd.to_numeric(clean_df["demographic.days_to_death"], errors="coerce")
    clean_df["diagnoses.days_to_last_follow_up"] = pd.to_numeric(clean_df["diagnoses.days_to_last_follow_up"], errors="coerce")
    
    clean_df["event"] = clean_df["demographic.vital_status"].map({"Dead": 1, "Alive": 0})
    clean_df["duration"] = clean_df["demographic.days_to_death"]
    clean_df.loc[clean_df["event"] == 0, "duration"] = clean_df.loc[clean_df["event"] == 0, "diagnoses.days_to_last_follow_up"]
    
    # Exclude missing survival info or non-positive durations
    clean_df = clean_df.dropna(subset=["event", "duration"])
    clean_df = clean_df[clean_df["duration"] > 0]
    
    # Exclude short-term follow-up / perioperative mortality (duration <= 30 days)
    clean_df = clean_df[clean_df["duration"] > 30]
    
    # 6. Encode Gender
    gender_col = "demographic.sex_at_birth" if "demographic.sex_at_birth" in clean_df.columns else "demographic.gender"
    clean_df["Gender"] = clean_df[gender_col].map({"male": 1, "female": 0})
    clean_df = clean_df.dropna(subset=["Gender"])
    clean_df["Gender"] = clean_df["Gender"].astype(int)
    
    return clean_df


def get_survival_data(clinical_path=Path("./TCGA_KIRC_Clinical/clinical.tsv")):
    """Returns clean clinical survival data (event, duration, Gender) indexed by case_id."""
    clean_df = load_clean_kirc_clinical(clinical_path)
    return clean_df[["event", "duration", "Gender"]]


def get_tcga_expr(ids="gene_name", format='tpm', limit=False):
    """
    Loads TCGA-KIRC RNA-seq gene expression data aligned with the clean KIRC clinical cohort.
    Applies aliquot deduplication (-01A- rank), log2(TPM + 1) transformation,
    and top 10,000 highest-variance gene filtering.
    """
    clean_clinical = load_clean_kirc_clinical()
    
    metadata_path = Path('./Metadata/tcga_kirc_rna_metadata.json')
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
        
    with open(metadata_path, 'r') as metadata_file:
        metadata = json.load(metadata_file)
        
    meta_rows = []
    for item in metadata:
        ent = item["associated_entities"][0]
        meta_rows.append({
            "file_name": item["file_name"],
            "case_id": ent["case_id"],
            "aliquot_id": ent["entity_submitter_id"]
        })
    meta_df = pd.DataFrame(meta_rows)
    
    # Rank aliquots to prefer primary solid tumor -01A-
    def aliquot_rank(a):
        if "-01A-" in a: return 0
        if "-01B-" in a: return 1
        return 2
    meta_df["rank"] = meta_df["aliquot_id"].apply(aliquot_rank)
    meta_df = meta_df.sort_values(["case_id", "rank", "aliquot_id"]).drop_duplicates(subset="case_id", keep="first")
    
    # Filter metadata for cases in the clean KIRC clinical cohort
    meta_df = meta_df[meta_df["case_id"].isin(clean_clinical.index)]
    metadata_dict = dict(zip(meta_df["file_name"], zip(meta_df["case_id"], meta_df["aliquot_id"])))
    
    rna_dir = Path("./TCGA_KIRC_RNA_seq")
    tcga_files = list(rna_dir.glob("**/*.rna_seq.augmented_star_gene_counts.tsv"))
    
    expr_dict = {}
    value_col = 'unstranded' if format == 'counts' else 'tpm_unstranded'
    
    print(f"Loading KIRC expression data for {len(metadata_dict)} patients...")
    for file in tqdm(tcga_files):
        filename = file.name
        if filename not in metadata_dict:
            continue
        caseid, aliquot_id = metadata_dict[filename]
        
        patient = pd.read_csv(file, sep="\t", skiprows=[0, 2, 3, 4, 5])
        expr_series = patient.set_index(ids)[value_col]
        expr_dict[caseid] = expr_series
        
    all_expr = pd.DataFrame(expr_dict)
    
    # Remove Ensembl version numbers if gene_id is selected
    if ids == "gene_id":
        all_expr.index = all_expr.index.str.replace(r'\..*', '', regex=True)
        
    # Group by duplicate gene names/IDs
    if format == 'counts':
        all_expr = all_expr.groupby(all_expr.index).sum()
    else:
        all_expr = all_expr.groupby(all_expr.index).mean()
        
    # Transpose to have patients as rows
    all_expr = all_expr.T
    
    if format == 'tpm':
        # Log2 transform with pseudocount and filter top 10,000 variance genes
        all_expr = np.log2(all_expr + 1)
        gene_var = all_expr.var(axis=0)
        top_genes = gene_var.nlargest(10000).index
        all_expr = all_expr[top_genes]
        
    return all_expr


def get_tcga_methy(limit=False): 
    """Loader for KIRC methylation data aligned with clean KIRC clinical cohort."""
    clean_clinical = load_clean_kirc_clinical()
    
    metadata_path = Path('./Metadata/tcga_kirc_methy_metadata.json')
    if not metadata_path.exists():
        # Return empty DataFrame with Gender column matching clinical if methylation metadata not present
        res = pd.DataFrame(index=clean_clinical.index)
        res['Gender'] = clean_clinical['Gender']
        return res
        
    with open(metadata_path, 'r') as metadata_file:
        metadata = json.load(metadata_file)

    meta_rows = []
    for item in metadata:
        ent = item["associated_entities"][0]
        meta_rows.append({
            "file_name": item["file_name"],
            "case_id": ent["case_id"],
            "aliquot_id": ent["entity_submitter_id"]
        })
    meta_df = pd.DataFrame(meta_rows)

    def aliquot_rank(a):
        if "-01A-" in a: return 0
        if "-01B-" in a: return 1
        return 2
    meta_df["rank"] = meta_df["aliquot_id"].apply(aliquot_rank)
    meta_df = meta_df.sort_values(["case_id", "rank", "aliquot_id"]).drop_duplicates(subset="case_id", keep="first")
    meta_df = meta_df[meta_df["case_id"].isin(clean_clinical.index)]
    metadata_dict = dict(zip(meta_df["file_name"], meta_df["case_id"]))

    methy_dir = Path("./TCGA_KIRC_Methy")
    if not methy_dir.exists():
        res = pd.DataFrame(index=clean_clinical.index)
        res['Gender'] = clean_clinical['Gender']
        return res

    methylation = pd.DataFrame()
    tcga_files = get_paths(methy_dir)
    
    for file in tcga_files:
        filename = str(Path(file).name)
        if filename not in metadata_dict:
            continue
        caseid = metadata_dict[filename]
        patient = pd.read_csv(file, sep="\t", index_col=0, header=None)
        patient.columns = [caseid]
        methylation = pd.concat([methylation, patient], axis=1)

    methylation.index.name = 'IlmnID'
    manifest_path = Path("./Metadata/humanmethylation450_15017482_v1-2.csv")
    if manifest_path.exists():
        ill_manifest = pd.read_csv(manifest_path, skiprows=7)[["IlmnID", "UCSC_RefGene_Name"]].dropna(axis=0)
        ill_manifest = ill_manifest.assign(UCSC_RefGene_Name=ill_manifest['UCSC_RefGene_Name'].str.split(';')).explode('UCSC_RefGene_Name')
        ill_manifest = ill_manifest.drop_duplicates(subset=["IlmnID", "UCSC_RefGene_Name"])

        beta_long = methylation.reset_index().melt(id_vars='IlmnID', var_name='Patient_ID', value_name='beta')
        beta_long = beta_long.merge(ill_manifest, on='IlmnID', how='inner')
        averaged_methy = (beta_long.groupby(['Patient_ID', 'UCSC_RefGene_Name'])['beta'].mean().unstack())
        averaged_m_vals = np.log2((averaged_methy + 1e-6) / (1 - averaged_methy + 1e-6))

        gene_var = averaged_m_vals.var(axis=0)
        top_genes = gene_var.nlargest(10000).index
        averaged_m_vals = averaged_m_vals[top_genes]
        final_methy = averaged_m_vals
    else:
        final_methy = methylation.T

    final_methy['Gender'] = final_methy.index.map(lambda cid: clean_clinical.loc[cid, 'Gender'])
    final_methy = final_methy.dropna(axis=1)
    return final_methy


def get_tcga_mirna(limit=False):
    """Loader for KIRC miRNA data aligned with clean KIRC clinical cohort."""
    clean_clinical = load_clean_kirc_clinical()
    res = pd.DataFrame(index=clean_clinical.index)
    res['Gender'] = clean_clinical['Gender']
    return res


def get_tcga_protein(limit=False): 
    """Loader for KIRC proteomics data with KNN imputation and missing value thresholding."""
    clean_clinical = load_clean_kirc_clinical()
    
    metadata_path = Path('./Metadata/tcga_kirc_protein_metadata.json')
    protein_dir = Path("./TCGA_KIRC_Protein")
    
    if not metadata_path.exists() or not protein_dir.exists():
        res = pd.DataFrame(index=clean_clinical.index)
        res['Gender'] = clean_clinical['Gender']
        return res
        
    with open(metadata_path, 'r') as metadata_file:
        metadata = json.load(metadata_file)

    meta_rows = []
    for item in metadata:
        ent = item["associated_entities"][0]
        meta_rows.append({
            "file_name": item["file_name"],
            "case_id": ent["case_id"],
            "aliquot_id": ent["entity_submitter_id"]
        })
    meta_df = pd.DataFrame(meta_rows)

    def aliquot_rank(a):
        if "-01A-" in a: return 0
        if "-01B-" in a: return 1
        return 2
    meta_df["rank"] = meta_df["aliquot_id"].apply(aliquot_rank)
    meta_df = meta_df.sort_values(["case_id", "rank", "aliquot_id"]).drop_duplicates(subset="case_id", keep="first")
    meta_df = meta_df[meta_df["case_id"].isin(clean_clinical.index)]
    metadata_dict = dict(zip(meta_df["file_name"], meta_df["case_id"]))

    protein = pd.DataFrame()
    tcga_files = get_paths(protein_dir)
    for file in tcga_files:
        filename = str(Path(file).name)
        if filename not in metadata_dict:
            continue
        caseid = metadata_dict[filename]
        patient = pd.read_csv(file, sep="\t", index_col="peptide_target", header=0)["protein_expression"]
        patient.name = caseid
        protein = pd.concat([protein, patient], axis=1)

    imputer = KNNImputer(n_neighbors=5)
    protein_filtered = protein[protein.isna().mean(axis=1) <= 0.2]
    protein_df = pd.DataFrame(imputer.fit_transform(protein_filtered.T), index=protein_filtered.columns, columns=protein_filtered.index)
    protein_df['Gender'] = protein_df.index.map(lambda cid: clean_clinical.loc[cid, 'Gender'])
    protein_df = protein_df.dropna(axis=1)
    return protein_df


def univariate_cox(df):
    """
    Performs univariate Cox proportional hazards regression for each feature in df.
    Expects df to contain 'duration' and 'event' as the first two columns.
    """
    results = []
    print("Starting univariate Cox for all features:")
    for gene in tqdm(df.columns.to_list()[2:]):
        uni_cox = CoxPHFitter(penalizer=0.01)
        one_gene = df[["duration", "event", gene]]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                uni_cox.fit(one_gene, duration_col="duration", event_col="event")

            summary = uni_cox.summary.loc[gene]
            results.append({
                "variable": gene,
                "HR": summary["exp(coef)"],
                "CI_lower": summary["exp(coef) lower 95%"],
                "CI_upper": summary["exp(coef) upper 95%"],
                "p_value": summary["p"]
            })
        except Exception: 
            continue
    
    univariate_printout = pd.DataFrame(results).sort_values(by="p_value", ascending=True)
    return univariate_printout
