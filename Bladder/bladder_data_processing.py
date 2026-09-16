import pandas as pd
from pathlib import Path
import json 
import numpy as np 
from lifelines import CoxPHFitter 
from tqdm import tqdm
import warnings 
from sklearn.impute import KNNImputer 



# function for getting all paths within a directory 
# for parsing TCGA data 
def get_paths(directory):
    directory = Path(directory)
    return sorted([p for p in directory.iterdir() if p.is_file()])


def get_tcga_expr(ids, format='counts', limit=False):

    # create empty dataframes to hold gene counts and TPM-normalized counts
    all_counts = pd.DataFrame()
    all_tpms = pd.DataFrame()

    # empty list to store target variable binary
    encoded_gender = []

    # load annotation data
    clinical_table = pd.read_csv(Path("./TCGA_BLCA_Clinical/clinical.tsv"), sep="\t")
    with open(Path('./Metadata/tcga_blca_rna_metadata.json'), 'r') as metadata_file:
        metadata = json.load(metadata_file)

    # create lookup dictionary for JSON
    metadata_dict = {item["file_name"]: item for item in metadata}

    # list containing stage of each 
    stage_info = []
    invasive = []
    grades = []

    
    tcga_files = get_paths(Path("./TCGA_BLCA_RNA_seq"))
    
    for file in tcga_files:
        # load each patient RNA-seq TSV, which includes raw counts and TPM values
        patient = pd.read_csv(file, sep="\t", skiprows=[0, 2, 3, 4, 5])

        # get name of current file 
        filename = str(Path(file).name)

        # use filename to get annotation data, then extract case ID
        caseid = metadata_dict.get(filename, {}).get("associated_entities")[0]['case_id']

        # extract counts and TPM
        gene_counts = patient[[ids, 'unstranded']].set_index(ids)
        tpms = patient[[ids, 'tpm_unstranded']].set_index(ids)

        # rename columns to patient case ID
        gene_counts.columns = [caseid]
        tpms.columns = [caseid]

        # append to running DataFrames
        all_counts = pd.concat([all_counts, gene_counts], axis=1)
        all_tpms = pd.concat([all_tpms, tpms], axis=1)

        # use the case ID to query the clinical info TSV
        patient = clinical_table[clinical_table['cases.case_id'] == caseid]

        # get tumor stage and append to running list
        try:
            tumor_stage = patient[patient['diagnoses.diagnosis_is_primary_disease'] == 'true']['diagnoses.ajcc_pathologic_stage'].iloc[0]
            metastat =  patient[patient['diagnoses.diagnosis_is_primary_disease'] == 'true']['diagnoses.ajcc_pathologic_m'].iloc[0]
            grade = patient[patient['diagnoses.diagnosis_is_primary_disease'] == 'true']['diagnoses.tumor_grade'].iloc[0]
        except IndexError: 
            tumor_stage = patient['diagnoses.ajcc_pathologic_stage'].iloc[0]
            metastat = patient['diagnoses.ajcc_pathologic_m'].iloc[0]
            grade = patient['diagnoses.tumor_grade'].iloc[0]
        stage_info.append(tumor_stage)
        invasive.append(metastat)
        grades.append(grade)

        # get gender 
        gender = patient['demographic.gender'].iloc[0]
        encoded_gender.append(gender)

    # remove version number from ENSEMBL ID then sum counts of duplicate indices or average them for TPM
    if ids == "gene_id":
        if format == 'counts':
            all_counts.index = all_counts.index.str.replace(r'\..*', '', regex=True)
        else:
            all_tpms.index = all_tpms.index.str.replace(r'\..*', '', regex=True)

    if format == 'counts':
        all_counts = all_counts.groupby(all_counts.index).sum()
        assert not all_counts.index.duplicated().any()
    else: 
        all_tpms = all_tpms.groupby(all_tpms.index).mean()
        assert not all_tpms.index.duplicated().any()



    # only turn this on if you want to remove lower grade and metastasized tumors 
    if limit == True: 
        # transpose counts/TPM to have patients as rows
        if format == 'counts':
            transposed = all_counts.T
        else:
            transposed = all_tpms.T

        # add clinical info
        transposed['stage'] = stage_info 
        transposed['metastasis'] = invasive 
        transposed['grade'] = grades
        transposed['gender'] = encoded_gender

        # filter tumors by stage, metastasis, and grade
        transposed = transposed[
            (transposed['stage'].isin(["Stage II", "Stage III", "Stage IV"])) &
            (transposed['metastasis'].isin(["M0", "MX"])) &
            (transposed['grade'] == "High Grade")
        ]

        # drop clinical info 
        if format == 'counts': 
            all_counts = transposed.drop(['stage', 'metastasis', 'grade', 'gender'], axis=1)
        else: 
            all_tpms = transposed.drop(['stage', 'metastasis', 'grade', 'gender'], axis=1)
            
    
    else: 

        if format == 'counts':
            all_counts = all_counts.T
            all_counts['Gender'] = encoded_gender
            all_counts["Gender"] = all_counts["Gender"].map({"male": 1, "female": 0})
        else:
            # log transform with pseudocount and remove genes with bottom 20% variance 
            all_tpms = np.log2(all_tpms.T + 1)
            # Keep top 10,000 genes by variance
            gene_var = all_tpms.var(axis=0)
            top_genes = gene_var.nlargest(10000).index
            all_tpms = all_tpms[top_genes]
            all_tpms['Gender'] = encoded_gender
            all_tpms["Gender"] = all_tpms["Gender"].map({"male": 1, "female": 0})


    if format == "counts":

        return all_counts

    elif format == "tpm":

        return all_tpms

    else:

        raise KeyError("Input valid format for expressions...")



def get_tcga_methy(limit=False): 

    # create empty dataframes to hold gene counts and TPM-normalized counts
    methylation = pd.DataFrame()

    # empty list to store target variable binary
    encoded_gender = []

    # load annotation data
    clinical_table = pd.read_csv(r"spring_2026/TCGA_BLCA_Clinical/clinical.tsv", sep="\t")
    with open(r'spring_2026/Metadata/tcga_blca_methy_metadata.json', 'r') as metadata_file:
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
        if "-01A-" in a:
            return 0
        if "-01B-" in a:
            return 1
        return 2

    meta_df["rank"] = meta_df["aliquot_id"].apply(aliquot_rank)
    meta_df = (meta_df.sort_values(["case_id", "rank", "aliquot_id"]).drop_duplicates(subset="case_id", keep="first"))

    # create lookup dictionary for JSON (file → case_id)
    metadata_dict = dict(zip(meta_df["file_name"], meta_df["case_id"]))

    # list containing stage of each 
    stage_info = []
    invasive = []
    grades = []

    tcga_files = get_paths(r"spring_2026/TCGA_BLCA_Methy")
    
    for file in tcga_files:
        patient = pd.read_csv(file, sep="\t", index_col=0, header=None)

        filename = str(Path(file).name)

        if filename not in metadata_dict:
            continue  # skip duplicate aliquots

        caseid = metadata_dict[filename]

        # rename columns to patient case ID
        patient.columns = [caseid]

        # append to running DataFrames
        methylation = pd.concat([methylation, patient], axis=1)

        # use the case ID to query the clinical info TSV
        patient = clinical_table[clinical_table['cases.case_id'] == caseid]

        try:
            tumor_stage = patient[patient['diagnoses.diagnosis_is_primary_disease'] == 'true']['diagnoses.ajcc_pathologic_stage'].iloc[0]
            metastat = patient[patient['diagnoses.diagnosis_is_primary_disease'] == 'true']['diagnoses.ajcc_pathologic_m'].iloc[0]
            grade = patient[patient['diagnoses.diagnosis_is_primary_disease'] == 'true']['diagnoses.tumor_grade'].iloc[0]
        except IndexError: 
            tumor_stage = patient['diagnoses.ajcc_pathologic_stage'].iloc[0]
            metastat = patient['diagnoses.ajcc_pathologic_m'].iloc[0]
            grade = patient['diagnoses.tumor_grade'].iloc[0]

        stage_info.append(tumor_stage)
        invasive.append(metastat)
        grades.append(grade)

        gender = patient['demographic.gender'].iloc[0]
        encoded_gender.append(gender)
    
    # create dataframe to store case ID and encoded gender: 
    clinical_df = pd.DataFrame({'Gender': encoded_gender, 'Stage': stage_info, 'Metastasis': invasive, 'Grade': grades}, index=methylation.columns)
    
    # name index for later merges 
    methylation.index.name = 'IlmnID'

    # map probes to genes - illumina ID to gene name. Remove probes without a match 
    ill_manifest = pd.read_csv("spring_2026/Metadata/humanmethylation450_15017482_v1-2.csv", skiprows=7)[["IlmnID", "UCSC_RefGene_Name"]].dropna(axis=0)

    # one probe can have multiple associated genes, explode the dataframe such that these are separated into different rows
    ill_manifest = ill_manifest.assign(UCSC_RefGene_Name=ill_manifest['UCSC_RefGene_Name'].str.split(';')).explode('UCSC_RefGene_Name')

    # remove duplicate gene-probe pairs 
    ill_manifest = ill_manifest.drop_duplicates(subset=["IlmnID", "UCSC_RefGene_Name"])

    # turn into long format, where each row is probe-patient pair 
    beta_long = methylation.reset_index().melt(id_vars='IlmnID', var_name='Patient_ID', value_name='beta')

    # merge on probe ID, such that a gene name is assigned to each entry
    beta_long = beta_long.merge(ill_manifest, on='IlmnID', how='inner')

    # average duplicate gene names 
    averaged_methy = (beta_long.groupby(['Patient_ID', 'UCSC_RefGene_Name'])['beta'].mean().unstack())

    # convert to M values
    averaged_m_vals = np.log2((averaged_methy + 1e-6) / (1 - averaged_methy + 1e-6))

    # select top 10,000 features in terms of variance 
    gene_var = averaged_m_vals.var(axis=0)
    top_genes = gene_var.nlargest(10000).index
    averaged_m_vals = averaged_m_vals[top_genes]

    # align clinical info to M values 
    clinical_df = clinical_df.reindex(averaged_m_vals.index)

    if limit == True: 

        # concatenate clinical and methylation data 
        transposed = pd.concat([averaged_m_vals, clinical_df], axis=1)

        # filter tumors by stage, metastasis, and grade
        transposed = transposed[
            (transposed['Stage'].isin(["Stage II", "Stage III", "Stage IV"])) &
            (transposed['Metastasis'].isin(["M0", "MX"])) &
            (transposed['Grade'] == "High Grade")
        ]

        # drop clinical info 
        final_methy = transposed.drop(['Stage', 'Metastasis', 'Grade'], axis=1)
    
    else: 

        final_methy = pd.concat([averaged_m_vals, clinical_df['Gender']], axis=1)

    # encode gender to binary 
    final_methy["Gender"] = final_methy["Gender"].map({"male": 1, "female": 0})

    # drop genes with missing values 
    final_methy = final_methy.dropna(axis=1)

    return final_methy


def get_tcga_mirna(limit=False): 

    # create empty dataframes to hold gene counts and TPM-normalized counts
    mirna = pd.DataFrame()

    # empty list to store target variable binary
    encoded_gender = []

    # load annotation data
    clinical_table = pd.read_csv(r"spring_2026/TCGA_BLCA_Clinical/clinical.tsv", sep="\t")
    with open(r'spring_2026/Metadata/tcga_blca_mirna_metadata.json', 'r') as metadata_file:
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
        if "-01A-" in a:
            return 0
        if "-01B-" in a:
            return 1
        return 2

    meta_df["rank"] = meta_df["aliquot_id"].apply(aliquot_rank)
    meta_df = (meta_df.sort_values(["case_id", "rank", "aliquot_id"]).drop_duplicates(subset="case_id", keep="first"))

    # create lookup dictionary for JSON (file → case_id)
    metadata_dict = dict(zip(meta_df["file_name"], meta_df["case_id"]))

    # list containing stage of each 
    stage_info = []
    invasive = []
    grades = []

    tcga_files = get_paths(r"spring_2026/TCGA_BLCA_miRNA")
    
    for file in tcga_files:
        patient = pd.read_csv(file, sep="\t", index_col=0, header=0)["reads_per_million_miRNA_mapped"]

        filename = str(Path(file).name)

        if filename not in metadata_dict:
            continue  # skip duplicate aliquots

        caseid = metadata_dict[filename]

        # rename columns to patient case ID
        patient.name = caseid

        # append to running DataFrames
        mirna = pd.concat([mirna, patient], axis=1)

        # use the case ID to query the clinical info TSV
        patient = clinical_table[clinical_table['cases.case_id'] == caseid]

        try:
            tumor_stage = patient[patient['diagnoses.diagnosis_is_primary_disease'] == 'true']['diagnoses.ajcc_pathologic_stage'].iloc[0]
            metastat = patient[patient['diagnoses.diagnosis_is_primary_disease'] == 'true']['diagnoses.ajcc_pathologic_m'].iloc[0]
            grade = patient[patient['diagnoses.diagnosis_is_primary_disease'] == 'true']['diagnoses.tumor_grade'].iloc[0]
        except IndexError: 
            tumor_stage = patient['diagnoses.ajcc_pathologic_stage'].iloc[0]
            metastat = patient['diagnoses.ajcc_pathologic_m'].iloc[0]
            grade = patient['diagnoses.tumor_grade'].iloc[0]

        stage_info.append(tumor_stage)
        invasive.append(metastat)
        grades.append(grade)

        gender = patient['demographic.gender'].iloc[0]
        encoded_gender.append(gender)
    
    # create dataframe to store case ID and encoded gender: 
    clinical_df = pd.DataFrame({'Gender': encoded_gender, 'Stage': stage_info, 'Metastasis': invasive, 'Grade': grades}, index=mirna.columns)
    
    # log transform 
    mirna = np.log2(mirna.T + 1)

    # align clinical info 
    clinical_df = clinical_df.reindex(mirna.index)

    if limit == True: 

        # concatenate clinical and methylation data 
        transposed = pd.concat([mirna, clinical_df], axis=1)

        # filter tumors by stage, metastasis, and grade
        transposed = transposed[
            (transposed['Stage'].isin(["Stage II", "Stage III", "Stage IV"])) &
            (transposed['Metastasis'].isin(["M0", "MX"])) &
            (transposed['Grade'] == "High Grade")
        ]

        # drop clinical info 
        mirna = transposed.drop(['Stage', 'Metastasis', 'Grade'], axis=1)
    
    else: 

        mirna = pd.concat([mirna, clinical_df['Gender']], axis=1)

    # encode gender to binary 
    mirna["Gender"] = mirna["Gender"].map({"male": 1, "female": 0})

    # drop genes with missing values 
    mirna = mirna.dropna(axis=1)

    return mirna

def get_tcga_protein(limit=False): 

    # create empty dataframes to hold gene counts and TPM-normalized counts
    protein = pd.DataFrame()

    # empty list to store target variable binary
    encoded_gender = []

    # load annotation data
    clinical_table = pd.read_csv(r"spring_2026/TCGA_BLCA_Clinical/clinical.tsv", sep="\t")
    with open(r'spring_2026/Metadata/tcga_blca_protein_metadata.json', 'r') as metadata_file:
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
        if "-01A-" in a:
            return 0
        if "-01B-" in a:
            return 1
        return 2

    meta_df["rank"] = meta_df["aliquot_id"].apply(aliquot_rank)
    meta_df = (meta_df.sort_values(["case_id", "rank", "aliquot_id"]).drop_duplicates(subset="case_id", keep="first"))

    # create lookup dictionary for JSON (file → case_id)
    metadata_dict = dict(zip(meta_df["file_name"], meta_df["case_id"]))

    # list containing stage of each 
    stage_info = []
    invasive = []
    grades = []

    tcga_files = get_paths(r"spring_2026/TCGA_BLCA_Protein")
    
    for file in tcga_files:
        patient = pd.read_csv(file, sep="\t", index_col="peptide_target", header=0)["protein_expression"]

        filename = str(Path(file).name)

        if filename not in metadata_dict:
            continue  # skip duplicate aliquots

        caseid = metadata_dict[filename]

        # rename columns to patient case ID
        patient.name = caseid

        # append to running DataFrames
        protein = pd.concat([protein, patient], axis=1)

        # use the case ID to query the clinical info TSV
        patient = clinical_table[clinical_table['cases.case_id'] == caseid]

        try:
            tumor_stage = patient[patient['diagnoses.diagnosis_is_primary_disease'] == 'true']['diagnoses.ajcc_pathologic_stage'].iloc[0]
            metastat = patient[patient['diagnoses.diagnosis_is_primary_disease'] == 'true']['diagnoses.ajcc_pathologic_m'].iloc[0]
            grade = patient[patient['diagnoses.diagnosis_is_primary_disease'] == 'true']['diagnoses.tumor_grade'].iloc[0]
        except IndexError: 
            tumor_stage = patient['diagnoses.ajcc_pathologic_stage'].iloc[0]
            metastat = patient['diagnoses.ajcc_pathologic_m'].iloc[0]
            grade = patient['diagnoses.tumor_grade'].iloc[0]

        stage_info.append(tumor_stage)
        invasive.append(metastat)
        grades.append(grade)

        gender = patient['demographic.gender'].iloc[0]
        encoded_gender.append(gender)
    
    # create dataframe to store case ID and encoded gender: 
    clinical_df = pd.DataFrame({'Gender': encoded_gender, 'Stage': stage_info, 'Metastasis': invasive, 'Grade': grades}, index=protein.columns)
    
    # impute missing values with KNNImputer 
    imputer = KNNImputer(n_neighbors=5)

    # Filter proteins by missing value threshold - keep those with less than 20% missing values
    protein_filtered = protein[protein.isna().mean(axis=1) <= 0.2]
   
    # use KNN imputation
    protein = pd.DataFrame(imputer.fit_transform(protein_filtered.T), index=protein_filtered.columns, columns=protein_filtered.index)

    # no log transform - there are negative RPPA values so they must be normalized to some extent 

    # align clinical info to M values 
    clinical_df = clinical_df.reindex(protein.index)

    if limit == True: 

        # concatenate clinical and methylation data 
        transposed = pd.concat([protein, clinical_df], axis=1)

        # filter tumors by stage, metastasis, and grade
        transposed = transposed[
            (transposed['Stage'].isin(["Stage II", "Stage III", "Stage IV"])) &
            (transposed['Metastasis'].isin(["M0", "MX"])) &
            (transposed['Grade'] == "High Grade")
        ]

        # drop clinical info 
        protein = transposed.drop(['Stage', 'Metastasis', 'Grade'], axis=1)
    
    else: 

        protein = pd.concat([protein, clinical_df['Gender']], axis=1)

    # encode gender to binary 
    protein["Gender"] = protein["Gender"].map({"male": 1, "female": 0})

    # drop genes with missing values 
    protein = protein.dropna(axis=1)

    return protein

def get_survival_data(clinical_path=Path("./TCGA_BLCA_Clinical/clinical.tsv")):
    
    # Load only the columns we need
    clinical_info = pd.read_csv(
        clinical_path,
        sep="\t",
        index_col="cases.case_id"
    )[
        ["demographic.vital_status", "demographic.days_to_death", "diagnoses.days_to_last_follow_up"]
    ]
    
    # Convert duration columns to numeric, coercing errors to NaN
    clinical_info["demographic.days_to_death"] = pd.to_numeric(
        clinical_info["demographic.days_to_death"], errors="coerce"
    )
    clinical_info["diagnoses.days_to_last_follow_up"] = pd.to_numeric(
        clinical_info["diagnoses.days_to_last_follow_up"], errors="coerce"
    )
    
    # Keep only one row per patient (first row that has at least one numeric value)
    clinical_info = clinical_info[clinical_info[["demographic.days_to_death", "diagnoses.days_to_last_follow_up"]].notna().any(axis=1)]
    clinical_info = clinical_info.groupby(clinical_info.index).first()
    
    # Encode vital status as binary 
    clinical_info["event"] = clinical_info["demographic.vital_status"].map({"Dead": 1, "Alive": 0})
    
    # Create duration column: days_to_death for dead, days_to_last_follow_up for alive
    clinical_info["duration"] = clinical_info["demographic.days_to_death"]
    clinical_info.loc[clinical_info["event"] == 0, "duration"] = clinical_info.loc[
        clinical_info["event"] == 0, 
        "diagnoses.days_to_last_follow_up"
    ]
    
    # Drop rows with missing event or duration
    clinical_info = clinical_info.dropna(subset=["event", "duration"])

    # Remove non-positive durations
    clinical_info = clinical_info[clinical_info["duration"] > 0]    
    
    # Drop old columns
    clinical_info = clinical_info.drop(
        ["demographic.vital_status", "demographic.days_to_death", "diagnoses.days_to_last_follow_up"], 
        axis=1
    )
    
    return clinical_info
       


def univariate_cox(df):

    # empty container for results
    results = []

    # iterate through all genes - exclude first two survival data columns
    print("Starting univariate Cox for all genes:")
    for gene in tqdm(df.columns.to_list()[2:]):

        # instantiate cox model
        uni_cox = CoxPHFitter(penalizer=0.01)

        # make it so df only includes survival data and gene of interest 
        one_gene = df[["duration", "event", gene]]

        # fit model using single gene, if it doesn't converge due to variance issues then skip 
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                uni_cox.fit(one_gene, duration_col="duration", event_col="event")

            # obtain summary and store in dictionary
            summary = uni_cox.summary.loc[gene]
            results.append({
                "variable": gene,
                "HR": summary["exp(coef)"],
                "CI_lower": summary["exp(coef) lower 95%"],
                "CI_upper": summary["exp(coef) upper 95%"],
                "p_value": summary["p"]
            })
        except: 
            continue
    
    univariate_printout = pd.DataFrame(results).sort_values(by="p_value", ascending=True)

    return univariate_printout
    

    

    






