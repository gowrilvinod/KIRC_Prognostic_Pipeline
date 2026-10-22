#!/bin/bash -l
#SBATCH --job-name=cptac_sex_spec_eval
#SBATCH --comment="Freezing, internal OOF CV, and external validation for CPTAC sex-specific models"
#SBATCH --account=tumor
#SBATCH --partition=debug
#SBATCH --output=cptac_eval_%x_%j.out
#SBATCH --error=cptac_eval_%x_%j.err
#SBATCH --mail-user=slack:@jrp6430
#SBATCH --mail-type=END,FAIL
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16g

source /home/gl7097/miniconda3/etc/profile.d/conda.sh
conda activate multi_omics_S26

echo "=== STEP 1: FREEZING CPTAC MALE AND FEMALE MODELS ==="
python cptac_model_freezer_sex_specific.py

echo "=== STEP 2: CPTAC INTERNAL 5-FOLD OOF VALIDATION ==="
python cptac_internal_validation_sex_specific.py

echo "=== STEP 3: EXTERNAL VALIDATION (TCGA, GSE22541, ICGC RECA-EU) ==="
python cptac_eval_external_sex_specific.py

echo "=== CPTAC SEX-SPECIFIC EVALUATION PIPELINE COMPLETE ==="
