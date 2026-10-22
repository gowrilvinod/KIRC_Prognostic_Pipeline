#!/bin/bash -l
#SBATCH --job-name=cptac_sex_specific_select
#SBATCH --comment="4-method ensemble, 5-fold CV, 20 repeats, CPTAC expression, sex-specific cox"
#SBATCH --account=tumor
#SBATCH --partition=debug
#SBATCH --output=cptac_sex_spec_%x_%j_%a.out        
#SBATCH --error=cptac_sex_spec_%x_%j_%a.err
#SBATCH --mail-user=slack:@jrp6430
#SBATCH --mail-type=FAIL             
#SBATCH --time=04:00:00            
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          
#SBATCH --mem=16g
#SBATCH --array=0-19%5                 

source /home/gl7097/miniconda3/etc/profile.d/conda.sh
conda activate multi_omics_S26

# Execute CPTAC sex-specific discovery fold task for assigned random seed (runs both male and female cohorts)
python cptac_execution_sex_specific.py --rando $SLURM_ARRAY_TASK_ID --data_type expr
