#!/bin/bash -l
#SBATCH --job-name=cptac_sex_agnostic_select
#SBATCH --comment="4-method ensemble, 5-fold CV, 20 repeats, CPTAC expression, sex-agnostic cox"
#SBATCH --account=tumor
#SBATCH --partition=debug
#SBATCH --output=cptac_select_%x_%j_%a.out        
#SBATCH --error=cptac_select_%x_%j_%a.err
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

# Execute CPTAC sex-agnostic discovery fold task for assigned random seed
python cptac_execution_sex_agnostic.py --rando $SLURM_ARRAY_TASK_ID --data_type expr
