#!/bin/bash -l
#SBATCH --job-name=bca_prog_selection_v2
#SBATCH --comment="4-method ensemble, 5-fold CV, 20 repeats, expression, sex-specific cox"
#SBATCH --account=tumor
#SBATCH --partition=debug
#SBATCH --output=%x_%j_%a.out        
#SBATCH --error=%x_%j_%a.err
#SBATCH --mail-user=slack:@jrp6430
#SBATCH --mail-type=FAIL             
#SBATCH --time=02:00:00            
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          
#SBATCH --mem=16g
#SBATCH --array=0-19%5                 

source /home/gl7097/miniconda3/etc/profile.d/conda.sh
conda activate multi_omics_S26
python /shared/rc/tumor/lgv_tumor/KIRC_Prognostic/execution.py --rando $SLURM_ARRAY_TASK_ID --data_type expr