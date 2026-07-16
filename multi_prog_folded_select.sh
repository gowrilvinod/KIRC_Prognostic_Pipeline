#!/bin/bash -l
#SBATCH --job-name=bca_prog_selection_v2
#SBATCH --comment="4-method ensemble, 5-fold CV, 20 repeats, expression, sex-specific cox"
#SBATCH --account=tumor
#SBATCH --partition=tier3
#SBATCH --output=%x_%j_%a.out        
#SBATCH --error=%x_%j_%a.err
#SBATCH --mail-user=slack:@jrp6430
#SBATCH --mail-type=FAIL             
#SBATCH --time=3-00:00:00            
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18          
#SBATCH --mem=32g
#SBATCH --array=0-19%5                 

source ~/conda/etc/profile.d/conda.sh
conda activate multi_prog
python ~/spring_2026/execution.py --rando $SLURM_ARRAY_TASK_ID --data_type expr