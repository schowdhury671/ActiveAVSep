#!/bin/bash
#SBATCH --job-name=lr1e4_60dBnoise
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=sagnikmjr2002@gmail.com

## %j is the job id, %u is the user id
#SBATCH --output=slurm/out-%j-%x.out

## filename for job standard error output (stderr)
#SBATCH --error=slurm/error-%j-%x.err

#SBATCH --partition=eht,devaccel
#SBATCH --nodes=1
#SBATCH --constraint='volta32gb'
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40GB
#SBATCH --time=48:00:00
#SBATCH --exclude=learnfair1951,learnfair1961,learnfair1948,learnfair1950,learnfair1369,learnfair1120
## #SBATCH --nodelist=learnfair1963

echo "SLURM_JOBID: " $SLURM_JOBID

module purge
module load anaconda3
module load cuda/10.1
source activate habitat_v0.1.4 		# mae, lavila

# ENV_PYTHON=/private/home/sagnikmjr2002/.conda/envs/mae/bin/python3
# ENV_PYTHON=/private/home/sagnikmjr2002/.conda/envs/lavila/bin/python3
ENV_PYTHON=/private/home/sagnikmjr2002/.conda/envs/habitat_v0.1.4/bin/python3


srun -N1 --gres gpu:1 ${ENV_PYTHON} cnn-regression/cnn_regression_binaural.py
