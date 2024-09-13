#!/bin/bash
#SBATCH --job-name=trnActvSp___lctnPrd__lr1e4_15dBnoise

#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=sagnikmjr2002@gmail.com

## %j is the job id, %u is the user id
#SBATCH --output=slurm/out-%j-%x.out

## filename for job standard error output (stderr)
#SBATCH --error=slurm/error-%j-%x.err

#SBATCH --partition=learnaccel,learnfair
#SBATCH --nodes=1
#SBATCH --constraint='volta32gb'
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=80
#SBATCH --mem=480GB
#SBATCH --time=72:00:00
#SBATCH --exclude=learnfair1951,learnfair1961,learnfair1948,learnfair1950,learnfair1369,learnfair1120
## #SBATCH --nodelist=learnfair1963

echo "SLURM_JOBID: " $SLURM_JOBID
# 
module purge
module load anaconda3
module load cuda/10.1
source activate habitat_v0.1.4

ENV_PYTHON=/private/home/sagnikmjr2002/.conda/envs/habitat_v0.1.4/bin/python3
RUN_DIR=/checkpoint/sagnikmjr2002/code/ActiveAVSepMovingSource/runs/active_train/trnActvSp___lctnPrd__lr1e4_15dBnoise

srun -N1 --gres gpu:8 ${ENV_PYTHON} -u -m torch.distributed.launch --use_env --nproc_per_node 8 main.py --model-dir ${RUN_DIR} --run-type train --exp-config audio_separation/config/train/nearTarget_predLocation_newDistReward.yaml --val-cfg audio_separation/config/val/nearTarget.yaml RL.PPO.locationPredictor_ckpt runs/passivePretrain_locationPredictor/lr1e4_15dBnoise/best_val_ckpt.pth 	# L.PPO.locationPredictor_ckpt runs/passivePretrain_locationPredictor/{lr1e4_noNoise|lr1e4_15dBnoise|lr1e4_30dBnoise|lr1e4_45dBnoise|lr1e4_60dBnoise}/best_val_ckpt.pth
