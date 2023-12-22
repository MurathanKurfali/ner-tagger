#!/bin/bash
#SBATCH --gpus-per-node=V100:2
#SBATCH --nodes=1
#SBATCH -t 06:00:0
#SBATCH --output=log-%j.out
#SBATCH -A NAISS2023-22-1238
#SBATCH --exclude=alvis3-14

module load CUDA/12.3.0
module load Python/3.11.3-GCCcore-12.3.0
source rise_env/bin/activate
./run.sh $1 $2 $3
