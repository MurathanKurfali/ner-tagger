#!/bin/bash
#SBATCH --gpus-per-node=A40:4
#SBATCH --nodes=1
#SBATCH -t 24:00:00
#SBATCH --output=log-%j.out
#SBATCH -A NAISS2023-22-1238

module load CUDA/12.3.0
module load Python/3.11.3-GCCcore-12.3.0
source rise_env/bin/activate
e.g ./run.sh --system-type $1 --bert-name $2
