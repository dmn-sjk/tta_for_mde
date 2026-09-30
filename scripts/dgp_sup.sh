#!/bin/bash

#SBATCH -p gpu20
#SBATCH --gres gpu:2
#SBATCH -o /path/to/logs/slurm-%A_%a.out
#SBATCH -t 3-0

cmd="python train.py --model_name dgp_sup --dataset dgp --data_path PATH_TO_ddad.json --width 640 --height 384 --batch_size 4"

echo $(date)
echo $cmd

$cmd
