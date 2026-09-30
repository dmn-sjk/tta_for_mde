#!/bin/bash

#SBATCH -p gpu20
#SBATCH --gres gpu:2
#SBATCH -o /path/to/logs/slurm-%A_%a.out
#SBATCH -t 3-0

cmd="python train.py --model_name dgp_unsup --monodepth --dataset dgp --data_path PATH_TO_ddad.json --width 960 --height 608 --batch_size 4"

echo $(date)
echo $cmd

$cmd
