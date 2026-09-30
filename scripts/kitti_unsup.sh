#!/bin/bash

#SBATCH -p gpu20
#SBATCH --gres gpu:2
#SBATCH -o /path/to/logs/slurm-%A_%a.out
#SBATCH -t 5-0

cmd="python train.py --model_name kitti_unsup --monodepth --width 1216 --height 352 --batch_size 6"

echo $(date)
echo $cmd

$cmd
