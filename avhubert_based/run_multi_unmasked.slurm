#!/bin/bash
#SBATCH -J masking_avhubert
#SBATCH -p p-A100
#SBATCH -N 1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1

source /home/chenshutang/.bashrc
conda activate avhubert
cd /mntnfs/lee_data1/chenshutang/robust_SS/avhubert_based
bash run_multi_unmasked.sh
