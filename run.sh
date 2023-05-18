#!/bin/bash
#SBATCH --job-name=draw-layout-guidance
#SBATCH --output=/vulcanscratch/chuonghm/layout-guidance/logs/cora2_%A.out
#SBATCH --error=/vulcanscratch/chuonghm/layout-guidance/logs/cora2_%A.err
#SBATCH --time=10:00:00
#SBATCH --account=abhinav
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=16gb
#SBATCH --cpus-per-task=4


source /cfarhomes/chuonghm/.zshrc
conda activate layout-guidance
python inference.py