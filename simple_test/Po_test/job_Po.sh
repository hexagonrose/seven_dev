#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
##SBATCH --gres=gpu:1
#SBATCH --nodelist=n008
#SBATCH --job-name=test_Po
#SBATCH --time=04-00:00              # Runtime limit: Day-HH:MM

# python3 compare_gnome_diffcsp.py
# export CUDA_DEVICE_ORDER=PCI_BUS_ID
# export CUDA_VISIBLE_DEVICES=1
python_path=/home/haekwan98/miniconda3/envs/seven_test/bin/python3

python test_Po.py
