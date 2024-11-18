#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --partition=gpul
##SBATCH --gres=gpu:1
##SBATCH --nodelist=n008
#SBATCH --job-name=batch_test
#SBATCH --time=04-00:00              # Runtime limit: Day-HH:MM

# export CUDA_DEVICE_ORDER=PCI_BUS_ID
# export CUDA_VISIBLE_DEVICES=1

material=Mg4Ta8O24

python batch_relaxation.py $material