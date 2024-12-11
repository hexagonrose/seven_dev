#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu3
#SBATCH --gres=gpu:1
##SBATCH --nodelist=n007
#SBATCH --job-name=mc_cli_test
#SBATCH --time=04-00:00              # Runtime limit: Day-HH:MM

# python3 compare_gnome_diffcsp.py
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

sevenn input.yaml