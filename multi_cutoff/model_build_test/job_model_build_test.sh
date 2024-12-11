#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
##SBATCH --gres=gpu:1
#SBATCH --nodelist=n007
#SBATCH --job-name=test_code
#SBATCH --time=04-00:00              # Runtime limit: Day-HH:MM

# python3 compare_gnome_diffcsp.py
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
python_path=/home/haekwan98/miniconda3/envs/seven_test/bin/python3

# material=Mg4Ta8O24
material=Li20Ge2P4S24_222
python run_model_build_test.py $material
