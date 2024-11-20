#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu3
#SBATCH --gres=gpu:1
##SBATCH --nodelist=n008
#SBATCH --job-name=7net_ref
#SBATCH --time=04-00:00              # Runtime limit: Day-HH:MM
#SBATCH -o STDOUT.%N.%j.out          # STDOUT, %N : nodename, %j : JobID
#SBATCH -e STDERR.%N.%j.err          # STDERR, %N : nodename, %j : JobID

# python3 compare_gnome_diffcsp.py
# export CUDA_DEVICE_ORDER=PCI_BUS_ID
# export CUDA_VISIBLE_DEVICES=5
python_path=/home/haekwan98/miniconda3/envs/seven_test/bin/python3
code_path=/data2_1/haekwan98/sevennet_test/calculator

materials=(Li20Ge2P4S24)

for material in "${materials[@]}"; do
    # then relax unique structures
    relax_log=relax_"$material"_mace.log
    $python_path relax_mace.py $material > $material/$relax_log
done