#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu3
#SBATCH --gres=gpu:1
##SBATCH --nodelist=n008
#SBATCH --job-name=benchmark_LGPS
#SBATCH --time=04-00:00              # Runtime limit: Day-HH:MM

# python3 compare_gnome_diffcsp.py
# export CUDA_DEVICE_ORDER=PCI_BUS_ID
# export CUDA_VISIBLE_DEVICES=1
python_path=/home/haekwan98/miniconda3/envs/seven_test/bin/python3
code_path=/data2_1/haekwan98/sevennet_test/calculator

# materials=(Mg4Ta8O24 Li20Ge2P4S24 Li56V8N32)
materials=(Li20Ge2P4S24)
run_type='ase'

for material in "${materials[@]}"; do
    # then relax unique structures
    relax_log="$material"_seven_"$run_type"_avg.log
    $python_path relax_seven.py $material $run_type > $material/$relax_log
done