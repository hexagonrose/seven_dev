#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
##SBATCH --gres=gpu:1
#SBATCH --nodelist=n008
#SBATCH --job-name=benchmark_LGPS
#SBATCH --time=04-00:00             


python atomsgraph.py