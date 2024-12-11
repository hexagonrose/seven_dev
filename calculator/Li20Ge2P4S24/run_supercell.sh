#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --partition=loki3
##SBATCH --nodelist=n008
#SBATCH --job-name=benchmark
#SBATCH --time=04-00:00              # Runtime limit: Day-HH:MM

python make_supercell.py