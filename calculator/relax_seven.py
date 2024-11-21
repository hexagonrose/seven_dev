import os
import sys
import time

# import pandas as pd
from ase.io import read, write
from ase.optimize import FIRE
from ase.constraints import UnitCellFilter
from sevenn.sevennet_calculator import SevenNetCalculator
import torch

from glob import glob

def write_log(log_path, msg, mode='a'):
    with open(log_path, mode) as f:
        f.write(msg + '\n')



if __name__ == '__main__':
    # materials you target
    material = sys.argv[1]
    run_type = sys.argv[2]

    # settings for relaxation
    log_path = os.getcwd()
    log_mat_path = os.path.join(log_path, material)

    start_time = time.time()
    # 2. relax structures using 7net-0
    seven_cal = SevenNetCalculator()
    end_time = time.time()
    print(f'Initiation done. Time : {end_time - start_time} s.')

    cif_paths = glob(os.path.join(log_mat_path, f'structure/{material}*.cif'))

    for file_path in cif_paths:
        mat_full_name = os.path.basename(file_path).split('.')[0]
        log_file = os.path.join(log_mat_path, f'{mat_full_name}_{run_type}.txt')
        write_log(log_file, f'7net-0 relaxation for {mat_full_name}.', mode='w')
        write_log(log_file, f'GPU: {torch.cuda.get_device_name(0)}')
        time_list = []
        for i in range(0, 21):
            # for idx in unique_id_list:
            atoms = read(file_path)
            write_log(log_file, f'Number of atoms: {len(atoms)}')
            relaxed_cif_path = os.path.join(log_mat_path, f'relaxed_{mat_full_name}.cif')

            atoms.calc = seven_cal

            # add unitcell filter
            ucf = UnitCellFilter(atoms)

            # relax the structure
            relax = FIRE(ucf)
            start_time = time.time()
            relax.run(fmax=0.05, steps=3000)
            # relax.run(steps=200)
            end_time = time.time()

            # saved relaxed structure
            write(relaxed_cif_path, atoms, format='cif')

            pot_energy = atoms.get_potential_energy()
            write_log(log_file, f'Energy: {pot_energy}')

            write_log(log_file, f'Relaxation done_{i}. Time : {end_time - start_time} s.')
            time_list.append(end_time - start_time)

        write_log(log_file, f'Average time : {sum(time_list[1:]) / len(time_list[1:])} s.')
