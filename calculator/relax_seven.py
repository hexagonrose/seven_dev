import os
import sys
import time

# import pandas as pd
from ase.io import read, write
from ase.optimize import FIRE
from ase.constraints import UnitCellFilter
from sevenn.sevennet_calculator import SevenNetCalculator
import torch
print(torch.cuda.get_device_name(0))

# materials you target
material = sys.argv[1]
run_type = sys.argv[2]

# settings for relaxation
log_path = f'/data2_1/haekwan98/sevennet_test/calculator/'
log_mat_path = os.path.join(log_path, material)

start_time = time.time()
# 2. relax structures using 7net-0
seven_cal = SevenNetCalculator()
end_time = time.time()
print(f'Initiation done. Time : {end_time - start_time} s.')

time_list = []
for i in range(0, 100):
    # for idx in unique_id_list:
    start_time = time.time()
    try:
        file_path = os.path.join(log_mat_path, f'{material}.cif')
        atoms = read(file_path)
    except:
        file_path = os.path.join(log_mat_path, f'POSCAR_{material}')
        atoms = read(file_path)
    relaxed_cif_path = os.path.join(log_mat_path, f'relaxed_{material}.cif')

    print(f'Relaxing {file_path}...')
    atoms.calc = seven_cal

    # add unitcell filter
    ucf = UnitCellFilter(atoms)

    # relax the structure
    relax = FIRE(ucf)
    relax.run(fmax=0.02, steps=3000)

    # saved relaxed structure
    write(relaxed_cif_path, atoms, format='cif')

    pot_energy = atoms.get_potential_energy()
    print(f'energy: {pot_energy}')

    end_time = time.time()
    print(f'Relaxation done_{i}. Time : {end_time - start_time} s.')
    time_list.append(end_time - start_time)

print(f'Average time : {sum(time_list) / len(time_list)} s.')
