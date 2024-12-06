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


if __name__ == '__main__':
    # materials you target
    material = sys.argv[1]
    
    # settings for relaxation
    log_path = os.getcwd()
    file_path = os.path.join(log_path, f'{material}.cif')

    start_time = time.time()
    # 2. relax structures using 7net-0
    seven_cal = SevenNetCalculator()
    end_time = time.time()
    print(f'Initiation done. Time : {end_time - start_time} s.')
   
    # for idx in unique_id_list:
    atoms = read(file_path)
    # relaxed_cif_path = os.path.join(log_path, f'relaxed_{mat_full_name}.cif')

    atoms.calc = seven_cal

    # add unitcell filter
    ucf = UnitCellFilter(atoms)

    # relax the structure
    relax = FIRE(ucf)
    start_time = time.time()
    # relax.run(fmax=0.05, steps=3000)
    # relax.run(steps=200)
    end_time = time.time()
    # saved relaxed structure
    # write(relaxed_cif_path, atoms, format='cif')
    pot_energy = atoms.get_potential_energy()
    print(f'Energy: {pot_energy}')
    

    