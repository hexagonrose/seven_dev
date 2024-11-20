import os
import sys
import time
import argparse

# import pandas as pd
from pymatgen.core import Structure
from ase.io import read, write
from ase.optimize import FIRE
from ase.constraints import UnitCellFilter

# from sevenn.sevennet_calculator import SevenNetCalculator
from mace.calculators import mace_mp

# materials you target
material = sys.argv[1]

# settings for relaxation
log_path = f'/data2_1/haekwan98/sevennet_test/calculator/'
log_mat_path = os.path.join(log_path, material)

start_time = time.time()
# 2. relax structures using 7net-0
mace_cal = mace_mp(
    model='medium', dispersion=False, device='cuda', default_dtype='float64'
)
end_time = time.time()
print(f'Initiation done. Time : {end_time - start_time} s.')

# for idx in unique_id_list:
try:
    file_path = os.path.join(log_mat_path, f'{material}.cif')
    # file_path = f"/data2/team_csp/diffcsp/log/{material}/mpts_unique/relaxed_{material}_492.cif"
    atoms = read(file_path)
except:
    file_path = os.path.join(log_mat_path, f'POSCAR_{material}')
    atoms = read(file_path)
relaxed_cif_path = os.path.join(log_mat_path, f'relaxed_{material}.cif')

print(f'Relaxing {file_path}...')
# print(atoms)
atoms.calc = mace_cal

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
print(f'Relaxation done. Time : {end_time - start_time} s.')
