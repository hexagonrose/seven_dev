import os
import sys
import time

# import pandas as pd
from ase.io import read, write
from ase import Atoms
from ase.optimize import FIRE
from ase.constraints import UnitCellFilter
from sevenn.sevennet_calculator import SevenNetCalculator
import torch

from glob import glob


if __name__ == '__main__':
    start_time = time.time()
    # 2. relax structures using 7net-0
    seven_cal = SevenNetCalculator()
    end_time = time.time()
    print(f'Initiation done. Time : {end_time - start_time} s.')
   
    # for idx in unique_id_list:
   # Define the lattice constant for Po
    lattice_constant = 3.35  # in Ångstroms

    # Create a simple cubic structure for Polonium
    symbol = 'Cu'
    atoms = Atoms(
        symbols=symbol,  # Element symbol for Polonium
        positions=[
            (0, 0, 0),  # Atom at the corner of the cube
        ],
        cell=[
            [lattice_constant, 0, 0],
            [0, lattice_constant, 0],
            [0, 0, lattice_constant],
        ],
        pbc=True,  # Periodic boundary conditions
    )
    # relaxed_cif_path = os.path.join(log_path, f'relaxed_{mat_full_name}.cif')
    write(f'POSCAR_{symbol}', atoms, 'vasp')
    atoms.calc = seven_cal

    # add unitcell filter
    ucf = UnitCellFilter(atoms)

    # relax the structure
    relax = FIRE(ucf)
    start_time = time.time()
    relax.run(fmax=0.02, steps=3000)
    # relax.run(steps=200)
    end_time = time.time()
    # saved relaxed structure
    # write(relaxed_cif_path, atoms, format='cif')
    pot_energy = atoms.get_potential_energy()
    print(f'Energy: {pot_energy}')
    

    