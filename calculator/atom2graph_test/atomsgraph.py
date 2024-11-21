from ase.io import read
from ase.neighborlist import primitive_neighbor_list # ase
from matscipy.neighbours import neighbour_list # matscipy
# torch_geometric
# import torch
# from torch_geometric.data import Data
# from torch_geometric.transforms import RadiusGraph 

# # pymatgen
# from pymatgen.core import Structure
# from pymatgen.analysis.local_env import CutOffDictNN
# from pymatgen.analysis.graphs import StructureGraph

import sys
import numpy as np
# from mace.data.neighborhood import get_neighborhood
import matplotlib.pyplot as plt
import time

atoms_list = ['Li20Ge2P4S24.cif', 'Li20Ge2P4S24_112.cif', 'Li20Ge2P4S24_122.cif', 'Li20Ge2P4S24_222.cif']
for atom_path in atoms_list:
    atoms = read(atom_path)
    pos = atoms.get_positions()
    cell = np.array(atoms.get_cell())
    cutoff = 5.0
    pbc = atoms.get_pbc()

    ase_time_list = []
    matscipy_time_list = []

    for i in range(50):
        start = time.time()
        edge_src, edge_dst, edge_vec, shifts = primitive_neighbor_list(
                'ijDS', pbc, cell, pos, cutoff, self_interaction=False
        )
        end = time.time()
        ase_time_list.append(end - start)

        start = time.time()
        edge_src, edge_dst, edge_vec, shifts = neighbour_list(
            quantities="ijDS",
            pbc=pbc,
            cell=cell,
            positions=pos,
            cutoff=5.0,
            # self_interaction=True,  # we want edges from atom to itself in different periodic images
            # use_scaled_positions=False,  # positions are not scaled positions
        )
        end = time.time()
        matscipy_time_list.append(end - start)

    print(f'Atom: {atom_path}')
    print(f'ASE: {np.mean(ase_time_list)}')
    print(f'MatScipy: {np.mean(matscipy_time_list)}')
    print('')