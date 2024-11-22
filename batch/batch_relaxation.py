import os
import sys
import time
from typing import Callable, List, Optional
from glob import glob

from ase.io import read
from tqdm import tqdm
import torch
import torch.multiprocessing as mp

from sevenn.atom_graph_data import AtomGraphData
from sevenn.nn.sequential import AtomGraphSequential
from sevenn.train.dataload import unlabeled_atoms_to_graph
import sevenn._keys as KEY
import sevenn.util as util
from sevenn.sevennet_calculator import SevenNetCalculator

# for test
from torch_geometric.loader import DataLoader # test
from torch_geometric.data import Batch # batch test

def unlabeled_graph_build(
    atoms_list: List,
    cutoff: float,
    num_cores: int = 1,
    transfer_info: bool = True,
    y_from_calc: bool = False,
) -> List[AtomGraphData]:
    """
    parallel version of graph_build
    build graph from atoms_list and return list of AtomGraphData
    Args:
        atoms_list (List): list of ASE atoms
        cutoff (float): cutoff radius of graph
        num_cores (int): number of cores to use
        transfer_info (bool): if True, copy info from atoms to graph,
                              defaults to True
        y_from_calc (bool): Get reference y labels from calculator, defaults to False
    Returns:
        List[AtomGraphData]: list of AtomGraphData
    """
    serial = num_cores == 1
    inputs = [(atoms, cutoff) for atoms in atoms_list]

    if not serial:
        pool = mp.Pool(num_cores)
        graph_list = pool.starmap(
            unlabeled_atoms_to_graph,
            tqdm(inputs, total=len(atoms_list), desc=f'graph_build ({num_cores})'),
        )
        pool.close()
        pool.join()
    else:
        graph_list = [
            unlabeled_atoms_to_graph(*input_)
            for input_ in tqdm(inputs, desc='graph_build (1)')
        ]

    graph_list = [AtomGraphData.from_numpy_dict(g) for g in graph_list]

    return graph_list

material = sys.argv[1]
working_dir = os.getcwd()
device='cuda'
# Read the structure paths
structure_paths = glob(os.path.join(working_dir, material, "*.cif"))
structures = [] # List of ASE Atoms objects
start = time.time()
for i, structure_path in enumerate(structure_paths):
    structure = read(structure_path)
    structures.append(structure)
    
end = time.time()
print(f"Reading structures took {end-start:.2f} seconds.")

# Convert the structures to graphs
start = time.time()
# data = AtomGraphData.from_numpy_dict(
#     unlabeled_atoms_to_graph(structures[0], 5.0)
# )
# print(data)
# print(data_numpy)
data = unlabeled_graph_build(atoms_list=structures, cutoff=5.0, num_cores=5)
end = time.time()
print(f"Converting structures to graphs took {end-start:.2f} seconds.")

# import model
start = time.time()
model_path = '/data2/shared_data/pretrained_experimental/7net_chgTot/checkpoint_best.pth'

model_loaded, config = util.model_from_checkpoint(model_path)
model_loaded.to(device)
end = time.time()
print(f"Loading model took {end-start:.2f} seconds.")

# Predict the energy, forces, and stress
# before running the model, we need to set some parameters
start = time.time()
type_map = config[KEY.TYPE_MAP]
data_loader = DataLoader(data, batch_size=4, shuffle=True)
end = time.time()
print(f"Setting up the data loader took {end-start:.2f} seconds.")

print('no batch')
print(data)
print('batch')
# batch = Batch.from_data_list(data)
batch = Batch.from_data_list(data[0:1])
print(batch)

start = time.time()
batch = batch.to(device)
output = model_loaded(batch)
print(output)
energy = output[KEY.PRED_TOTAL_ENERGY].detach().cpu().numpy()
forces = output[KEY.PRED_FORCE].detach().cpu().numpy()
stress = -output[KEY.PRED_STRESS].detach().cpu().numpy()
print(energy)
print(forces)
print(stress)
end = time.time()
print(f"Predicting energy, forces, and stress took {end-start:.2f} seconds.")





# for data_ in data:
#     data_[KEY.NODE_FEATURE] = torch.tensor(
#         [type_map[z.item()] for z in data_[KEY.NODE_FEATURE]],
#         dtype=torch.int64,
#         device='cuda',
#     )
#     data_[KEY.POS].requires_grad_(True)  # backward compatibility
#     data_[KEY.EDGE_VEC].requires_grad_(True)  # backward compatibility
#     data_ = data_.to_dict()
#     del data_['data_info']
# print(data)

# output = model_loaded(data)
# print(output)