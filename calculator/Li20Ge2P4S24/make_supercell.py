from ase.io import read, write
from ase.build import make_supercell

# Load the structure (e.g., from a CIF or POSCAR file)
structure = read('Li20Ge2P4S24.cif')  # Replace with your file

size_list = [[1, 1, 2], [1, 2, 2], [2, 2, 2]]

for size in size_list:
    i = size[0]
    j = size[1]
    k = size[2]
    # Define the supercell matrix (e.g., 2x2x2 supercell)
    supercell_matrix = [[i, 0, 0],
                        [0, j, 0],
                        [0, 0, k]]

    # Create the supercell
    supercell = make_supercell(structure, supercell_matrix)

    # Save the new supercell structure
    write(f'Li20Ge2P4S24_{i}{j}{k}.cif', supercell)  # Save as CIF or POSCAR
