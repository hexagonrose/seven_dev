# load test model
import os
import time
import warnings
warnings.filterwarnings("ignore")
from copy import deepcopy

from tqdm import tqdm
import torch
import ase.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from ase.constraints import UnitCellFilter
from ase.optimize import LBFGS
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

from sevenn.sevennet_calculator import SevenNetCalculator

def density_colored_scatter_plot(dft_energy, nnp_energy, dft_force, nnp_force, dft_stress, nnp_stress):
    modes = ['energy', 'force', 'stress']
    plt.figure(figsize=(18/2.54, 6/2.54))
    for num, (x, y) in enumerate(zip([dft_energy, dft_force, dft_stress], [nnp_energy, nnp_force, nnp_stress])):
        mode = modes[num]
        idx = np.random.choice(len(x), 1000) if len(x) > 1000 else list(range(len(x)))
        xsam = [x[i] for i in idx]
        ysam = [y[i] for i in idx]
        xy = np.vstack([x, y])
        xysam = np.vstack([xsam, ysam])
        zsam = gaussian_kde(xysam)

        z = zsam.pdf(xy)
        idx = z.argsort()

        x = [x[i] for i in idx]
        y = [y[i] for i in idx]
        z = [z[i] for i in idx]
        
        ax = plt.subplot(int(f'13{num+1}'))
        plt.scatter(x, y, c=z, s=4, cmap='plasma')

        mini = min(min(x), min(y))
        maxi = max(max(x), max(y))
        ran = (maxi-mini) / 20
        plt.plot([mini-ran, maxi+ran], [mini-ran, maxi+ran], color='grey', linestyle='dashed')
        plt.xlim(mini-ran, maxi+ran)
        plt.ylim(mini-ran, maxi+ran)

        plt.xlabel(f'DFT {mode} ({unit[mode]})')
        plt.ylabel(f'MLP {mode} ({unit[mode]})')
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()
    plt.savefig('parity_plot.png', dpi=300)


# codes for drawing EOS curve

def atom_oneshot(atoms, calc):
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()

    calc_results = {"energy": energy, "forces": forces, "stress": stress}
    calculator = SinglePointCalculator(atoms, **calc_results)
    atoms = calculator.get_atoms()

    return atoms

def atom_cell_relax(atoms, calc, logfile="-"):
    atoms.calc = calc
    cf = UnitCellFilter(atoms, hydrostatic_strain=True)
    opt = LBFGS(cf, logfile=logfile)
    opt.run(fmax=0.05, steps=1000)

    return atoms

def make_eos_structures(relaxed):
    relaxed_cell = relaxed.get_cell()
    relaxed_lat = relaxed_cell.lengths()[0]
    
    eos_structures = []
    for strain in np.linspace(-0.05, 0.05, 11):
        strained_lat = relaxed_lat * (1+strain)
        relaxed.set_cell([strained_lat]*3, scale_atoms=True)
        eos_structures.append(deepcopy(relaxed))

    return eos_structures

def get_eos_and_volume(eos_list):
    en_list = []
    vol_list = []
    for atoms in eos_list:
        en_list.append(atoms.get_potential_energy())
        vol_list.append(atoms.get_volume())
        
    rel_en_list = np.array(en_list) - min(en_list)

    return rel_en_list, vol_list


if __name__ == '__main__':
    # Let's test our model by predicting DFT MD trajectory
    # Instead of using other functions in SevenNet, we will use ASE calculator as an interface of our model
    working_dir = os.getcwd()
    data_path = '/data2_1/haekwan98/sevennet_tutorial/data'
    DFT_md_xyz = os.path.join(data_path, 'evaluation/test_md.extxyz')

    # initialize calculator from checkpoint.
    sevennet_calc = SevenNetCalculator(os.path.join(working_dir, 'checkpoint', 'checkpoint_best.pth'))

    # load DFT md trajectory
    traj = ase.io.read(DFT_md_xyz, index=':')

    dft_energy = []
    dft_forces = []
    dft_stress = []

    mlp_energy = []
    mlp_forces = []
    mlp_stress = []
    to_kBar = 1602.1766208

    start = time.time()
    for atoms in tqdm(traj):
        atoms.calc = sevennet_calc
        mlp_energy.append(atoms.get_potential_energy() / len(atoms))  # as per atom energy
        mlp_forces.append(atoms.get_forces())
        mlp_stress.extend(-atoms.get_stress() * to_kBar)  # eV/Angstrom^3 to kBar unit

        dft_energy.append(atoms.info['DFT_energy'] / len(atoms))
        dft_forces.append(atoms.arrays['DFT_forces'])
        dft_stress.append(-atoms.info['DFT_stress'] * to_kBar)
    end = time.time()
    print(f'Total inference time: {end-start} s')

    # flatten forces and stress for parity plot
    mlp_forces = np.concatenate([mf.reshape(-1,) for mf in mlp_forces])
    mlp_stress = np.concatenate([ms.reshape(-1,) for ms in mlp_stress])

    dft_forces = np.concatenate([df.reshape(-1,) for df in dft_forces])
    dft_stress = np.concatenate([ds.reshape(-1,) for ds in dft_stress])

    # draw a parity plot of energy / force / stress
    unit = {"energy": "eV/atom", "force": r"eV/$\rm{\AA}$", "stress": "kbar"}
   
    density_colored_scatter_plot(dft_energy, mlp_energy, dft_forces, mlp_forces, dft_stress, mlp_stress)
    

    # get relaxed structure
    os.makedirs('eos', exist_ok=True)
    atoms_list = read(os.path.join(data_path, 'evaluation/eos.extxyz'), ':')  # most stable structure idx
    most_stable_idx = np.argmin([at.get_potential_energy() for at in atoms_list])
    print(f"(DFT) potential_energy (eV/atom): {atoms_list[most_stable_idx].get_potential_energy() / len(atoms_list[0])}")
    atoms = atoms_list[most_stable_idx]

    log_path = './eos/seven_relax_log.txt'
    print("Relax with from-scratch potential...")
    relaxed = atom_cell_relax(atoms, sevennet_calc, log_path)
    print(f"(From scratch) potential_energy (eV/atom): {relaxed.get_potential_energy() / len(relaxed)}")

    # make EOS structures and calculate EOS curve
    eos_structures = make_eos_structures(relaxed)
    eos_oneshot = []
    for structure in eos_structures:
        eos_oneshot.append(atom_oneshot(structure, sevennet_calc))

    write('./eos/eos.extxyz', eos_oneshot)
    # draw EOS curve and compare with DFT
    dft_eos, dft_vol = get_eos_and_volume(read(os.path.join(data_path, 'evaluation/eos.extxyz'), ':'))
    mlp_eos, mlp_vol = get_eos_and_volume(read(os.path.join(working_dir, 'eos/eos.extxyz'), ':'))

    plt.figure(figsize=(10/2.54, 8/2.54))
    plt.plot(dft_vol, dft_eos, label='DFT')
    plt.plot(mlp_vol, mlp_eos, label='From scratch')

    plt.xlabel(r"Volume ($\rm{\AA}^3$)")
    plt.ylabel("Relative energy (eV)")

    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig('eos_curve.png', dpi=300)