import sys
from typing import Any, Dict, List, Tuple, Union
import warnings

import torch
import numpy as np
import sevenn._keys as KEY
import sevenn.util as util
from sevenn.atom_graph_data import AtomGraphData
from sevenn.nn.sequential import AtomGraphSequential
from sevenn.train.dataload import unlabeled_atoms_to_graph
from ase.io import read
from sevenn.nn.edge_embedding import (
    BesselBasis,
    EdgeEmbedding,
    PolynomialCutoff,
    SphericalEncoding,
    XPLORCutoff,
)
from sevenn._const import DEFAULT_E3_EQUIVARIANT_MODEL_CONFIG, model_defaults
from sevenn.model_build import init_edge_embedding, build_E3_equivariant_model
from sevenn.sevennet_calculator import SevenNetCalculator
import time

def _patch_old_config(config: Dict[str, Any]):
    # Fixing my old mistakes
    if config[KEY.CUTOFF_FUNCTION][KEY.CUTOFF_FUNCTION_NAME] == 'XPLOR':
        config[KEY.CUTOFF_FUNCTION].pop('poly_cut_p_value', None)
    if KEY.TRAIN_DENOMINTAOR not in config:
        config[KEY.TRAIN_DENOMINTAOR] = config.pop('train_avg_num_neigh', False)
    _opt = config.pop('optimize_by_reduce', None)
    if _opt is False:
        raise ValueError(
            'This checkpoint(optimize_by_reduce: False) is no longer supported'
        )
    if KEY.CONV_DENOMINATOR not in config:
        config[KEY.CONV_DENOMINATOR] = 0.0
    if KEY._NORMALIZE_SPH not in config:
        config[KEY._NORMALIZE_SPH] = False
        # Warn this in the docs, not here for SevenNet-0 (22May2024)
    return config

def _map_old_model(old_model_state_dict):
    """
    For compatibility with old namings (before 'correct' branch merged 2404XX)
    Map old model's module names to new model's module names
    """
    _old_module_name_mapping = {
        'EdgeEmbedding': 'edge_embedding',
        'reducing nn input to hidden': 'reduce_input_to_hidden',
        'reducing nn hidden to energy': 'reduce_hidden_to_energy',
        'rescale atomic energy': 'rescale_atomic_energy',
    }
    for i in range(10):
        _old_module_name_mapping[f'{i} self connection intro'] = (
            f'{i}_self_connection_intro'
        )
        _old_module_name_mapping[f'{i} convolution'] = f'{i}_convolution'
        _old_module_name_mapping[f'{i} self interaction 2'] = (
            f'{i}_self_interaction_2'
        )
        _old_module_name_mapping[f'{i} equivariant gate'] = f'{i}_equivariant_gate'

    new_model_state_dict = {}
    for k, v in old_model_state_dict.items():
        key_name = k.split('.')[0]
        follower = '.'.join(k.split('.')[1:])
        if 'denumerator' in follower:
            follower = follower.replace('denumerator', 'denominator')
        if key_name in _old_module_name_mapping:
            new_key_name = _old_module_name_mapping[key_name] + '.' + follower
            new_model_state_dict[new_key_name] = v
        else:
            new_model_state_dict[k] = v
    return new_model_state_dict


if __name__ == '__main__':
    material = sys.argv[1]
    atoms = read(f'{material}.cif')

    # load weight manually
    # checkpoint = util.pretrained_name_to_path('7net-0')
    checkpoint = '/home/hexagonrose/SevenNet_exp/pretrained/m3g_c6/checkpoint_best.pth'
    check_point_loaded = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model_state_dict = check_point_loaded['model_state_dict']
    config = check_point_loaded['config']
    defaults = {**model_defaults(config)}
    config = _patch_old_config(config)

    for k, v in defaults.items():
        if k not in config:
            warnings.warn(f'{k} not in config, using default value {v}', UserWarning)
            config[k] = v
    for k, v in config.items():
        if isinstance(v, torch.Tensor):
            config[k] = v.cpu()

    # multi cutoff model
    multi_cutoff_list = [6, 6, 6, 6, 6]
    print(f'multi cutoff list: {multi_cutoff_list}')
    config['multi_cutoff'] = multi_cutoff_list
    model = build_E3_equivariant_model(config)
    missing, _ = model.load_state_dict(model_state_dict, strict=False)
    if len(missing) > 0:
        updated = _map_old_model(model_state_dict)
        missing, not_used = model.load_state_dict(updated, strict=False)
        if len(not_used) > 0:
            print(f'missing keys: {missing}')
            print(f'not used keys: {not_used}')
            for missing_key, not_used_key in zip(missing, not_used):
                if not_used_key in updated:
                    updated[missing_key] = updated[not_used_key]
                    print(f"Mapped {not_used_key} to {missing_key}")
        missing, not_used = model.load_state_dict(updated, strict=False)
        print("After remapping:")
        print(f'missing keys: {missing}')
        print(f'not used keys: {not_used}')

    # data preprocess
    data = AtomGraphData.from_numpy_dict(
        unlabeled_atoms_to_graph(atoms, 6.0)
    )
    data.to('cuda')

    # run model
    model.to('cuda')
    model.eval()
    model.set_is_batch_data(False)

    start = time.time()
    output = model(data)
    end = time.time()
    print(f'mc inference time: {end-start} s')

    mc_energy = output[KEY.PRED_TOTAL_ENERGY].detach().cpu().item()
    mc_forces = output[KEY.PRED_FORCE].detach().cpu().numpy()
    mc_stress = np.array(-output[KEY.PRED_STRESS].detach().cpu().numpy()[[0, 1, 2, 4, 5, 3]])
    print(f'multi cutoff energy: {mc_energy}')
    print(f'multi cutoff forces: {mc_forces}')
    print(f'multi cutoff stress: {mc_stress}')

    # just normal sevennet calc
    calc = SevenNetCalculator(model=checkpoint)
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()
    print(f'energy: {energy}')
    print(f'forces: {forces}')
    print(f'stress: {stress}')

    # compare
    print(f'energy equal?: {np.isclose(mc_energy, energy)}')
    print(f'forces equal?: {np.allclose(mc_forces, forces, atol=1e-6)}')
    print(f'stress equal?: {np.allclose(mc_stress, stress, atol=1e-6)}')