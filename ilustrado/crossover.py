# coding: utf-8
""" This file implements crossover functionality. """
from copy import deepcopy
import numpy as np
from matador.utils.chem_utils import get_stoich
from matador.utils.cell_utils import create_simple_supercell, standardize_doc_cell


def crossover(parents, method='random_slice', debug=False):
    """ Attempt to create a child structure from two parents structures.

    Parameters:

        parents (list(dict)) : list of two parent structures
        method (str)         : currently only 'random_slice'

    Returns:

        dict : newborn structure from parents.

    """

    if method == 'random_slice':
        _crossover = random_slice

    return _crossover(parents, debug=debug)


def random_slice(parent_seeds, standardize=True, supercell=True, shift=True, debug=False):
    """ Simple cut-and-splice crossover of two parents.

    The overall size of the child can vary between 0.5 and 1.5 the size of the
    parent structures. Both parent structures are cut and spliced along the
    same crystallographic axis.

    Parameters:

        parents (list(dict)) : parent structures to crossover,
        standardize (bool)   : use spglib to standardize parents pre-crossover,
        supercell (bool)     : make a random supercell to rescale parents,
        shift (bool)         : randomly shift atoms in parents to unbias.

    Returns:

        dict: newborn structure from parents.

    """
    parents = deepcopy(parent_seeds)
    child = dict()
    # child_size is a number between 0.5 and 2
    child_size = 0.5 + 1.5*np.random.rand()
    # cut_val is a number between 0.25*child_size and 0.75*child_size
    # the slice position of one parent in fractional coordinates
    # (the other is (child_size-cut_val))
    cut_val = child_size*(0.25 + (np.random.rand() / 2.0))

    if standardize:
        parents = [standardize_doc_cell(parent) for parent in parents]

    if supercell:
        # check ratio of num atoms in parents and grow the smaller one
        parent_extent_ratio = (parents[0]['cell_volume'] /
                               parents[1]['cell_volume'])
        if debug:
            print(parent_extent_ratio, parents[0]['cell_volume'],
                  'vs', parents[1]['cell_volume'])
        if parent_extent_ratio < 1:
            supercell_factor = int(round(1/parent_extent_ratio))
            supercell_target = 0
        elif parent_extent_ratio >= 1:
            supercell_factor = int(round(parent_extent_ratio))
            supercell_target = 1
        if debug:
            print(supercell_target, supercell_factor)
        supercell_vector = [1, 1, 1]
        if supercell_factor > 1:
            for ind in range(supercell_factor):
                min_lat_vec_abs = 1e10
                min_lat_vec_ind = -1
                for i in range(3):
                    lat_vec_abs = np.sum(np.asarray(parents[supercell_target]['lattice_cart'][i])**2)
                    if lat_vec_abs < min_lat_vec_abs:
                        min_lat_vec_abs = lat_vec_abs
                        min_lat_vec_ind = i
                supercell_vector[min_lat_vec_ind] += 1
        if debug:
            print('Making supercell of {} with {}'.format(parents[supercell_target]['source'][0],
                                                          supercell_vector))
        if supercell_vector != [1, 1, 1]:
            parents[supercell_target] = create_simple_supercell(parents[supercell_target],
                                                                supercell_vector,
                                                                standardize=False)
    child['positions_frac'] = []
    child['atom_types'] = []
    child['lattice_cart'] = (cut_val * np.asarray(parents[0]['lattice_cart']) +
                             (child_size-cut_val) * np.asarray(parents[1]['lattice_cart']))
    child['lattice_cart'] = child['lattice_cart'].tolist()

    # choose slice axis
    axis = np.random.randint(low=0, high=3)
    for ind, parent in enumerate(parents):
        if shift:
            # apply same random shift to all atoms in parents
            shift_vec = np.random.rand(3)
            for idx, _ in enumerate(parent['positions_frac']):
                for k in range(3):
                    parent['positions_frac'][idx][k] += shift_vec[k]
                    if parent['positions_frac'][idx][k] >= 1:
                        parent['positions_frac'][idx][k] -= 1
                    elif parent['positions_frac'][idx][k] < 0:
                        parent['positions_frac'][idx][k] += 1
        # slice parent
        for atom, pos in zip(parent['atom_types'], parent['positions_frac']):
            if ind == (pos[axis] <= cut_val):
                child['positions_frac'].append(pos)
                child['atom_types'].append(atom)
    # check child is sensible
    child['mutations'] = ['crossover']
    child['stoichiometry'] = get_stoich(child['atom_types'])
    child['num_atoms'] = len(child['atom_types'])
    return child
