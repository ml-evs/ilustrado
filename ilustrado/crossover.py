# coding: utf-8
""" This file implements crossover functionality. """
import numpy as np
from matador.utils.chem_utils import get_stoich
from matador.utils.cell_utils import create_simple_supercell


def crossover(parents, method='random_slice', debug=False):
    if method is 'random_slice':
        _crossover = random_slice
    elif method is 'periodic_cut':
        _crossover = periodic_cut

    return _crossover(parents, debug=debug)


def random_slice(parents, debug=False):
    """ Random reflection, rotation and slicing
    a la XtalOpt.

    TO-DO:

        * random rotation

    """
    child = dict()
    # child_size is a number between 0.5 and 2
    child_size = 0.5 + 1.5*np.random.rand()
    # cut_val is a number between 0.25*child_size and 0.75*child_size
    # the slice position of one parent in fractional coordinates (the other is (child_size-cut_val))
    cut_val = child_size*(0.25 + (np.random.rand() / 2.0))

    child['positions_frac'] = []
    child['atom_types'] = []
    child['lattice_cart'] = (cut_val * np.asarray(parents[0]['lattice_cart'])
                             + (child_size-cut_val) * np.asarray(parents[1]['lattice_cart']))

    # check ratio of num atoms in parents and grow the smaller one
    parent_extent_ratio = parents[0]['num_atoms'] / parents[1]['num_atoms']
    if parent_extent_ratio < 1:
        supercell_factor = int(round(1/parent_extent_ratio))
        supercell_target = 0
    else:
        supercell_factor = int(round(parent_extent_ratio))
        supercell_target = 1
    supercell_vector = [1, 1, 1]
    if supercell_factor > 1:
        for i in range(supercell_factor):
            supercell_vector[np.random.randint(low=0, high=3)] += 1
    if supercell_vector != [1, 1, 1]:
        parents[supercell_target] = create_simple_supercell(parents[supercell_target],
                                                            supercell_vector,
                                                            standardize=False)
    # choose slice axis
    axis = np.random.randint(low=0, high=3)
    for ind, parent in enumerate(parents):
        # apply same random shift to all atoms in parents
        shift = np.random.rand(3)
        for ind, atom in enumerate(parent['positions_frac']):
            for k in range(3):
                parent['positions_frac'][ind][k] += shift[k]
                if parent['positions_frac'][ind][k] >= 1:
                    parent['positions_frac'][ind][k] -= 1
                elif parent['positions_frac'][ind][k] < 0:
                    parent['positions_frac'][ind][k] += 1
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


def periodic_cut(parents):
    """ Periodic cut a la CASTEP/Abraham & Probert. """
    child = dict()
    raise NotImplementedError
    return child
