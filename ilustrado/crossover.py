# coding: utf-8
""" This file implements crossover functionality. """
import numpy as np
from matador.utils.chem_utils import get_stoich


def crossover(parents, method='random_slice', debug=False):
    if method is 'random_slice':
        _crossover = random_slice
    elif method is 'periodic_cut':
        _crossover = periodic_cut

    return _crossover(parents)


def random_slice(parents):
    """ Random reflection, rotation and slicing
    a la XtalOpt.

    TO-DO:

    * random rotation
    """
    child = dict()
    # cut_val is a number between 0.25 and 0.75, the slice position in fractional coordinates
    cut_val = 0.25 + (np.random.rand() / 2.0)

    child['positions_frac'] = []
    child['atom_types'] = []
    child['lattice_cart'] = (cut_val * np.asarray(parents[0]['lattice_cart'])
                             * (1 - cut_val) * np.asarray(parents[0]['lattice_cart'])).tolist()
    axis = np.random.randint(low=0, high=3)
    for ind, parent in enumerate(parents):
        for atom, pos in zip(parent['atom_types'], parent['positions_frac']):
            if ind == (pos[axis] >= cut_val):
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
