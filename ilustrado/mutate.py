# coding: utf-8
""" This file implements all possible single mutant
mutations.

TO-DO: fix metadata of mutated structures, e.g. sources.

"""
import random
import numpy as np
from copy import deepcopy
from matador.utils.cell_utils import cart2abc
from traceback import print_exc
from sys import exit
from bson.json_util import dumps
from collections import defaultdict
from math import gcd


def _mutate(mutant, debug=False):
    """ Choose a random mutation and apply it. """

    debug = debug
    possible_mutations = [permute_atoms, random_strain, nudge_positions, vacancy]
    num_mutations = random.randint(1, 3)
    # num_mutations = 1
    if debug:
        print('num_mutations', num_mutations)
    # get random list of num_mutations mutators to apply
    mutations = []
    for i in range(num_mutations):
        mutations.append(possible_mutations[random.randint(0, len(possible_mutations)-1)])
    # apply successive mutations to mutant
    [mutator(mutant, debug=debug) for mutator in mutations]


def mutate(parent, debug=False):
    """ Wrap _mutate to check for null/invalid mutations. """

    mutant = deepcopy(parent)
    attempts = 0
    try:
        while parent == mutant:
            _mutate(mutant)
            attempts += 1
    except:
        print_exc()
        print('original:')
        print(dumps(parent, indent=2))
        print('mutant:')
        print(dumps(mutant, indent=2))
        exit()
    return mutant


def permute_atoms(mutant, debug=False):
    """ Swap the positions of random pairs of atoms.

    TO-DO: - should this favour similar atomic masses?
           - how many swaps should be done relative to num_atoms?
    """

    num_atoms = mutant['num_atoms']
    initial_atoms = deepcopy(mutant['atom_types'])

    # return non-mutated structure if only one atom type
    if len(mutant['stoichiometry']) == 1:
        return mutant

    # choose atoms to swap
    idx_a = random.randint(0, num_atoms-1)
    idx_b = idx_a
    while mutant['atom_types'][idx_a] == mutant['atom_types'][idx_b]:
        idx_b = random.randint(0, num_atoms-1)

    # swap atoms
    if debug:
        print(idx_b, mutant['atom_types'][idx_b], idx_a, mutant['atom_types'][idx_a])
    temp = mutant['atom_types'][idx_b]
    mutant['atom_types'][idx_b] = mutant['atom_types'][idx_a]
    mutant['atom_types'][idx_a] = temp

    if debug:
        print(list(zip(range(0, num_atoms), initial_atoms, mutant['atom_types'])))


def vacancy(mutant, debug=False):
    """ Remove a random atom from the structure.

    TO-DO: farm out stoich calc to matador.
    """

    vacancy_idx = random.randint(0, mutant['num_atoms']-1)
    if debug:
        print('Removing atom {} of type {} from cell.'.format(vacancy_idx, mutant['atom_types'][vacancy_idx]))
    del mutant['atom_types'][vacancy_idx]
    del mutant['positions_frac'][vacancy_idx]
    try:
        del mutant['positions_cart'][vacancy_idx]
    except:
        pass
    mutant['num_atoms'] = len(mutant['atom_types'])
    # calculate stoichiometry
    mutant['stoichiometry'] = defaultdict(float)
    for atom in mutant['atom_types']:
        if atom not in mutant['stoichiometry']:
            mutant['stoichiometry'][atom] = 0
        mutant['stoichiometry'][atom] += 1
    gcd_val = 0
    for atom in mutant['atom_types']:
        if gcd_val == 0:
            gcd_val = mutant['stoichiometry'][atom]
        else:
            gcd_val = gcd(mutant['stoichiometry'][atom], gcd_val)
    # convert stoichiometry to tuple for fryan
    temp_stoich = []
    try:
        for key, value in mutant['stoichiometry'].items():
            if float(value)/gcd_val % 1 != 0:
                temp_stoich.append([key, float(value)/gcd_val])
            else:
                temp_stoich.append([key, value/gcd_val])
    except AttributeError:
        for key, value in mutant['stoichiometry'].iteritems():
            if float(value)/gcd_val % 1 != 0:
                temp_stoich.append([key, float(value)/gcd_val])
            else:
                temp_stoich.append([key, value/gcd_val])
    mutant['stoichiometry'] = temp_stoich


def random_strain(mutant, debug=False):
    """ Apply random strain tensor to unit cell from 6
    \epsilon_i components with values between -1 and 1.
    The cell is then scaled to the parent mutant's volume.
    """

    def generate_cell_transform_matrix():
        strain_components = 2*np.random.rand(6)-1
        cell_transform_matrix = np.eye(3)
        for i in range(3):
            cell_transform_matrix[i][i] += strain_components[i]
        cell_transform_matrix[0][1] += strain_components[3] / 2
        cell_transform_matrix[1][0] += strain_components[3] / 2
        cell_transform_matrix[2][0] += strain_components[4] / 2
        cell_transform_matrix[0][2] += strain_components[4] / 2
        cell_transform_matrix[1][2] += strain_components[5] / 2
        cell_transform_matrix[2][1] += strain_components[5] / 2
        return cell_transform_matrix

    valid = False
    while not valid:
        cell_transform_matrix = generate_cell_transform_matrix()
        # only accept matrices with positive determinant, then scale that det to 1
        if np.linalg.det(cell_transform_matrix) > 0:
            cell_transform_matrix /= pow(np.linalg.det(cell_transform_matrix), 1/3)
            valid = True
        if valid:
            # assert symmetry
            assert np.allclose(cell_transform_matrix.T, cell_transform_matrix)
            assert np.linalg.det(cell_transform_matrix) > 0
            if debug:
                print(cell_transform_matrix)
            # exclude all strains that take us to sub-60 and sup-120 cell angles
            new_lattice_abc = cart2abc(np.matmul(cell_transform_matrix, np.array(mutant['lattice_cart'])))
            for angle in new_lattice_abc[1]:
                if angle > 120 or angle < 60:
                    valid = False
            # also exclude all cells where at least one lattice vector is less than 2 A
            mean_lat_vec = np.mean(new_lattice_abc[0])
            for length in new_lattice_abc[0]:
                if length < mean_lat_vec / 2:
                    valid = False

    mutant['lattice_cart'] = np.matmul(cell_transform_matrix, np.array(mutant['lattice_cart'])).tolist()
    mutant['lattice_abc'] = cart2abc(mutant['lattice_cart'])
    if debug:
        print('lattice_abc:', mutant['lattice_abc'])
        print('lattice_cart:', mutant['lattice_cart'])
        print('cell_transform_matrix:', cell_transform_matrix.tolist())


def nudge_positions(mutant, amplitude=0.1, debug=False):
    """ Apply Gaussian noise to all atomic positions. """

    new_positions_frac = np.array(mutant['positions_frac'])
    for ind, atom in enumerate(mutant['positions_frac']):
        # generate random noise vector between -amplitude and amplitude
        new_positions_frac[ind] += amplitude * np.random.rand(3) - amplitude
        for i in range(3):
            if new_positions_frac[ind][i] > 1:
                new_positions_frac[ind][i] -= 1
            elif new_positions_frac[ind][i] < 0:
                new_positions_frac[ind][i] += 1
    mutant['positions_frac'] = new_positions_frac.tolist()
