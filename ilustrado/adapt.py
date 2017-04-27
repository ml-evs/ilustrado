# coding: utf-8
""" This file contains a wrapper for mutation and crossover. """
from .mutate import mutate
from .crossover import crossover
from matador.utils.cell_utils import cart2volume, frac2cart, cart2abc
from scipy.spatial.distance import cdist
from itertools import product
import numpy as np
import logging


def adapt(possible_parents, mutation_rate, crossover_rate,
          mutations=None, max_num_mutations=3, max_num_atoms=40, debug=False):
    """ Take a list of possible parents and adapt
    according to given mutation weightings.

    Input:

        possible_parents  : list of all breeding stock,
        mutation_rate     : rate of mutations relative to crossover,
        crossover_rate    : see above,
        mutations         : list of desired mutations to choose from (as strings),
        max_num_mutations : rand(1, this) mutations will be performed,
        max_num_atoms     : any structures with more than this many atoms will be filtered out.

    Returns:

        newborn           : the mutated/newborn structure.

    """
    total_rate = mutation_rate + crossover_rate
    if total_rate != 1.0:
        logging.debug('Total mutation rate not 1 ({}), rescaling...'
                      .format(total_rate))
    mutation_rate /= total_rate
    crossover_rate /= total_rate
    assert mutation_rate + crossover_rate == 1.0
    mutation_rand_seed = np.random.rand()
    # loop over *SAME* branch (i.e. crossover vs mutation) until valid cell is produced
    # with max attempts of 1000, at which point everything will crash
    valid_cell = False
    max_restarts = 1000
    num_iter = 0
    while not valid_cell and num_iter < max_restarts:
        # turn specified mutations string into corresponding functions
        if mutations is not None:
            _mutations = []
            from .mutate import nudge_positions, null_nudge_positions, permute_atoms
            from .mutate import random_strain, vacancy, voronoi_shuffle
            for mutation in mutations:
                if mutation is 'nudge_positions':
                    _mutations.append(nudge_positions)
                elif mutation is 'null_nudge_positions':
                    _mutations.append(null_nudge_positions)
                elif mutation is 'permute_atoms':
                    _mutations.append(permute_atoms)
                elif mutation is 'random_strain':
                    _mutations.append(random_strain)
                elif mutation is 'voronoi':
                    _mutations.append(voronoi_shuffle)
                elif mutation is 'vacancy':
                    _mutations.append(vacancy)
        else:
            _mutations = None

        # if random number is less than mutant rate, then mutate
        if mutation_rand_seed < mutation_rate:
            parent = np.random.choice(possible_parents)
            newborn = mutate(parent,
                             mutations=_mutations,
                             max_num_mutations=max_num_mutations,
                             debug=debug)
            parents = [parent]
        # otherwise, do crossover
        else:
            parents = np.random.choice(possible_parents, size=2, replace=False)
            newborn = crossover(parents, debug=debug)
        valid_cell = check_feasible(newborn, parents)
        num_iter += 1
    if num_iter == max_restarts:
        logging.warning('Max restarts reached in mutations, something has gone wrong...\
                         running with possibly unphysical cell')
    # set parents in newborn dict
    newborn['parents'] = []
    for parent in parents:
        for source in parent['source']:
            if source.endswith('.res') or source.endswith('.castep'):
                parent_source = source.split('/')[-1] \
                                      .replace('.res', '').replace('.castep', '')
        newborn['parents'].append(parent_source)
    return newborn


def check_feasible(mutant, parents):
    """ Check if a mutated/newly-born cell is "feasible".

    Here, feasible means:

        * number density within 25% of pre-mutation/birth level,
        * no overlapping atoms,
        * cell angles between 50 and 130 degrees,
        * fewer than max_num_atoms in the cell,
        * ensure number of atomic types is maintained.

    Input:

        mutant  : doc containing new structure.
        parents : list of doc(s) containing parent structures.

    Returns:

        feasibility : bool determined by points above.

    """

    # check number density first
    if 'cell_volume' not in mutant:
        mutant['cell_volume'] = cart2volume(mutant['lattice_cart'])
    number_density = mutant['num_atoms'] / mutant['cell_volume']
    parent_densities = []
    for ind, parent in enumerate(parents):
        if 'cell_volume' not in parent:
            parents[ind]['cell_volume'] = cart2volume(parent['lattice_cart'])
        parent_densities.append(parent['num_atoms'] / parent['cell_volume'])
    target_density = sum(parent_densities) / len(parent_densities)
    if number_density > 1.25 * target_density or number_density < 0.75 * target_density:
        logging.debug('Mutant with {} failed number density.'.format(', '.join(mutant['mutations'])))
        return False
    # now check element-agnostic minseps
    if 'positions_abs' not in mutant:
        mutant['positions_abs'] = frac2cart(mutant['lattice_cart'], mutant['positions_frac'])
    poscart = mutant['positions_abs']
    distances = np.array([])
    for prod in product(range(-1, 2), repeat=3):
        trans = np.zeros((3))
        for ind, multi in enumerate(prod):
            trans += np.asarray(mutant['lattice_cart'][ind]) * multi
        distances = np.append(distances, cdist(poscart+trans, poscart))
    distances = np.ma.masked_where(distances < 1e-12, distances)
    distances = distances.compressed()
    if np.min(distances) <= 1.4:
        logging.debug('Mutant with {} failed minsep check.'.format(', '.join(mutant['mutations'])))
        return False
    # check all cell angles are between 60 and 120.
    if 'lattice_abc' not in mutant:
        mutant['lattice_abc'] = cart2abc(mutant['lattice_cart'])
    for i in range(3):
        if mutant['lattice_abc'][1][i] < 50:
            logging.debug('Mutant with {} failed cell angle check.'.format(', '.join(mutant['mutations'])))
            return False
        elif mutant['lattice_abc'][1][i] > 130:
            logging.debug('Mutant with {} failed cell angle check.'.format(', '.join(mutant['mutations'])))
            return False
    # check that we haven't deleted/transmuted all atoms of a certain type
    if len(set(mutant['atom_types'])) < len(set(parents[0]['atom_types'])):
        return False
    return True
