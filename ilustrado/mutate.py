# coding: utf-8
""" This file implements all possible single mutant
mutations.
"""
import numpy as np
import logging
from traceback import print_exc
from copy import deepcopy
from matador.utils.cell_utils import cart2abc
from matador.utils.chem_utils import get_stoich
from matador.voronoi_interface import get_voronoi_points
from sklearn.cluster import KMeans


def mutate(parent, mutations=None, max_num_mutations=2, debug=False):
    """ Wrap _mutate to check for null/invalid mutations.

    Input:

        | parent: dict, parent structure to mutate,

    Args:

        | mutations        : list(fn), list of possible mutation functions to apply,
        | max_num_mutations: int, maximum number of mutations to apply.

    """
    mutant = deepcopy(parent)
    attempts = 0
    max_attempts = 100
    while parent == mutant and attempts < max_attempts:
        try:
            _mutate(mutant,
                    mutations=mutations,
                    max_num_mutations=max_num_mutations,
                    debug=debug)
        except:
            print_exc()
            logging.warning('Issue with {}'.format(mutations))
        attempts += 1
    if attempts == max_attempts:
        logging.warning('Failed to mutate with {}'.format(mutations))
        return parent

    return mutant


def _mutate(mutant, mutations=None, max_num_mutations=2, debug=False):
    """ Choose a random number of mutations and apply them.

    Input:

        | mutant: dict, structure to mutate in-place,

    Args:

        | mutations        : list(fn), list of possible mutation functions,
        | max_num_mutations: int, maximum number of mutations to apply.

    """
    if mutations is None:
        possible_mutations = [permute_atoms,
                              random_strain,
                              nudge_positions,
                              vacancy,
                              voronoi_shuffle]
    else:
        possible_mutations = mutations
    if max_num_mutations == 1:
        num_mutations = 1
    else:
        num_mutations = np.random.randint(1, high=max_num_mutations+1)
    if debug:
        print('num_mutations', num_mutations)
    # get random list of num_mutations mutators to apply
    mutations = []
    mutant['mutations'] = []
    for i in range(num_mutations):
        mutation = np.random.choice(possible_mutations)
        mutations.append(mutation)
        mutant['mutations'].append(str(mutation).split(' ')[1])
    # apply successive mutations to mutant
    [mutator(mutant, debug=debug) for mutator in mutations]


def permute_atoms(mutant, debug=False):
    """ Swap the positions of random pairs of atoms.

    Input:

        | mutant: dict, structure to mutate in-place.

    """
    num_atoms = mutant['num_atoms']
    initial_atoms = deepcopy(mutant['atom_types'])

    # choose atoms to swap
    valid = True
    idx_a = np.random.randint(0, num_atoms-1)
    idx_b = np.random.randint(0, num_atoms-1)
    while not valid:
        if mutant['atom_types'][idx_a] != mutant['atom_types'][idx_b]:
            valid = True
        idx_b = np.random.randint(0, num_atoms-1)

    # swap atoms
    if debug:
        print(idx_b, mutant['atom_types'][idx_b], idx_a, mutant['atom_types'][idx_a])
    temp = deepcopy(mutant['atom_types'][idx_b])
    mutant['atom_types'][idx_b] = deepcopy(mutant['atom_types'][idx_a])
    mutant['atom_types'][idx_a] = deepcopy(temp)

    if debug:
        print(list(zip(range(0, num_atoms), initial_atoms, mutant['atom_types'])))


def vacancy(mutant, debug=False):
    """ Remove a random atom from the structure.

    Input:

        | mutant: dict, structure to mutate in-place.

    """
    vacancy_idx = np.random.randint(0, mutant['num_atoms']-1)
    if debug:
        print('Removing atom {} of type {} from cell.'.format(vacancy_idx,
                                                              mutant['atom_types'][vacancy_idx]))
    del mutant['atom_types'][vacancy_idx]
    del mutant['positions_frac'][vacancy_idx]
    try:
        del mutant['positions_cart'][vacancy_idx]
    except:
        pass
    mutant['num_atoms'] = len(mutant['atom_types'])
    # calculate stoichiometry
    mutant['stoichiometry'] = get_stoich(mutant['atom_types'])


def voronoi_shuffle(mutant, element_to_remove=None, preserve_stoich=False, debug=False):
    """ Remove all atoms of type element, then perform Voronoi analysis
    on the remaining sublattice. Cluster the nodes with KMeans, then
    repopulate the clustered Voronoi nodes with atoms of the removed element.

    Input:

        | mutant: dict, structure to mutate in-place.

    Args:

        | element_to_remove: str, symbol of element to remove,
        | preserve_stoich  : bool, whether to always reinsert the same number
                             of atoms.

    """
    if element_to_remove is None:
        element_to_remove = np.random.choice(list(set(mutant['atom_types'])))
    mutant['atom_types'], mutant['positions_frac'] = \
        zip(*[(atom, pos) for (atom, pos) in zip(mutant['atom_types'],
                                                 mutant['positions_frac']) if atom != element_to_remove])
    num_removed = mutant['num_atoms'] - len(mutant['atom_types'])

    if debug:
        print('Removed {} atoms of type {}'.format(num_removed, element_to_remove))

    mutant['num_atoms'] = len(mutant['atom_types'])
    mutant['atom_types'], mutant['positions_frac'] = list(mutant['atom_types']), list(mutant['positions_frac'])
    try:
        mutant['voronoi_nodes'] = get_voronoi_points(mutant)
    except:
        if debug:
            print_exc()
        raise RuntimeError('Voronoi code failed')

    if mutant['voronoi_nodes'] is False:
        raise RuntimeError('Voronoi code failed')

    if debug:
        print('Computed {} Voronoi nodes'.format(len(mutant['voronoi_nodes'])))

    if preserve_stoich:
        num_to_put_back = num_removed
    else:
        std_dev = int(np.sqrt(num_removed))
        try:
            num_to_put_back = np.random.randint(low=max(num_removed-std_dev, 1), high=min(num_removed+std_dev, len(mutant['voronoi_nodes'])))
        except:
            num_to_put_back = len(mutant['voronoi_nodes'])

    if debug:
        print('Going to insert {} atoms of type {}'.format(num_to_put_back, element_to_remove))

    k_means = KMeans(n_clusters=num_to_put_back, precompute_distances=True)
    k_means.fit(mutant['voronoi_nodes'])
    mutant['voronoi_nodes'] = k_means.cluster_centers_.tolist()
    for node in mutant['voronoi_nodes']:
        mutant['atom_types'].append(element_to_remove)
        mutant['positions_frac'].append(node)

    if debug:
        print('Previously {} atoms in cell'.format(mutant['num_atoms']))

    mutant['num_atoms'] = len(mutant['atom_types'])
    mutant['stoichiometry'] = get_stoich(mutant['atom_types'])
    if debug:
        print('Now {} atoms in cell'.format(mutant['num_atoms']))


def random_strain(mutant, debug=False):
    """ Apply random strain tensor to unit cell from 6 \epsilon_i components
    with values between -1 and 1. The cell is then scaled to the parent's volume.

    Input:

        | mutant: dict, structure to mutate in-place.

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
            new_lattice_abc = cart2abc(np.matmul(cell_transform_matrix,
                                       np.array(mutant['lattice_cart'])))
            for angle in new_lattice_abc[1]:
                if angle > 120 or angle < 60:
                    valid = False
            # also exclude all cells where at least one lattice vector is less than 2 A
            mean_lat_vec = np.mean(new_lattice_abc[0])
            for length in new_lattice_abc[0]:
                if length < mean_lat_vec / 2:
                    valid = False

    mutant['lattice_cart'] = np.matmul(cell_transform_matrix,
                                       np.array(mutant['lattice_cart'])).tolist()
    mutant['lattice_abc'] = cart2abc(mutant['lattice_cart'])
    if debug:
        print('lattice_abc:', mutant['lattice_abc'])
        print('lattice_cart:', mutant['lattice_cart'])
        print('cell_transform_matrix:', cell_transform_matrix.tolist())


def nudge_positions(mutant, amplitude=0.5, debug=False):
    """ Apply Gaussian noise to all atomic positions.

    Input:

        | mutant: dict, structure to mutate in-place.

    Args:

        | amplitude: float, amplitude of random noise in Angstroms.

    """
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


def null_nudge_positions(mutant, debug=False):
    """ Apply minimal Gaussian noise to all atomic positions, mostly
    for testing purposes.

    Input:

        | mutant: dict, structure to mutate in-place.

    """
    nudge_positions(mutant, amplitude=0.001)
