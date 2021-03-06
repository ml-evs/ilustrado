# coding: utf-8
""" This file contains a wrapper for mutation and crossover. """

from itertools import product
from traceback import print_exc
import copy

import numpy as np
import periodictable
from scipy.spatial.distance import cdist

from matador.utils.cell_utils import cart2volume, frac2cart, cart2abc
from matador.utils.chem_utils import get_root_source

from .mutate import mutate
from .crossover import crossover
from .util import strip_useless, LOG


def adapt(
    possible_parents,
    mutation_rate,
    crossover_rate,
    mutations=None,
    max_num_mutations=3,
    max_num_atoms=40,
    structure_filter=None,
    minsep_dict=None,
    debug=False,
):
    """ Take a list of possible parents and randomly adapt
    according to given mutation weightings.

    Parameters:
        possible_parents (list(dict)) : list of all breeding stock,
        mutation_rate (float): rate of mutations relative to crossover,
        crossover_rate (float): see `mutation_rate`.

    Keyword Arguments:
        mutations (list(str)): list of desired mutations to choose from (as strings),
        max_num_mutations (int): rand(1, this) mutations will be performed,
        max_num_atoms (int): any structures with more than this many atoms will be filtered out.
        structure_filter (callable(dict)): custom filter to pass to check_feasible.
        minsep_dict (dict): dictionary containing element-specific minimum separations, e.g.
            `{('K', 'K'): 2.5, ('K', 'P'): 2.0}`.

    Returns:
        dict: the mutated/newborn structure.

    """
    total_rate = mutation_rate + crossover_rate
    if total_rate != 1.0:
        LOG.debug("Total mutation rate not 1 ({}), rescaling...".format(total_rate))
    mutation_rate /= total_rate
    crossover_rate /= total_rate
    assert mutation_rate + crossover_rate == 1.0
    mutation_rand_seed = np.random.rand()

    # turn specified mutations string into corresponding functions
    if mutations is not None:
        _mutations = []
        from .mutate import nudge_positions, null_nudge_positions, permute_atoms
        from .mutate import random_strain, vacancy, voronoi_shuffle, transmute_atoms

        for mutation in mutations:
            if mutation == "nudge_positions":
                _mutations.append(nudge_positions)
            elif mutation == "null_nudge_positions":
                _mutations.append(null_nudge_positions)
            elif mutation == "permute_atoms":
                _mutations.append(permute_atoms)
            elif mutation == "random_strain":
                _mutations.append(random_strain)
            elif mutation == "voronoi":
                _mutations.append(voronoi_shuffle)
            elif mutation == "vacancy":
                _mutations.append(vacancy)
            elif mutation == "transmute_atoms":
                _mutations.append(transmute_atoms)
    else:
        _mutations = None

    # loop over *SAME* branch (i.e. crossover vs mutation) until valid cell is produced
    # with max attempts of 1000, at which point it will continue with a terrible cell
    valid_cell = False
    max_restarts = 1000
    num_iter = 0
    while not valid_cell and num_iter < max_restarts:
        # if random number is less than mutant rate, then mutate
        if mutation_rand_seed < mutation_rate:
            parent = strip_useless(np.random.choice(possible_parents), to_run=True)
            try:
                newborn = mutate(
                    parent,
                    mutations=_mutations,
                    max_num_mutations=max_num_mutations,
                    debug=debug,
                )
                parents = [parent]
                valid_cell = check_feasible(
                    newborn,
                    parents,
                    max_num_atoms,
                    structure_filter=structure_filter,
                    minsep_dict=minsep_dict,
                )
            # this will be raised if the mutation fails for a good reason
            except RuntimeError:
                valid_cell = False
            except Exception as oops:
                if debug:
                    print_exc()
                LOG.warning("Mutation failed with error {}".format(oops))
                valid_cell = False
        # otherwise, do crossover
        else:
            if len(possible_parents) > 2:
                parents = [
                    strip_useless(parent, to_run=True)
                    for parent in np.random.choice(
                        possible_parents, size=2, replace=False
                    )
                ]
            elif len(possible_parents) == 2:
                parents = copy.deepcopy(possible_parents)
            elif len(possible_parents) == 1:
                parents = 2 * [copy.deepcopy(possible_parents[0])]
                LOG.warning("Only one possible parent: performing self-crossover...")

            try:
                newborn = crossover(parents, debug=debug)
                valid_cell = check_feasible(
                    newborn,
                    parents,
                    max_num_atoms,
                    structure_filter=structure_filter,
                    minsep_dict=minsep_dict,
                )
            except RuntimeError:
                valid_cell = False
            except Exception as oops:
                if debug:
                    print_exc()
                LOG.warning("Crossover failed with error {}".format(oops))
                valid_cell = False
        num_iter += 1

    LOG.debug("Initialised newborn after {} trials".format(num_iter))
    if num_iter == max_restarts:
        LOG.warning(
            "Max restarts reached in mutations, something has gone wrong... "
            "running with possibly unphysical cell"
        )
        newborn = adapt(
            possible_parents,
            mutation_rate,
            crossover_rate,
            mutations=mutations,
            max_num_mutations=max_num_mutations,
            max_num_atoms=max_num_atoms,
            minsep_dict=minsep_dict,
            debug=debug,
        )
    # set parents in newborn dict
    if "parents" not in newborn:
        newborn["parents"] = []
        for parent in parents:
            parent_source = get_root_source(parent["source"])
            newborn["parents"].append(parent_source)
    return newborn


def check_feasible(
    mutant, parents, max_num_atoms, structure_filter=None, minsep_dict=None, debug=False
):
    """ Check if a mutated/newly-born cell is "feasible".
    Here, feasible means:

      * number density within 25% of pre-mutation/birth level,
      * no overlapping atoms, parameterised by minsep_dict,
      * cell angles between 50 and 130 degrees,
      * fewer than max_num_atoms in the cell,
      * ensure number of atomic types is maintained,
      * any custom filter is obeyed.

    Parameters:
        mutant (dict): matador doc containing new structure.
        parents (list(dict)): list of doc(s) containing parent structures.
        max_num_atoms (int): any structures with more than this many atoms will be filtered out.

    Keyword Arguments:
        structure_filter (callable): any function that takes a matador document and returns True or False.
        minsep_dict (dict): dictionary containing element-specific minimum separations, e.g.
            {('K', 'K'): 2.5, ('K', 'P'): 2.0}.

    Returns:
        bool: True if structure is feasible, else False.

    """
    # first check the structure filter
    if structure_filter is not None and not structure_filter(mutant):
        message = "Mutant with {} failed to pass the custom filter.".format(
            ", ".join(mutant["mutations"])
        )
        LOG.debug(message)
        if debug:
            print(message)
        return False
    # check number of atoms
    if "num_atoms" not in mutant or mutant["num_atoms"] != len(mutant["atom_types"]):
        mutant["num_atoms"] = len(mutant["atom_types"])
    if mutant["num_atoms"] > max_num_atoms:
        message = "Mutant with {} contained too many atoms ({} vs {}).".format(
            ", ".join(mutant["mutations"]), mutant["num_atoms"], max_num_atoms
        )
        LOG.debug(message)
        if debug:
            print(message)
        return False
    # check number density
    if "cell_volume" not in mutant:
        mutant["cell_volume"] = cart2volume(mutant["lattice_cart"])
    number_density = mutant["num_atoms"] / mutant["cell_volume"]
    parent_densities = []
    for ind, parent in enumerate(parents):
        if "cell_volume" not in parent:
            parents[ind]["cell_volume"] = cart2volume(parent["lattice_cart"])
        parent_densities.append(parent["num_atoms"] / parent["cell_volume"])
    target_density = sum(parent_densities) / len(parent_densities)
    if number_density > 1.5 * target_density or number_density < 0.5 * target_density:
        message = "Mutant with {} failed number density.".format(
            ", ".join(mutant["mutations"])
        )
        LOG.debug(message)
        if debug:
            print(message)
        return False

    # now check element-agnostic minseps
    if not minseps_feasible(mutant, minsep_dict=minsep_dict, debug=debug):
        return False

    # check all cell angles are between 60 and 120.
    if "lattice_abc" not in mutant:
        mutant["lattice_abc"] = cart2abc(mutant["lattice_cart"])

    if min(mutant["lattice_abc"][1]) < 30:
        message = "Mutant with {} failed cell angle check.".format(
            ", ".join(mutant["mutations"])
        )
        LOG.debug(message)
        if debug:
            print(message)
        return False

    if max(mutant["lattice_abc"][1]) > 120:
        message = "Mutant with {} failed cell angle check.".format(
            ", ".join(mutant["mutations"])
        )
        LOG.debug(message)
        if debug:
            print(message)
        return False

    # check that we haven't deleted/transmuted all atoms of a certain type
    if len(set(mutant["atom_types"])) < len(set(parents[0]["atom_types"])):
        message = "Mutant with {} transmutation error.".format(
            ", ".join(mutant["mutations"])
        )
        LOG.debug(message)
        if debug:
            print(message)
        return False
    return True


def minseps_feasible(mutant, minsep_dict=None, debug=False):
    """ Check if minimum separations between species of atom are satisfied by mutant.

    Parameters:
        mutant (dict): trial mutated structure
        minsep_dict (dict): dictionary containing element-specific minimum separations, e.g.
            {('K', 'K'): 2.5, ('K', 'P'): 2.0}.

    Returns:
        bool: True if minseps are greater than desired value else False.

    """
    elems = set(mutant["atom_types"])
    elem_pairs = set()
    for elem in elems:
        for _elem in elems:
            elem_key = tuple(sorted([elem, _elem]))
            elem_pairs.add(elem_key)

    if minsep_dict is None:
        minsep_dict = dict()
    else:
        marked_for_del = []
        for key in minsep_dict:
            if tuple(sorted(key)) != tuple(key):
                minsep_dict[tuple(sorted(key))] = minsep_dict[key]
                marked_for_del.append(key)
        for key in marked_for_del:
            del minsep_dict[key]

    # use 0.5 * average covalent radii (NOT just average covalent radius) as rough default minsep guess
    for elem_key in elem_pairs:
        if elem_key not in minsep_dict:
            minsep_dict[elem_key] = 0.5 * sum(
                [
                    periodictable.elements.symbol(elem).covalent_radius
                    for elem in elem_key
                ]
            )

    if "positions_abs" not in mutant:
        mutant["positions_abs"] = frac2cart(
            mutant["lattice_cart"], mutant["positions_frac"]
        )
    poscart = mutant["positions_abs"]

    for prod in product(range(-1, 2), repeat=3):
        trans = np.zeros((3))
        for ind, multi in enumerate(prod):
            trans += np.asarray(mutant["lattice_cart"][ind]) * multi
        distances = cdist(poscart + trans, poscart)
        distances = np.ma.masked_where(distances < 1e-12, distances)
        for i, dists in enumerate(distances):
            for j, dist in enumerate(dists):
                min_dist = minsep_dict[
                    tuple(sorted([mutant["atom_types"][i], mutant["atom_types"][j]]))
                ]
                if dist < min_dist:
                    message = "Mutant with {} failed minsep check.".format(
                        ", ".join(mutant["mutations"])
                    )
                    LOG.debug(message)
                    return False
    return True
