# coding: utf-8
""" This file implements all notions of fitness. """
import numpy as np
from matador.utils.cursor_utils import get_array_from_cursor
from matador.utils.chem_utils import get_concentration, get_formation_energy


class FitnessCalculator:
    """ This class calculates the fitnesses of generations,
    by some global definition of generation-agnostic fitness.

    Parameters:
        fitness_metric (str): either 'dummy', 'hull' or 'hull_test'.
        fitness_function (callable): function to operate on numpy array of raw fitness values,
        hull (QueryConvexHull): matador hull from which to calculate metastability,
        sandbagging (bool): whether or not to "sandbag" particular compositions, i.e. lower
            a structure's fitness based on the number of nearby phases

    """
    def __init__(self, fitness_metric='dummy', fitness_function=None, hull=None, sandbagging=False, debug=False):
        """ Initialise fitness calculator, if from hull then
        extract chemical potentials.

        """
        self.testing = False
        self.debug = debug
        self.fitness_metric = fitness_metric
        if self.fitness_metric == 'hull':
            self._get_raw = self._get_hull_distance
            if hull is None:
                raise RuntimeError('Cannot calculate hull distance without a hull!')
            self.hull = hull
            self.chempots = hull.chempot_cursor
        elif self.fitness_metric == 'dummy':
            self._get_raw = self._get_dummy_fitness
        elif self.fitness_metric == 'hull_test':
            self.testing = True
            self._get_raw = self._get_hull_distance
            self.hull = hull
            self.chempots = hull.chempot_cursor
        else:
            raise RuntimeError('No recognised fitness metric given.')

        self.sandbagging = False
        if sandbagging:
            self.sandbagging = True
            self.sandbag_multipliers = dict()

        if fitness_function is None:
            self.fitness_function = default_fitness_function
        else:
            self.fitness_function = fitness_function

    def evaluate(self, generation):
        """ Assign normalised fitnesses to an entire generation.
        Normalisation uses the logistic function such that

        `fitness = 1 - tanh(2*distance_from_hull)`,

        Parameters:
            generation (Generation/list): list/iterator over optimised structures,

        """

        raw = self._get_raw(generation)
        fitnesses = self.fitness_function(raw)

        for ind, _ in enumerate(generation):
            generation[ind]['raw_fitness'] = raw[ind]
            generation[ind]['fitness'] = fitnesses[ind]

        if self.sandbagging:
            self.update_sandbag_multipliers(generation)
            self.apply_sandbag_multipliers(generation)

    def update_sandbag_multipliers(self, generation, modifier=0.95):
        """ Assign composition penalty based on number of nearby structures.
        Updates fitness.sandbag_multipliers to a dictionary with chemical concentration as keys
        and values of fitness penalty.

        Parameters:
            generation (Generation): list of optimised structures.

        """
        for structure in generation:
            if tuple(structure['concentration']) in self.sandbag_multipliers:
                self.sandbag_multipliers[tuple(structure['concentration'])] *= modifier
            else:
                self.sandbag_multipliers[tuple(structure['concentration'])] = modifier

    def apply_sandbag_multipliers(self, generation, locality=0.05):
        """ Scale the generation's fitness by the sandbag modifier. This
        updates the 'fitness' key and the 'modifier' key (total scaling) of each document
        in the generation.

        Parameters:
            generation (Generation): list of optimised structures.

        Keyword Arguments:
            locality (float): tolerance by which two structures are "nearby"

        """
        for ind, structure in enumerate(generation):
            generation[ind]['modifier'] = 1
            for concentration in self.sandbag_multipliers:
                if np.sqrt(np.sum(np.abs(np.asarray(structure['concentration']) - np.asarray(list(concentration)))**2)) <= locality:
                    generation[ind]['modifier'] *= self.sandbag_multipliers[concentration]
            generation[ind]['fitness'] *= generation[ind]['modifier']

    def _get_hull_distance(self, generation):
        """ Assign distance from the hull from hull for generation,
        assigning it.

        Parameters:
            generation (Generation): list of optimised structures.

        Returns:
            hull_dist (list(float)): list of distances to the hull.

        """
        for ind, populum in enumerate(generation):
            generation[ind]['concentration'] = get_concentration(populum, self.hull.elements)
            generation[ind]['formation_enthalpy_per_atom'] = get_formation_energy(self.chempots,
                                                                                  populum)
            if self.debug:
                print(generation[ind]['concentration'],
                      generation[ind]['formation_enthalpy_per_atom'])
        if self.testing:
            for ind, populum in enumerate(generation):
                generation[ind]['formation_enthalpy_per_atom'] = np.random.rand() - 0.5
        structures = np.hstack((get_array_from_cursor(generation, 'concentration'),
                                get_array_from_cursor(generation, 'formation_enthalpy_per_atom')
                                .reshape(len(generation), 1)))
        if self.debug:
            print(structures)
        hull_dist = self.hull.get_hull_distances(structures, precompute=False)
        for ind, populum in enumerate(generation):
            generation[ind]['hull_distance'] = hull_dist[ind]
        return hull_dist

    def _get_dummy_fitness(self, generation):
        """ Generate dummy hull distances from -0.01 to 0.05.

        Parameters:
            generation (Generation): list of optimised structures.

        Returns:
            list(float): dummy list of hull distances.

        """
        return (0.05 * np.random.rand(len(generation)) - 0.01).tolist()


def default_fitness_function(raw, c=50, offset=0.075):
    """ Default fitness function: logistic function.

    Parameters:
        raw (ndarray): 1D array of raw fitness values.

    Returns:
        ndarray: 1D array of rescaled fitnesses.

    """
    fitnesses = 1 / (1 + np.exp(c*(raw - offset)))
    if isinstance(fitnesses, np.float64):
        fitnesses = min(1, fitnesses)
    else:
        fitnesses[fitnesses > 1.0] = 1.0
    return fitnesses
