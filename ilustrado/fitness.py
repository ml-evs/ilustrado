# coding: utf-8
""" This file implements all notions of fitness. """
import numpy as np
from matador.utils.cursor_utils import get_array_from_cursor
from matador.utils.chem_utils import get_concentration, get_formation_energy


class FitnessCalculator(object):
    """ This class calculates the fitnesses of generations,
    by some global definition of generation-agnostic fitness.

    Input:

        | fitness_metric   : str, either 'dummy', 'hull' or 'hull_test'.
        | fitness_function : fn, function to operate on numpy array of raw fitness values,
        | hull             : QueryConvexHull, matador hull from which to calculate metastability,

    """
    def __init__(self, fitness_metric='dummy', fitness_function=None, hull=None, debug=False):
        """ Initialise fitness calculator, if from hull then
        extract chemical potentials.
        """
        self.testing = False
        self.debug = debug
        self.fitness_metric = fitness_metric
        if self.fitness_metric is 'hull':
            self._get_raw = self._get_hull_distance
            if hull is None:
                raise RuntimeError('Cannot calculate hull distanace without a hull!')
            self.hull = hull
            self.chempots = hull.match
        elif self.fitness_metric is 'dummy':
            self._get_raw = self._get_dummy_fitness
        elif self.fitness_metric is 'hull_test':
            self.testing = True
            self._get_raw = self._get_hull_distance
            self.hull = hull
            self.chempots = hull.match
        else:
            raise RuntimeError('No recognised fitness metric given.')
        if fitness_function is None:
            self.fitness_function = self._default_fitness_function
        else:
            self.fitness_function = fitness_function

    def _default_fitness_function(self, raw):
        """ Default fitness function: logistic function.

        Input:

            | raw: ndarray, array of raw fitness values.
        """
        c = 100
        offset = 0.05
        fitnesses = 1 / (1 + np.exp(c*(raw - offset)))
        fitnesses[fitnesses > 1.0] = 1.0
        return fitnesses

    def evaluate(self, generation):
        """ Assign normalised fitnesses to an entire generation.
        Normalisation uses the logistic function such that

        | fitness = 1 - tanh(2*distance_from_hull),

        Input:

            | generation: list(dict), list of optimised structures,


        """

        raw = self._get_raw(generation)
        fitnesses = self.fitness_function(raw)

        for ind, populum in enumerate(generation):
            generation[ind]['fitness'] = fitnesses[ind]
            generation[ind]['raw_fitness'] = raw[ind]

    def _get_hull_distance(self, generation):
        """ Assign distance from the hull from hull for generation,
        assigning it.

        Input:

            | generation: list(dict), list of optimised structures.

        Returns:

            | hull_dist : list(float), list of distances to the hull.

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
        hull_dist, _, _ = self.hull.get_hull_distances(structures)
        for ind, populum in enumerate(generation):
            generation[ind]['hull_distance'] = hull_dist[ind]
        return hull_dist

    def _get_dummy_fitness(self, generation):
        """ Generate dummy hull distances from -0.01 to 0.05.

        Input:

            | generation: list(dict), list of optimised structures.

        Returns:

            | list(float), dummy list of hull distances.

        """
        return (0.05 * np.random.rand(len(generation)) - 0.01).tolist()
