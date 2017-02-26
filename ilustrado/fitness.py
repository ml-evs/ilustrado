# coding: utf-8
""" This file implements all notions of fitness. """
import numpy as np
from matador.cursor_utils import get_array_from_cursor
from matador.chem import get_concentration


class FitnessCalculator(object):
    """ This class calculates the fitnesses of generations,
    by some global definition of generation-agnostic fitness.
    """
    def __init__(self, fitness_metric='dummy', hull=None):
        self.fitness_metric = fitness_metric
        if self.fitness_metric is 'hull_distance':
            self._get_raw = self._get_hull_distance
            if hull is None:
                raise RuntimeError('Cannot calculate hull distanace without a hull!')
            self.hull = hull
        elif self.fitness_metric is 'dummy':
            self._get_raw = self._get_dummy_fitness
        else:
            raise RuntimeError('No recognised fitness metric given.')

    def evaluate(self, generation):
        raw = self._get_raw(generation)
        fitnesses = 1 - np.tanh(raw)
        fitnesses[fitnesses > 1.0] = 1.0

        for ind, populum in enumerate(generation):
            generation[ind]['fitness'] = fitnesses[ind]
            generation[ind]['raw_fitness'] = raw[ind]

    def _get_hull_distance(self, generation):
        for ind, populum in enumerate(generation):
            generation[ind]['concentration'] = get_concentration(populum, self.hull.elements)
        structures = np.hstack((get_array_from_cursor(generation, 'concentration'),
                                get_array_from_cursor(generation, 'formation_enthalpy_per_atom')
                                .reshape(len(generation), 1)))
        self.hull.get_hull_distances(structures)
        raise NotImplementedError

    def _get_dummy_fitness(self, generation):
        """ Generate dummy hull distances from -0.01 to 0.05. """
        return (0.05 * np.random.rand(len(generation)) - 0.01).tolist()
