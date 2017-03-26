# coding: utf-8
""" This file implements all notions of fitness. """
import numpy as np
from matador.utils.cursor_utils import get_array_from_cursor
from matador.utils.chem_utils import get_concentration, get_formation_energy


class FitnessCalculator(object):
    """ This class calculates the fitnesses of generations,
    by some global definition of generation-agnostic fitness.
    """
    def __init__(self, fitness_metric='dummy', hull=None, debug=False):
        self.testing = False
        self.debug = debug
        self.fitness_metric = fitness_metric
        if self.fitness_metric is 'hull':
            self._get_raw = self._get_hull_distance
            if hull is None:
                raise RuntimeError('Cannot calculate hull distanace without a hull!')
            self.hull = hull
        elif self.fitness_metric is 'dummy':
            self._get_raw = self._get_dummy_fitness
        elif self.fitness_metric is 'hull_test':
            self.testing = True
            self._get_raw = self._get_hull_distance
            self.hull = hull
            self.chempots = hull.match
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
            generation[ind]['formation_enthalpy_per_atom'] = get_formation_energy(self.chempots, populum)
            if self.debug:
                print(generation[ind]['concentration'], generation[ind]['formation_enthalpy_per_atom'])
        if self.testing:
            for ind, populum in enumerate(generation):
                generation[ind]['formation_enthalpy_per_atom'] = np.random.rand() - 0.5
        print('concs', np.shape(get_array_from_cursor(generation, 'concentration')))
        print('ef', np.shape(get_array_from_cursor(generation, 'formation_enthalpy_per_atom')))
        structures = np.hstack((get_array_from_cursor(generation, 'concentration'),
                                get_array_from_cursor(generation, 'formation_enthalpy_per_atom')))
        if self.debug:
            print(structures)
        hull_dist, _, _ = self.hull.get_hull_distances(structures)
        for ind, populum in enumerate(generation):
            generation[ind]['hull_distance'] = hull_dist[ind]
        return hull_dist

    def _get_dummy_fitness(self, generation):
        """ Generate dummy hull distances from -0.01 to 0.05. """
        return (0.05 * np.random.rand(len(generation)) - 0.01).tolist()
