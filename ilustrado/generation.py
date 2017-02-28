""" This file implements the Generation class which
is used to store each generation of structures, and to
evaulate their fitness.
"""

# matador modules
from matador.utils.chem_utils import get_formula_from_stoich
# external libraries
# standard library
import json


class Generation(object):
    """ Stores each generation of structures. """

    def __init__(self, populace=[], fitness_calculator=None, num_survivors=5):

        self.populace = populace
        self._num_survivors = num_survivors
        self._fitness_calculator = fitness_calculator

    def __len__(self):
        return len(self.populace)

    def __str__(self):
        gen_string = 80*'=' + '\n'
        gen_string += 'Number of members: {}\n'.format(len(self.populace))
        gen_string += 'Number of survivors: {}\n'.format(len(self.bourgeoisie))
        for populum in self.populace:
            gen_string += '{:10} {:5f}\n'.format(get_formula_from_stoich(populum['stoichiometry']), populum['fitness'])
        gen_string += 80*'=' + '\n'
        return gen_string

    def __getitem__(self, key):
        return self.populace[int(key)]

    def __iter__(self):
        return iter(self.populace)

    def dump(self, gen_idx):
        with open('gen_{}.json'.format(gen_idx), 'w') as f:
            json.dump(self.populace, f)

    def birth(self, populum):
        self.populace.append(populum)

    def rank(self):
        self._fitness_calculator.evaluate(self)

    @property
    def bourgeoisie(self):
        return sorted(self.populace, key=lambda member: member['fitness'], reverse=True)[:self._num_survivors]

    @property
    def fitnesses(self):
        return [populum['fitness'] for populum in self.populace]

    @property
    def raw_fitnesses(self):
        return [populum['raw_fitness'] for populum in self.populace]

    @property
    def most_fit(self):
        try:
            assert self.bourgeoisie[0]['fitness'] == max(self.fitnesses)
        except AssertionError:
            print(self.bourgeoisie)
            print('{} != {}'.format(self.bourgeoisie[0]['fitness'], max(self.fitnesses)))
            raise AssertionError

        return self.bourgeoisie[-1]

    @property
    def average_pleb_fitness(self):
        population = len(self.populace)
        average_fitness = 0
        for populum in self.populace:
            average_fitness += populum['fitness'] / population
        return average_fitness

    @property
    def average_bourgeois_fitness(self):
        population = len(self.bourgeoisie)
        average_fitness = 0
        for populum in self.bourgeoisie:
            average_fitness += populum['fitness'] / population
        return average_fitness

