""" This file implements the Generation class which
is used to store each generation of structures, and to
evaulate their fitness.
"""

# matador modules
from matador.utils.chem_utils import get_formula_from_stoich
# external libraries
# standard library
import json
from traceback import print_exc


class Generation():
    """ Stores each generation of structures. """

    def __init__(self, run_hash, generation_idx, num_survivors,
                 populace=None, fitness_calculator=None):

        self.populace = []
        if populace is not None:
            self.populace = populace
        self._num_survivors = num_survivors
        self._fitness_calculator = fitness_calculator
        self.run_hash = run_hash
        self.generation_idx = generation_idx

    def __len__(self):
        return len(self.populace)

    def __str__(self):
        gen_string = '\nCompleted generation {}:\n'.format(self.generation_idx)
        gen_string += 'Number of members: {}\n'.format(len(self.populace))
        gen_string += 'Number of survivors: {}\n'.format(len(self.bourgeoisie))
        gen_string += 74*'─' + '\n'
        gen_string += ('{:^25} {:^10} {:^10} {:^25}\n'
                       .format('ID', 'Formula', 'Fitness', 'Hull distance (eV/atom)'))
        gen_string += 74*'─' + '\n'
        for populum in self.populace:
            gen_string += ('{:^25} {:^10} {: ^10.5f} {:^25.5f}\n'
                           .format(populum['source'][0].split('/')[-1]
                                   .replace('.res', '').replace('.castep', ''),
                                   get_formula_from_stoich(populum['stoichiometry']),
                                   populum['fitness'], populum['raw_fitness']))
        gen_string += '\n'
        return gen_string

    def __getitem__(self, key):
        return self.populace[int(key)]

    def __iter__(self):
        return iter(self.populace)

    def dump(self, gen_suffix):
        with open('{}-gen{}.json'.format(self.run_hash, gen_suffix), 'w') as f:
            json.dump(self.populace, f, sort_keys=False, indent=2)

    def birth(self, populum):
        self.populace.append(populum)

    def rank(self):
        self._fitness_calculator.evaluate(self)

    @property
    def bourgeoisie(self):
        return sorted(self.populace,
                      key=lambda member: member['fitness'],
                      reverse=True)[:self._num_survivors]

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
        except(IndexError, AssertionError):
            print_exc()
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
