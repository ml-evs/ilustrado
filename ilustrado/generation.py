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

    def __init__(self, run_hash, generation_idx, num_survivors, num_accepted,
                 populace=None, dumpfile=None, fitness_calculator=None):

        self.populace = []
        if populace is not None:
            self.populace = populace
        self._num_survivors = num_survivors
        self._num_accepted = num_accepted
        self._fitness_calculator = fitness_calculator
        self.run_hash = run_hash
        self.generation_idx = generation_idx
        if dumpfile is not None:
            self.load(dumpfile)

    def __len__(self):
        return len(self.populace)

    def __str__(self):
        gen_string = '\nCompleted generation {}:\n'.format(self.generation_idx)
        gen_string += 'Number of members: {}\n'.format(len(self.populace))
        gen_string += 'Number of survivors: {}\n'.format(len(self.bourgeoisie))
        gen_string += 84*'─' + '\n'
        gen_string += ('{:^10} {:^10} {:^25} {:^35}\n'
                       .format('Formula', 'Fitness', 'Hull distance (eV/atom)', 'ID'))
        gen_string += 84*'─' + '\n'
        for populum in self.populace:
            gen_string += ('{:^10} {: ^10.5f} {:^25.5f} {:^35}\n'
                           .format(get_formula_from_stoich(populum['stoichiometry']),
                                   populum['fitness'], populum['raw_fitness'],
                                   populum['source'][0].split('/')[-1]
                                   .replace('.res', '').replace('.castep', '')))
        gen_string += '\n'
        return gen_string

    def __getitem__(self, key):
        return self.populace[int(key)]

    def __iter__(self):
        return iter(self.populace)

    def dump(self, gen_suffix):
        with open('{}-gen{}.json'.format(self.run_hash, gen_suffix), 'w') as f:
            json.dump(self.populace, f, sort_keys=False, indent=2)

    def load(self, gen_fname):
        with open(gen_fname, mode='r') as f:
            populace = json.load(f)
        self.populace = populace

    def birth(self, populum):
        self.populace.append(populum)

    def rank(self):
        self._fitness_calculator.evaluate(self)

    def clean(self):
        """ Remove structures with pathological formation enthalpies. """
        init_len = len(self.populace)
        self.populace = [populum for populum in self.populace if self.populace['formation_enthalpy_per_atom'] > -3.5]
        return init_len-len(self.populace)

    def set_bourgeoisie(self, elites=None):
        self.bourgeoisie = sorted(self.populace,
                                  key=lambda member: member['fitness'],
                                  reverse=True)[:self._num_accepted]
        if elites is not None:
            self.bourgeoisie.extend(elites)

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
