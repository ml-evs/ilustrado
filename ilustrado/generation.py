""" This file implements the Generation class which
is used to store each generation of structures, and to
evaulate their fitness.
"""

# matador modules
from matador.utils.chem_utils import get_formula_from_stoich
from matador.similarity.pdf_similarity import PDF
# external libraries
# standard library
import json
from traceback import print_exc


class Generation():
    """ Stores each generation of structures.

    Args:
        run_hash           : str, hash for this GA run,
        generation_idx     : int, index of this generation,
        num_survivors      : int, number of structures to aim for per generation,
        num_accepted       : int, number to accept from this generation, i.e. excluding elites,
        populace           : list(dict), initial structures to populate generation with (optional),
        dumpfile           : str, dumpfile name for this generation (optional),
        fitness_calculator : str, fitness metric to use, e.g. 'hull'.

    """

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
        self.populace = [populum for populum in self.populace if
                         (populum['formation_enthalpy_per_atom'] > -3.5 and populum['formation_enthalpy_per_atom'] < 1)]
        return init_len-len(self.populace)

    def set_bourgeoisie(self, elites=None, best_from_stoich=True):
        """ Set the list of structures that will continue to the next generation.

        Args:

            elites            : list of elite structures from previous generations to include,
            best_from_stoich  : bool, whether to include one structure from each stoichiometry.

        """

        # first populate with best precomputd "num_accepted" structures, where "num_accepted"
        # takes into account the number of elites
        self.bourgeoisie = sorted(self.populace,
                                  key=lambda member: member['fitness'],
                                  reverse=True)[:self._num_accepted]

        # find the fittest structure from each stoichiometry sampled
        if best_from_stoich:
            best_from_stoichs = dict()
            for struc in self.populace:
                stoich = get_formula_from_stoich(sorted(struc['stoichiometry']))
                best_from_stoichs[stoich] = {'fitness': 0}
            for struc in self.populace:
                stoich = get_formula_from_stoich(sorted(struc['stoichiometry']))
                if best_from_stoichs[stoich]['fitness'] < struc['fitness']:
                    best_from_stoichs[stoich] = struc

            # if its not already included, add the best structure from this stoichiometry
            # in exchange for the least fit structure already included
            for stoich in best_from_stoichs:
                if best_from_stoichs[stoich] not in self.bourgeoisie:
                    self.bourgeoisie.pop()
                    self.bourgeoisie.insert(0, best_from_stoichs[stoich])

        if elites is not None:
            self.bourgeoisie.extend(elites)

    def calc_pdfs(self):
        """ Compute PDFs of entire generation. """
        self._pdfs = []
        self._stoichs = []
        for structure in self.populace:
            self._pdfs.append(PDF(structure, projected=True))
            self._stoichs.append(sorted(structure['stoichiometry']))

    def is_dupe(self, doc):
        """ Compare current doc with all other structures at same stoichiometry. """
        new_pdf = PDF(doc, projected=True)
        for ind, pdf in enumerate(self.pdfs):
            if sorted(doc['stoichiometry']) == self._stoichs[ind]:
                dist = new_pdf.get_sim_distance(pdf, projected=True)
                if dist < 5e-2:
                    return True
        return False

    @property
    def pdfs(self):
        """ Returns list of PDFs for generation, calculating if necessary. """
        try:
            return self._pdfs
        except(AttributeError, AssertionError):
            self.calc_pdfs()
            return self._pdfs

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
