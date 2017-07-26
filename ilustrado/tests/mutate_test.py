#!/usr/bin/env python
import unittest
import logging
from os.path import realpath
from copy import deepcopy
from os import chdir
from ilustrado.mutate import voronoi_shuffle
from ilustrado.adapt import check_feasible
from matador.export import doc2res
from matador.utils.chem_utils import get_formula_from_stoich
from json import load, dumps
import numpy as np
REAL_PATH = '/'.join(realpath(__file__).split('/')[:-1]) + '/'


class MutationTest(unittest.TestCase):
    """ Tests basic crossover functionality. """
    def testVoronoi(self):
        chdir(REAL_PATH)
        with open('jdkay7-gen1.json') as f:
            generation = load(f)
        print([get_formula_from_stoich(doc['stoichiometry']) for doc in generation])
        _iter = 0
        loglevel = 'debug'
        numeric_loglevel = getattr(logging, loglevel.upper(), None)
        logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                            filename='test.log',
                            level=numeric_loglevel)
        while _iter < 100:
            mutant = deepcopy(np.random.choice(generation[1:-1]))
            initial_stoich = deepcopy(mutant['stoichiometry'])
            voronoi_shuffle(mutant, debug=False)
            print(get_formula_from_stoich(initial_stoich), '---->', get_formula_from_stoich(mutant['stoichiometry']))
            # feasible = check_feasible(mutant, [generation[3]], max_num_atoms=40, debug=True)
            # self.assertTrue(feasible)
            test_json = dumps(mutant)
            _iter += 1


if __name__ == '__main__':
    unittest.main()
