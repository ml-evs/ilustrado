#!/usr/bin/env python
import unittest
import logging
from os.path import realpath, isfile
from os import chdir, makedirs
from ilustrado.crossover import crossover, random_slice
from ilustrado.adapt import check_feasible
from matador.export import doc2res
from matador.scrapers.castep_scrapers import res2dict
from matador.utils.cell_utils import create_simple_supercell
from json import load, dumps
import numpy as np
REAL_PATH = '/'.join(realpath(__file__).split('/')[:-1]) + '/'


class CrossoverTest(unittest.TestCase):
    """ Tests basic crossover functionality. """
    # def testRandom(self):
        # chdir(REAL_PATH)
        # with open('jdkay7-gen7.json') as f:
            # generation = load(f)
        # _iter = 0
        # loglevel = 'debug'
        # numeric_loglevel = getattr(logging, loglevel.upper(), None)
        # logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                            # filename='test.log',
                            # level=numeric_loglevel)
        # while _iter < 100:
            # parent = np.random.choice(generation)
            # parent2 = np.random.choice(generation)
            # child = crossover([parent, parent2], debug=True)
            # # doc2res(child, 'failed.res', info=False)
            # _iter += 1

    def testSlice(self):
        chdir(REAL_PATH)
        if not isfile('crossover'):
            makedirs('crossover', exist_ok=True)

        with open('jdkay7-gen7.json') as f:
            generation = load(f)
        parents = [res2dict('query-K3P-KP/KP-NaP-CollCode182164.res')[0], res2dict('query-K6P-KP/KP-GA-uynlzz-17x7.res')[0]]
        doc2res(parents[0], 'crossover/parent1', hash_dupe=False, info=False)
        doc2res(parents[1], 'crossover/parent2', hash_dupe=False, info=False)
        _iter = 0
        while _iter < 100:
            child = random_slice(parents, standardize=True, supercell=True, shift=True, debug=True)
            feasible = check_feasible(child, parents, max_num_atoms=50)
            test_json = dumps(child)
            _iter += 1
            if feasible:
                doc2res(child, 'crossover/feasible', hash_dupe=True, info=False)
            else:
                doc2res(child, 'crossover/failed', hash_dupe=True, info=False)

        parents[1] = create_simple_supercell(parents[1], [3, 1, 1])
        doc2res(parents[1], 'crossover/supercell2', hash_dupe=False, info=False)


if __name__ == '__main__':
    unittest.main()
