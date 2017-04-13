#!/usr/bin/env python
import unittest
from os.path import realpath
from os import chdir
from ilustrado.crossover import crossover
from json import load
import numpy as np
REAL_PATH = '/'.join(realpath(__file__).split('/')[:-1]) + '/'


class CrossOverTest(unittest.TestCase):
    """ Tests basic crossover functionality. """
    def testClone(self):
        chdir(REAL_PATH)
        with open('jdkay7-gen7.json') as f:
            generation = load(f)
        parent = generation[3]
        child = crossover([parent, parent], debug=True)
        np.testing.assert_allclose(child['lattice_cart'], parent['lattice_cart'], rtol=1e-5)
        np.testing.assert_allclose(child['positions_frac'], parent['positions_frac'], rtol=1e-5)
        self.assertEqual(child['atom_types'], parent['atom_types'])


if __name__ == '__main__':
    unittest.main()
