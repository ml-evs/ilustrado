#!/usr/bin/env python
import unittest
import os
import shutil

from matador.export import doc2res
from matador.scrapers.castep_scrapers import res2dict
from matador.utils.cell_utils import create_simple_supercell

from ilustrado.crossover import random_slice
from ilustrado.adapt import check_feasible

REAL_PATH = "/".join(os.path.realpath(__file__).split("/")[:-1]) + "/"
TMP_DIR = REAL_PATH + "/tmp_test"
ROOT_DIR = os.getcwd()


class CrossoverTest(unittest.TestCase):
    """ Tests basic crossover functionality. """

    def setUp(self):
        os.makedirs(TMP_DIR, exist_ok=True)
        os.chdir(TMP_DIR)

    def tearDown(self):
        os.chdir(ROOT_DIR)
        shutil.rmtree(TMP_DIR)

    def test_slice(self):

        parents = [
            res2dict(REAL_PATH + "/query-K3P-KP/KP-NaP-CollCode182164.res")[0],
            res2dict(REAL_PATH + "/query-K6P-KP/KP-GA-uynlzz-17x7.res")[0],
        ]
        doc2res(parents[0], "parent1", hash_dupe=False, info=False)
        doc2res(parents[1], "parent2", hash_dupe=False, info=False)
        _iter = 0
        while _iter < 100:
            child = random_slice(
                parents, standardize=True, supercell=True, shift=True, debug=True
            )
            feasible = check_feasible(child, parents, max_num_atoms=50)
            _iter += 1
            if feasible:
                doc2res(child, "feasible", hash_dupe=True, info=False)
            else:
                doc2res(child, "failed", hash_dupe=True, info=False)

        parents[1] = create_simple_supercell(parents[1], [3, 1, 1])
        doc2res(parents[1], "supercell2", hash_dupe=False, info=False)


if __name__ == "__main__":
    unittest.main()
