#!/usr/bin/env python
import unittest
import os
import shutil
import multiprocessing
import glob
from matador.scrapers.castep_scrapers import res2dict


ROOT_DIR = os.getcwd()
REAL_PATH = "/".join(os.path.realpath(__file__).split("/")[:-1]) + "/"
TEST_DIR = REAL_PATH + "/tmp_test"
NUM_CORES = multiprocessing.cpu_count()

if os.uname()[1] == "cluster2":
    NUM_CORES -= 2


def hull_test(self):
    """ Perform a test run of Ilustrado with dummy DFT,
    on all cores of current machine.
    """
    from matador.hull import QueryConvexHull
    from ilustrado.ilustrado import ArtificialSelector

    res_files = glob.glob(REAL_PATH + "/data/hull-KP-KSnP_pub/*.res")
    cursor = [res2dict(_file, db=True)[0] for _file in res_files]

    # prepare best structures from hull as gene pool
    hull = QueryConvexHull(
        cursor=cursor,
        elements=["K", "P"],
        subcmd="hull",
        no_plot=True,
        source=True,
        summary=True,
        hull_cutoff=0,
    )

    cursor = hull.hull_cursor[1:-1]

    print("Running on {} cores on {}.".format(NUM_CORES, os.uname()[1]))

    minsep_dict = {("K", "K"): 2.5}

    ArtificialSelector(
        gene_pool=cursor,
        hull=hull,
        debug=False,
        fitness_metric="hull",
        nprocs=NUM_CORES,
        check_dupes=0,
        check_dupes_hull=False,
        sandbagging=True,
        minsep_dict=minsep_dict,
        ncores=1,
        testing=True,
        mutations=["nudge_positions", "permute_atoms", "random_strain", "vacancy"],
        max_num_mutations=1,
        max_num_atoms=50,
        mutation_rate=0.5,
        crossover_rate=0.5,
        num_generations=3,
        population=15,
        num_survivors=10,
        elitism=0.5,
        loglevel="debug",
    )

    run_hash = glob.glob("*.json")[0].split("-")[0]

    new_life = ArtificialSelector(
        gene_pool=cursor,
        hull=hull,
        debug=False,
        fitness_metric="hull",
        recover_from=run_hash,
        load_only=True,
        check_dupes=0,
        check_dupes_hull=False,
        minsep_dict=minsep_dict,
        mutations=["nudge_positions", "permute_atoms", "random_strain", "vacancy"],
        sandbagging=True,
        nprocs=NUM_CORES,
        ncores=1,
        testing=True,
        max_num_mutations=1,
        max_num_atoms=50,
        mutation_rate=0.5,
        crossover_rate=0.5,
        num_generations=10,
        population=15,
        num_survivors=10,
        elitism=0.5,
        loglevel="debug",
    )
    self.assertTrue(len(new_life.generations[-1]) >= 15)
    self.assertTrue(len(new_life.generations[-1].bourgeoisie) >= 10)

    new_life.start()

    self.assertTrue(os.path.isdir(new_life.run_hash + "-results"))
    num_structures = len(glob.glob(new_life.run_hash + "-results/*.res"))
    self.assertTrue(num_structures > 5)


class MatadorHullUnitTest(unittest.TestCase):
    """ Tests matador hull init of ilustrado. """

    def setUp(self):
        if os.path.isdir(TEST_DIR):
            shutil.rmtree(TEST_DIR)
        os.makedirs(TEST_DIR, exist_ok=True)
        os.chdir(TEST_DIR)

    def tearDown(self):
        os.chdir(ROOT_DIR)
        shutil.rmtree(TEST_DIR)

    def testIlustradoFromHull(self):
        # if main throws an error, so will unit test
        print(os.getcwd())
        hull_test(self)


if __name__ == "__main__":
    unittest.main()
