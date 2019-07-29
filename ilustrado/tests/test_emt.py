#!/usr/bin/env python
import unittest
import os
import shutil
import glob
from multiprocessing import cpu_count

from matador.scrapers import res2dict

ASE_FAILED_IMPORT = False
try:
    from ilustrado.util import AseRelaxation
except ImportError:
    ASE_FAILED_IMPORT = True

REAL_PATH = "/".join(os.path.realpath(__file__).split("/")[:-1]) + "/"
ROOT_DIR = os.getcwd()
TMP_DIR = REAL_PATH + "/tmp_test"


def emt_relax():
    """ Perform a test run of Ilustrado with dummy DFT,
    on all cores of current machine.
    """
    from matador.hull import QueryConvexHull
    from ilustrado.ilustrado import ArtificialSelector

    if os.uname()[1] == "cluster2":
        cpus = cpu_count() - 2
    else:
        cpus = cpu_count()

    structures, _ = res2dict(REAL_PATH + "AuCu_structures/*.res")

    # prepare best structures from hull as gene pool
    hull = QueryConvexHull(
        cursor=structures,
        elements=["Au", "Cu"],
        intersection=True,
        subcmd="hull",
        no_plot=True,
        hull_cutoff=0,
    )

    from multiprocessing import Queue

    cursor = hull.cursor
    for doc in cursor:
        queue = Queue()
        relaxer = AseRelaxation(doc, queue)
        relaxer.relax()
        doc.update(queue.get())

    print("Running on {} cores on {}.".format(cpus, os.uname()[1]))

    def filter_fn(doc):
        return len(doc["stoichiometry"]) == 2

    ArtificialSelector(
        gene_pool=cursor,
        hull=hull,
        debug=False,
        fitness_metric="hull",
        nprocs=cpus,
        structure_filter=filter_fn,
        check_dupes=0,
        check_dupes_hull=False,
        ncores=1,
        testing=True,
        emt=True,
        mutations=["nudge_positions", "permute_atoms", "random_strain", "vacancy"],
        max_num_mutations=1,
        max_num_atoms=50,
        mutation_rate=0.5,
        crossover_rate=0.5,
        num_generations=5,
        population=20,
        num_survivors=10,
        elitism=0.5,
        loglevel="debug",
    )


class EMTAuAgTest(unittest.TestCase):
    """ Tests matador hull init of ilustrado. """

    def setUp(self):
        os.makedirs(TMP_DIR, exist_ok=True)
        os.chdir(TMP_DIR)

    def tearDown(self):
        os.chdir(ROOT_DIR)
        shutil.rmtree(TMP_DIR)

    @unittest.skipIf(ASE_FAILED_IMPORT, "ase not found")
    def testIlustradoFromEMT(self):
        emt_relax()


if __name__ == "__main__":
    unittest.main()
