#!/usr/bin/env python
import unittest
import glob
from os.path import realpath
from os import uname, remove, chdir
from multiprocessing import cpu_count

from matador.scrapers.castep_scrapers import res2dict

ASE_FAILED_IMPORT = False
try:
    from ilustrado.util import AseRelaxation
except:
    ASE_FAILED_IMPORT = True

REAL_PATH = '/'.join(realpath(__file__).split('/')[:-1]) + '/'


def emt_test():
    """ Perform a test run of Ilustrado with dummy DFT,
    on all cores of current machine.
    """
    from matador.query import DBQuery
    from matador.hull import QueryConvexHull
    from ilustrado.ilustrado import ArtificialSelector

    chdir(REAL_PATH)

    if uname()[1] is 'cluster2':
        cpus = cpu_count() - 2
    else:
        cpus = cpu_count()

    query = DBQuery(composition=['AgAu'], db='oqmd_1.1', subcmd='hull', intersection=True)

    # prepare best structures from hull as gene pool
    hull = QueryConvexHull(query, elements=['Ag', 'Au'], intersection=True,
                           subcmd='hull', no_plot=True, source=True,
                           summary=True, hull_cutoff=0)

    from multiprocessing import Queue
    cursor = hull.cursor
    for doc in cursor:
        queue = Queue()
        relaxer = AseRelaxation(doc)
        relaxer.relax(queue)
        doc.update(queue.get())

    hull = QueryConvexHull(cursor=cursor, elements=['Ag', 'Au'], intersection=True,
                           subcmd='hull', no_plot=True, source=True,
                           summary=True, hull_cutoff=0.05)

    print('Running on {} cores on {}.'.format(cpus, uname()[1]))

    def filter(doc):
        if len(doc['stoichiometry']) == 2:
            return True
        else:
            return False

    cursor = hull.cursor
    cursor = [doc for doc in cursor if filter(doc)]

    ArtificialSelector(gene_pool=cursor,
                       hull=hull,
                       debug=False,
                       fitness_metric='hull',
                       nprocs=cpus,
                       structure_filter=filter,
                       check_dupes=0,
                       check_dupes_hull=False,
                       ncores=1,
                       testing=True,
                       emt=True,
                       mutations=['nudge_positions', 'permute_atoms', 'random_strain', 'vacancy'],
                       max_num_mutations=1,
                       max_num_atoms=50,
                       mutation_rate=0.5, crossover_rate=0.5,
                       num_generations=30, population=50,
                       num_survivors=10, elitism=0.5,
                       loglevel='debug')

    [remove(f) for f in glob.glob('*.json')]
    [remove(f) for f in glob.glob('*.log')]
    [remove(f) for f in glob.glob('*.kill')]


class EMTAuAgTest(unittest.TestCase):
    """ Tests matador hull init of ilustrado. """
    @unittest.skipIf(ASE_FAILED_IMPORT)
    def testIlustradoFromEMT(self):
        # if main throws an error, so will unit test
        emt_test()


if __name__ == '__main__':
    unittest.main()
