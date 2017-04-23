#!/usr/bin/env python
import unittest
from os.path import realpath
from os import chdir
from os import uname
from multiprocessing import cpu_count
REAL_PATH = '/'.join(realpath(__file__).split('/')[:-1]) + '/'


def hull_test():
    """ Perform a test run of Ilustrado with a low quality parameter set,
    on all cores of current machine.
    """
    from matador.query import DBQuery
    from matador.hull import QueryConvexHull
    from matador.similarity.similarity import get_uniq_cursor
    from ilustrado.ilustrado import ArtificialSelector

    chdir(REAL_PATH)

    if uname()[1] is 'cluster2':
        cpus = cpu_count() - 2
    else:
        cpus = cpu_count()

    # prepare best structures from hull as gene pool
    query = DBQuery(composition=['KP'], db=['KP_wtf'],
                    kpoint_tolerance=0.03, subcmd='hull', biggest=True)
    hull = QueryConvexHull(query,
                           subcmd='hull', no_plot=True, kpoint_tolerance=0.03,
                           summary=False, hull_cutoff=5e-2)

    uniq_list, _, _, _ = list(get_uniq_cursor(hull.hull_cursor[1:-1], debug=False))
    cursor = [hull.hull_cursor[1:-1][ind] for ind in uniq_list]
    print([doc['num_atoms'] for doc in cursor])
    cursor = [doc for doc in cursor if doc['num_atoms'] < 7]

    print('Running on {} cores on {}.'.format(cpus, uname()[1]))

    ArtificialSelector(gene_pool=cursor,
                       seed='KP',
                       hull=hull,
                       debug=False,
                       fitness_metric='hull',
                       nprocs=cpus,
                       ncores=1,
                       testing=True,
                       mutations=['null_nudge_positions'],
                       max_num_mutations=1,
                       mutation_rate=1, crossover_rate=0,
                       num_generations=10, population=50,
                       num_survivors=10, elitism=0.3,
                       loglevel='debug')


class MatadorHullUnitTest(unittest.TestCase):
    """ Tests matador hull init of ilustrado. """
    def testIlustradoFromHull(self):
        # if main throws an error, so will unit test
        hull_test()


if __name__ == '__main__':
    unittest.main()
