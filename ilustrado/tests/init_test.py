#!/usr/bin/env python
import unittest
from os.path import realpath
from os import uname, remove, chdir
from multiprocessing import cpu_count
from matador.scrapers.castep_scrapers import res2dict
import glob
REAL_PATH = '/'.join(realpath(__file__).split('/')[:-1]) + '/'


def hull_test():
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

    res_files = glob.glob(REAL_PATH + '/data/hull-KP-KSnP_pub/*.res')
    cursor = [res2dict(_file, db=True)[0] for _file in res_files]

    # prepare best structures from hull as gene pool
    hull = QueryConvexHull(cursor=cursor, elements=['K', 'P'],
                           subcmd='hull', no_plot=True, source=True,
                           summary=True, hull_cutoff=0)

    # uniq_list, _, _, _ = list(get_uniq_cursor(hull.hull_cursor[1:-1], debug=False))
    cursor = hull.hull_cursor[1:-1]
    # print([doc['num_atoms'] for doc in cursor])
    # cursor = [doc for doc in cursor if doc['num_atoms'] < 7]

    print('Running on {} cores on {}.'.format(cpus, uname()[1]))

    minsep_dict = {('K', 'K'): 2.5}

    ArtificialSelector(gene_pool=cursor,
                       seed='KP',
                       hull=hull,
                       debug=False,
                       fitness_metric='hull',
                       nprocs=cpus,
                       check_dupes=0,
                       check_dupes_hull=False,
                       sandbagging=True,
                       minsep_dict=minsep_dict,
                       ncores=1,
                       testing=True,
                       mutations=['nudge_positions', 'permute_atoms', 'random_strain', 'vacancy'],
                       max_num_mutations=1,
                       max_num_atoms=50,
                       mutation_rate=0.5, crossover_rate=0.5,
                       num_generations=3, population=15,
                       num_survivors=10, elitism=0.5,
                       loglevel='debug')

    run_hash = glob.glob('*.json')[0].split('-')[0]

    new_life = ArtificialSelector(gene_pool=cursor,
                                  seed='KP',
                                  hull=hull,
                                  debug=False,
                                  fitness_metric='hull',
                                  recover_from=run_hash,
                                  load_only=True,
                                  check_dupes=0,
                                  check_dupes_hull=False,
                                  minsep_dict=minsep_dict,
                                  mutations=['nudge_positions', 'permute_atoms', 'random_strain', 'vacancy'],
                                  sandbagging=True,
                                  nprocs=cpus,
                                  ncores=1,
                                  testing=True,
                                  max_num_mutations=1,
                                  max_num_atoms=50,
                                  mutation_rate=0.5, crossover_rate=0.5,
                                  num_generations=10, population=15,
                                  num_survivors=10, elitism=0.5,
                                  loglevel='debug')
    print(len(new_life.generations))
    print([len(new_life.generations[i]) for i in range(len(new_life.generations))])
    print([len(new_life.generations[i].bourgeoisie) for i in range(len(new_life.generations))])
    assert len(new_life.generations[-1]) >= 15
    assert len(new_life.generations[-1].bourgeoisie) >= 10

    new_life.start()

    [remove(f) for f in glob.glob('*.json')]
    [remove(f) for f in glob.glob('*.log')]
    [remove(f) for f in glob.glob('*.kill')]


class MatadorHullUnitTest(unittest.TestCase):
    """ Tests matador hull init of ilustrado. """
    def testIlustradoFromHull(self):
        # if main throws an error, so will unit test
        hull_test()


if __name__ == '__main__':
    unittest.main()
