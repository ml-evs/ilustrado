#!/usr/bin/env python
from matador.hull import QueryConvexHull
from matador.similarity.similarity import get_uniq_cursor
from matador.utils.chem_utils import get_formula_from_stoich
from matador.scrapers.castep_scrapers import res2dict
from ilustrado.ilustrado import ArtificialSelector
from glob import glob
from sys import argv

nprocs = int(argv[1])

cursor = [res2dict(res)[0] for res in glob('seed/*.res')]
hull = QueryConvexHull(cursor=cursor, no_plot=True, kpoint_tolerance=0.03,
                       summary=True, hull_cutoff=7.5e-2)
print('Filtering down to only ternary phases... {}'.format(len(hull.hull_cursor)))
hull.hull_cursor = [doc for doc in hull.hull_cursor if len(doc['stoichiometry']) == 3]
print('Filtering unique structures... {}'.format(len(hull.hull_cursor)))
uniq_list, _, _, _ = list(get_uniq_cursor(hull.hull_cursor[1:-1], debug=False))
cursor = [hull.hull_cursor[1:-1][ind] for ind in uniq_list]
print('Final cursor length... {}'.format(len(cursor)))
print('over {} stoichiometries...'.format(len(set([get_formula_from_stoich(doc['stoichiometry']) for doc in cursor]))))
print([doc['stoichiometry'] for doc in cursor])

ArtificialSelector(gene_pool=cursor,
                   seed='KPSn',
                   hull=hull,
                   debug=False,
                   fitness_metric='hull',
                   nodes=['node1', 'node2', 'node15'],
                   ncores=[16, 16, 20],
                   check_dupes=1,
                   verbosity=1,
                   nprocs=nprocs,
                   executable='castep.mpi',
                   relaxer_params=None,
                   best_from_stoich=True,
                   max_num_mutations=3,
                   max_num_atoms=40,
                   mutation_rate=0.5, crossover_rate=0.5,
                   num_generations=10, population=30,
                   num_survivors=20, elitism=0.5,
                   loglevel='debug')
