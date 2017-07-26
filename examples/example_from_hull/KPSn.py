#!/usr/bin/env python
from matador.query import DBQuery
from matador.hull import QueryConvexHull
from matador.similarity.similarity import get_uniq_cursor
from ilustrado.ilustrado import ArtificialSelector

# prepare best structures from hull as gene pool
query = DBQuery(composition=['KPSn'], db=['KP'], cutoff=[300, 301], intersection=True,
                kpoint_tolerance=0.03, subcmd='hull', biggest=True)
hull = QueryConvexHull(query, intersection=True,
                       subcmd='hull', no_plot=True, kpoint_tolerance=0.03,
                       summary=True, hull_cutoff=7.5e-2)

print('Filtering down to only ternary phases... {}'.format(len(hull.hull_cursor)))
hull.hull_cursor = [doc for doc in hull.hull_cursor if len(doc['stoichiometry']) == 3]
print('Filtering unique structures... {}'.format(len(hull.hull_cursor)))
uniq_list, _, _, _ = list(get_uniq_cursor(hull.hull_cursor[1:-1], debug=False))
cursor = [hull.hull_cursor[1:-1][ind] for ind in uniq_list]
print('Final cursor length... {}'.format(len(cursor)))

ArtificialSelector(gene_pool=cursor,
                   seed='KPSn',
                   hull=hull,
                   debug=False,
                   fitness_metric='hull',
                   monitor=False,
                   nodes=['node1', 'node2', 'node3', 'node4', 'node5', 'node6', 'node20', 'node21'],
                   ncores=[16 for val in range(6)]+[20 for val in range(2)],
                   check_dupes=1,
                   verbosity=0,
                   best_from_stoich=True,
                   max_num_mutations=3,
                   max_num_atoms=40,
                   mutation_rate=0.5, crossover_rate=0.5,
                   num_generations=10, population=30,
                   num_survivors=20, elitism=0.5,
                   loglevel='debug')
