#!/usr/bin/env python
""" This example runs directly from a matador DBQuery. """
from matador.hull import QueryConvexHull
from matador.scrapers import res2dict
from ilustrado.ilustrado import ArtificialSelector

# prepare best structures from hull as gene pool
cursor, failures = res2dict("hull/*.res")
hull = QueryConvexHull(
    cursor=cursor, intersection=True,
    no_plot=True, kpoint_tolerance=0.03,
    summary=True, hull_cutoff=7.5e-2
)


def filter_fn(doc):
    """ Filter out any non-binary phases. """
    return len(doc['stoichiometry']) == 2


print('Filtering down to only ternary phases... {}'.format(len(hull.hull_cursor)))
cursor = [doc for doc in hull.hull_cursor if len(doc['stoichiometry']) == 2]
print('Final cursor length... {}'.format(len(cursor)))

ArtificialSelector(
    gene_pool=cursor,
    compute_mode="manual",
    hull=hull,
    fitness_metric='hull',
    check_dupes=True,
    check_dupes_hull=False,
    structure_filter=filter_fn,
    best_from_stoich=True,
    max_num_mutations=3,
    max_num_atoms=40,
    mutation_rate=0.5, crossover_rate=0.5,
    num_generations=10, population=30,
    num_survivors=20, elitism=0.5,
    loglevel='info'
)
