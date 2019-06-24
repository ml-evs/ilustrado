#!/usr/bin/env python
""" This example runs from phases found inside the seed folder,
with an appropriate filter applied to choose only ternary phases.
It runs jobs through SLURM, this entrypoint will be used repeatedly
to check up on relaxations.
"""


def main():
    """ Run GA. """
    from glob import glob
    from sys import argv
    from matador.hull import QueryConvexHull
    from matador.fingerprints.similarity import get_uniq_cursor
    from matador.utils.chem_utils import get_formula_from_stoich
    from matador.scrapers.castep_scrapers import res2dict
    from ilustrado.ilustrado import ArtificialSelector

    cursor = [res2dict(res)[0] for res in glob('seed/*.res')]
    hull = QueryConvexHull(cursor=cursor, no_plot=True, kpoint_tolerance=0.03,
                           summary=True, hull_cutoff=1e-1)
    print('Filtering down to only ternary phases... {}'.format(len(hull.hull_cursor)))
    hull.hull_cursor = [doc for doc in hull.hull_cursor if len(doc['stoichiometry']) == 3]
    print('Filtering unique structures... {}'.format(len(hull.hull_cursor)))
    # uniq_list, _, _, _ = list(get_uniq_cursor(hull.hull_cursor[1:-1], debug=False))
    # cursor = [hull.hull_cursor[1:-1][ind] for ind in uniq_list]
    cursor = hull.hull_cursor
    print('Final cursor length... {}'.format(len(cursor)))
    print('over {} stoichiometries...'.format(len(set([get_formula_from_stoich(doc['stoichiometry'])
                                                       for doc in cursor]))))
    print([doc['stoichiometry'] for doc in cursor])

    def filter_fn(doc):
        """ Filter out any non-ternary phases. """
        return True if len(doc['stoichiometry']) == 3 else False

    ArtificialSelector(gene_pool=cursor,
                       seed='KPSn',
                       compute_mode='slurm',
                       entrypoint=__file__,
                       walltime_hrs=12,
                       slurm_template='template.slurm',
                       max_num_nodes=100,
                       hull=hull,
                       check_dupes=0,
                       structure_filter=filter_fn,
                       best_from_stoich=True,
                       max_num_mutations=3,
                       max_num_atoms=50,
                       mutation_rate=0.4, crossover_rate=0.6,
                       num_generations=20, population=30,
                       num_survivors=20, elitism=0.5,
                       loglevel='debug')

main()
