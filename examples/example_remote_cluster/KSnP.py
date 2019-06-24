#!/usr/bin/env python
""" This example runs from phases found inside the seed folder,
with an appropriate filter applied to choose only ternary phases.
This is suitable for use on a slurm scheduling system.
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

    nprocs = int(argv[1])  # specify nprocs at the command-line

    cursor = [res2dict(res)[0] for res in glob('seed/*.res')]
    hull = QueryConvexHull(cursor=cursor, no_plot=True, kpoint_tolerance=0.03,
                           summary=True, hull_cutoff=1e-1)
    print('Filtering down to only ternary phases... {}'.format(len(hull.hull_cursor)))
    hull.hull_cursor = [doc for doc in hull.hull_cursor if len(doc['stoichiometry']) == 3]
    print('Filtering unique structures... {}'.format(len(hull.hull_cursor)))
    uniq_list, _, _, _ = list(get_uniq_cursor(hull.hull_cursor[1:-1], debug=False))
    cursor = [hull.hull_cursor[1:-1][ind] for ind in uniq_list]
    print('Final cursor length... {}'.format(len(cursor)))
    print('over {} stoichiometries...'.format(len(set([get_formula_from_stoich(doc['stoichiometry'])
                                                       for doc in cursor]))))
    print([doc['stoichiometry'] for doc in cursor])

    def filter_fn(doc):
        """ Filter out any non-ternary phases. """
        return True if len(doc['stoichiometry']) == 3 else False

    relaxer_params = {'bnl': True}  # required to use srun instead of mpirun
    ArtificialSelector(gene_pool=cursor,
                       seed='KPSn',
                       hull=hull,
                       debug=False,
                       fitness_metric='hull',
                       # number of cores per individual calculation
                       # to use less than one node, decrease this to e.g. 10
                       # then increase nprocs to 2*nnodes
                       ncores=20,
                       check_dupes=1,
                       # number of total procs, taken as command-line argument to script
                       nprocs=nprocs,
                       executable='castep',
                       relaxer_params=relaxer_params,
                       structure_filter=filter_fn,
                       best_from_stoich=True,
                       max_num_mutations=3,
                       max_num_atoms=50,
                       mutation_rate=0.4, crossover_rate=0.6,
                       num_generations=20, population=30,
                       num_survivors=20, elitism=0.5,
                       loglevel='debug')

main()
